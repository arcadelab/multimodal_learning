import numpy as np
import torch
from typing import Tuple
import torch.nn as nn

class ConvNetStream(torch.nn.Module):
    def __init__(self, optical_flow_stream = False, out_features = 512) -> None:
        super().__init__()
        if not optical_flow_stream:
            self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 96, kernel_size = 5, stride = 2)
        else:
            self.conv1 = torch.nn.Conv2d(in_channels = 2*25, out_channels = 96, kernel_size = 5, stride = 2)
        self.conv2 = torch.nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 3, stride = 2)
        self.conv3 = torch.nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1)
        self.conv4 = torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1)
        self.conv5 = torch.nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1)

        self.linear1 = torch.nn.Linear(in_features = 512*2*3 , out_features = 4096)
        self.linear2 = torch.nn.Linear(in_features = 4096, out_features = out_features)
        self.dropout = torch.nn.Dropout(p = 0.5)

        self.pool = torch.nn.MaxPool2d(2)

        self.softmax = torch.nn.Softmax(dim=1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        #print('Shape of output after conv {} is {}'.format(1, x.size()))
        x = self.conv2(x)
        x = self.pool(x)
        #print('Shape of output after conv {} is {}'.format(2, x.size()))
        x = self.conv3(x)
        x = self.pool(x)
        #print('Shape of output after conv {} is {}'.format(3, x.size()))
        x = self.conv4(x)
        x = self.pool(x)
        #print(x.size())
        #print('Shape of output after conv {} is {}'.format(4, x.size()))
        #x = self.conv5(x)
        #x = self.pool(x)
        #print('Shape of output after conv {} is {}'.format(5, x.size()))

        x = x.view(-1, 512*x.size()[2]*x.size()[3])
        x = self.linear1(x)
        x = self.relu(x)
        #x = self.dropout(x)

        x = self.linear2(x)
        # # x = self.dropout(x)

        # x = self.softmax(x)
        return(x)

class twoStreamNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.spatial_net_stream = ConvNetStream(optical_flow_stream = False)
        self.temporal_net_stream = ConvNetStream(optical_flow_stream = True)

        self.linear1 = torch.nn.Linear(in_features = 2*2048, out_features = 512)
        #self.batch_norm = torch.nn.BatchNorm1d(512)
        self.linear2 = torch.nn.Linear(in_features = 512, out_features = 15)
        self.softmax = torch.nn.Softmax(dim = 1)
    
    def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
        x1 = self.spatial_net_stream(x[0]) 
        x2 = self.temporal_net_stream(x[1])
        # print(x[0[.size())

        x_net = torch.cat((x1, x2),dim= 1)
        x_net = self.linear1(x_net)

        x_net = self.linear2(x_net)

        x_net = self.softmax(x_net)
        return(x_net)

class kinematics_parser(torch.nn.Module):
    def __init__(self, out_features: int, parser: str = 'fcn'):
        super().__init__()
        self.parser = parser
        if self.parser == 'fcn':
            self.fcn1 = torch.nn.Linear(25*76, 512)
            self.relu = torch.nn.ReLU()
            self.fcn2 = torch.nn.Linear(512,out_features)
        elif self.parser == 'cnn': 
            self.conv1d = torch.nn.Conv1d(in_channels = 50, out_channels = 250, kernel_size = 10)
            self.conv1d_2 = torch.nn.Conv1d(in_channels = 250, out_channels = 10, kernel_size = 5)
            self.fcn = torch.nn.Linear(10*63, out_features)

    def forward(self, x):
        if self.parser == 'fcn':
            x = self.fcn1(x)
            x = self.relu(x)
            x = self.fcn2(x)
        elif self.parser == 'cnn':
            x = x.view(-1, 50, 76)
            x = self.conv1d(x)

            x = self.conv1d_2(x)
            x = x.view(-1, 10*63)
            x = self.fcn(x)
            # print(x.size())
        return(x)

class multiModalRepresentation_diff(torch.nn.Module):
    def __init__(self, out_features: int, lstm_num_layers: int, parser: str = 'fcn') -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size = 76, hidden_size = out_features, num_layers = lstm_num_layers)
        self.optical_flow_net = ConvNetStream(optical_flow_stream= True, out_features = out_features)
        self.attention = torch.nn.Linear(5*out_features, out_features)
        self.linear = torch.nn.Linear(1, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.softmax = torch.nn.Softmax(dim = 1)
        self.out_features = out_features
        self.kinematics_parser = kinematics_parser(out_features = out_features, parser = parser)
        self.final_linear = torch.nn.Linear(2*out_features, 2)
        self.cos_sim = torch.nn.CosineSimilarity(dim = 1)
        self.relu = torch.nn.ReLU()
        self.bn_opt = torch.nn.BatchNorm1d(out_features)
        self.bn_kin = torch.nn.BatchNorm1d(out_features)
    
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_opt = x[0]
        x_kin = x[1]
        x_opt = self.optical_flow_net(x_opt)
        x_opt = torch.nn.functional.normalize(x_opt, dim = 1)
        # x_kin = x_kin.view(-1, 25, 76) # Resize kinematics tensor for 1-D CNN
        x_kin = self.kinematics_parser(x_kin)
        x_kin = torch.nn.functional.normalize(x_kin, dim = 1)
        x_final = torch.sum((x_opt - x_kin)**2, dim = 1)
        # x_final2 = self.cos_sim(x_opt, x_kin)
        # import pdb; pdb.set_trace()
        # x_final = torch.stack([x_final, x_final2])
        x_final = x_final.view(-1, 1)
        x_final = self.linear(x_final)
        # x_final = self.relu(x_final)
        # x_final = self.linear2(x_final)
        x_final = self.softmax(x_final)
        return(x_final)

class multiModalRepresentation(torch.nn.Module):
    def __init__(self, out_features: int, lstm_num_layers: int, parser: str = 'fcn') -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size = 76, hidden_size = out_features, num_layers = lstm_num_layers)
        self.optical_flow_net = ConvNetStream(optical_flow_stream= True, out_features = out_features)
        self.attention = torch.nn.Linear(5*out_features, out_features)
        self.linear = torch.nn.Linear(1, 2)
        self.softmax = torch.nn.Softmax(dim = 1)
        self.out_features = out_features
        self.kinematics_parser = kinematics_parser(out_features = out_features, parser = parser)
        self.final_linear = torch.nn.Linear(2*out_features, 256)
        self.relu = torch.nn.ReLU()
        self.final_linear2 = torch.nn.Linear(256, 2)
        self.dropout = torch.nn.Dropout(p = 0.5)
    
    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_opt = x[0]
        x_kin = x[1]
        x_opt = self.optical_flow_net(x_opt)
        x_opt = torch.nn.functional.normalize(x_opt, dim = 1)
        # x_kin, _ = self.lstm(x_kin)
        # # x_kin = x_kin[-1, :, :].view(-1, 5*self.out_features)
        # x_kin = x_kin.view(-1, 5*self.out_features)
        # x_kin = self.attention(x_kin)
        # x_kin = x_kin.view(-1, 5*76) # Resize kinematics tensor for FCN
        x_kin = x_kin.view(-1, 5, 76) # Resize kinematics tensor for 1-D CNN
        print(x_kin.size())
        x_kin = self.kinematics_parser(x_kin)
        x_kin = torch.nn.functional.normalize(x_kin, dim = 1)
        # x_final = torch.sum((x_opt - x_kin)**2, dim = 1)
        # print('x_kin size: {}'.format(x_kin.size()))
        # print('x_opt size: {}'.format(x_opt.size()))
        # temp = torch.sum((x_opt - x_kin)**2, dim = 1)
        temp = torch.nn.functional.cosine_similarity(x_opt, x_kin)
        print('Temp size: {}'.format(temp.size())) 
        print('Embeddings difference norm: {}'.format(temp))
       
        x_final = torch.cat((x_opt, x_kin))
        x_final = x_final.view(-1, 2*self.out_features)

        x_final = self.dropout(x_final)
        # x_final = x_final.view(-1, 1)
        # x_final = self.linear(x_final)
        x_final = self.final_linear(x_final)
        x_final = self.relu(x_final)
        x_final = self.final_linear2(x_final)
        x_final = self.softmax(x_final)

        return(x_final)

class encoderDecoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.conv_net_stream = ConvNetStream(optical_flow_stream = True, out_features = embedding_dim)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(embedding_dim, 128), 
        nn.ReLU(), 
        torch.nn.Linear(128, 1024), 
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024, 4096), 
        nn.BatchNorm1d(4096), 
        nn.ReLU(),
        nn.Linear(4096, 25*76))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_net_stream(x)
        x = self.decoder(x)
        x = x.view(-1, 25, 1, 76)
        return(x)

