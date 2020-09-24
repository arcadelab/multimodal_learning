import numpy as np
import torch
import os
import pickle
from typing import Tuple, List
from torch.utils.data.dataloader import default_collate

def create_data_blobs(optical_flow_folder_path: str, transcriptions_folder_path: str, kinematics_folder_path: str, num_frames_per_blob: int, blobs_save_folder_path: str, spacing: int) -> None:
    if not os.path.exists(blobs_save_folder_path):
        os.makedirs(blobs_save_folder_path)
    
    blob_count = 0

    for file in os.listdir(transcriptions_folder_path):
        try:
            curr_file_path = os.path.join(transcriptions_folder_path, file)

            print('Processing file: {}'.format(curr_file_path.split('/')[-1]))

            curr_optical_flow_file = '_'.join([file.split('.')[0], 'capture1_resized.p'])
            curr_optical_flow_file = os.path.join(optical_flow_folder_path, curr_optical_flow_file)
            optical_flow_file = pickle.load(open(curr_optical_flow_file, 'rb'))

            curr_kinematics_file = '.'.join([file.split('.')[0], 'txt'])
            curr_kinematics_file = os.path.join(kinematics_folder_path, curr_kinematics_file)
            kinematics_list = []

            with open (curr_kinematics_file) as kf:
                for line in kf:
                    kinematics_list.append([float(v) for v in line.strip('\n').strip().split('     ')])
                kf.close()

            with open(curr_file_path, 'r') as f:
                for line in f:
                    line = line.strip('\n').strip()
                    line = line.split(' ')
                    start = int(line[0])
                    end = int(line[1])
                    gesture = line[2]
                    curr_blob = [torch.tensor(v) for v in optical_flow_file[start: start + spacing*num_frames_per_blob : spacing]]
                    curr_blob = torch.cat(curr_blob, dim = 2).permute(2, 0, 1)
                    curr_kinematics_blob = [torch.tensor(v).view(1, 76) for v in kinematics_list[start: start + spacing*num_frames_per_blob: spacing]]
                    curr_kinematics_blob = torch.stack(curr_kinematics_blob, dim = 0)
                    save_tuple = (curr_blob, curr_kinematics_blob)
                    curr_blob_save_path = 'blob_' + str(blob_count) + '_video_' + curr_file_path.split('/')[-1].split('.')[0].split('_')[-1] + '_gesture_' + gesture + '.p'
                    curr_blob_save_path = os.path.join(blobs_save_folder_path, curr_blob_save_path)
                    pickle.dump(save_tuple, open(curr_blob_save_path, 'wb'))

                    blob_count += 1
        except:
            pass

class gestureBlobDataset:
    def __init__(self, blobs_folder_path: str) -> None:
        self.blobs_folder_path = blobs_folder_path
        self.blobs_folder = os.listdir(self.blobs_folder_path)
        self.blobs_folder = list(filter(lambda x: '.DS_Store' not in x, self.blobs_folder))
        self.blobs_folder.sort(key = lambda x: int(x.split('_')[1]))

    def __len__(self) -> int:
        return(len(self.blobs_folder))

    def __getitem__(self, idx: int) -> torch.Tensor:
        curr_file_path = self.blobs_folder[idx]
        curr_file_path = os.path.join(self.blobs_folder_path, curr_file_path)
        curr_tensor_tuple = pickle.load(open(curr_file_path, 'rb'))
        # print(curr_tensor_tuple[0].size())
        if curr_tensor_tuple[0].size()[0] == 50:
            return(curr_tensor_tuple)
        else:
            return(None)

class gestureBlobBatchDataset:
    def __init__(self, gesture_dataset: gestureBlobDataset, random_tensor: str = 'random') -> None:
        self.gesture_dataset = gesture_dataset
        self.random_tensor = random_tensor
    
    def __len__(self) -> None:
        return(len(self.gesture_dataset))
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        curr_tensor = self.gesture_dataset.__getitem__(idx)
        
        if self.random_tensor == 'random':
            rand_idx = np.random.randint(low = 0, high = len(self.gesture_dataset))
        
        elif self.random_tensor == 'next':
            if idx != len(self.gesture_dataset) - 1:
                rand_idx = idx + 1
            else:
                rand_idx = idx - 1
        else:
            raise ValueError('Value of random_tensor should be "random" or "next".')
        
        random_tensor = self.gesture_dataset.__getitem__(rand_idx)

        y_match = torch.tensor([1, 0], dtype = torch.float32).view(1, 2)
        if idx != rand_idx:
            y_rand = torch.tensor([0, 1], dtype = torch.float32).view(1, 2)
        else:
            y_rand = torch.tensor([1, 0], dtype = torch.float32).view(1, 2)

        return((curr_tensor, random_tensor, y_match, y_rand))

class gestureBlobMultiDataset:
    def __init__(self, blobs_folder_paths_list: List[str]) -> None:
        self.blobs_folder_paths_list = blobs_folder_paths_list
        self.blobs_folder_dict = {path: [] for path in self.blobs_folder_paths_list}
        for path in self.blobs_folder_paths_list:
            self.blobs_folder_dict[path] = os.listdir(path)
            self.blobs_folder_dict[path] = list(filter(lambda x: '.DS_Store' not in x, self.blobs_folder_dict[path]))
            self.blobs_folder_dict[path].sort(key = lambda x: int(x.split('_')[1]))
        
        self.dir_lengths = [len(os.listdir(path)) for path in self.blobs_folder_paths_list]
        for i in range(1, len(self.dir_lengths)):
            self.dir_lengths[i] += self.dir_lengths[i - 1]

    def __len__(self) -> int:
        return(self.dir_lengths[-1])

    def __getitem__(self, idx: int) -> torch.Tensor:
        dir_idx = 0
        while idx >= self.dir_lengths[dir_idx]:
            dir_idx += 1
        adjusted_idx = idx - self.dir_lengths[dir_idx]
        path = self.blobs_folder_paths_list[dir_idx]

        curr_file_path = self.blobs_folder_dict[path][adjusted_idx]
        curr_file_path = os.path.join(path, curr_file_path)
        curr_tensor_tuple = pickle.load(open(curr_file_path, 'rb'))
        # print(curr_tensor_tuple[0].size())
        if curr_tensor_tuple[0].size()[0] == 50:
            return(curr_tensor_tuple)
        else:
            return(None)

def size_collate_fn(batch: torch.Tensor) -> torch.Tensor:
    batch = list(filter(lambda x: x is not None, batch))
    return(default_collate(batch))

def main():
    optical_flow_folder_path = '../jigsaw_dataset/Knot_Tying/optical_flow/'
    transcriptions_folder_path = '../jigsaw_dataset/Knot_Tying/transcriptions'
    num_frames_per_blob = 25
    blobs_save_folder_path = '../jigsaw_dataset/Knot_Tying/blobs'
    spacing = 2
    kinematics_folder_path = '../jigsaw_dataset/Knot_Tying/kinematics/AllGestures/'

    # create_data_blobs(optical_flow_folder_path = optical_flow_folder_path, transcriptions_folder_path = transcriptions_folder_path, kinematics_folder_path = kinematics_folder_path, num_frames_per_blob = num_frames_per_blob, blobs_save_folder_path = blobs_save_folder_path, spacing = spacing)

    blobs_folder_paths_list = ['../jigsaw_dataset/Knot_Tying/blobs/', '../jigsaw_dataset/Needle_Passing/blobs/', '../jigsaw_dataset/Suturing/blobs/']
    # dataset = gestureBlobDataset(blobs_folder_path = '../jigsaw_dataset/Knot_Tying/blobs/')
    dataset = gestureBlobMultiDataset(blobs_folder_paths_list = blobs_folder_paths_list)
    out = dataset.__getitem__(3)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()



