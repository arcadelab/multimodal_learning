import numpy as np
import cv2
import os
import pickle
import sys
from tqdm import tqdm
from typing import Tuple

class computeOpticalFlow:
    def __init__(self, source_directory: str, resized_video_directory: str = 'None', destination_directory: str = './optical_flows', resize_dim: Tuple[int, int] = (320, 240)) -> None:
        self.source_directory = source_directory
        self.destination_directory = destination_directory
        self.resized_video_directory = resized_video_directory
        
        if not os.path.exists(self.destination_directory):
            os.makedirs(self.destination_directory)

        self.source_listdir = os.listdir(self.source_directory)
        self.source_listdir = list(filter(lambda x: '.DS_Store' not in x, self.source_listdir))
        self.source_listdir.sort()
        self.resize_dim = resize_dim

        self.resized_listdir = None

    def get_optical_frame(self):
        for video in self.resized_listdir:
            print('Processing video {}'.format(video))
            curr_video_path = os.path.join(self.resized_video_directory, video)
            vidcap = cv2.VideoCapture(curr_video_path)

            success, image = vidcap.read()
            
            prvs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            m, n = prvs.shape
            # prvs = cv2.resize(prvs, (int(n), int(m)))
            prvs = cv2.resize(prvs, self.resize_dim)
            temp_optical_flow_frames = []

            count = 1

            while success:
                success, image = vidcap.read()
                try:
                    _next = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _next = cv2.resize(_next, self.resize_dim)
                    flow = cv2.calcOpticalFlowFarneback(prvs, _next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    temp_optical_flow_frames.append(flow)
                    prvs = _next
                    count += 1
                except:
                    pass
                if count % 100 == 0:
                    print(count)

            print('Saving file')
            save_path = os.path.join(self.destination_directory, video.split('.')[0] + '.p')
            pickle.dump(temp_optical_flow_frames, open(save_path, 'wb'))

    def rescale_video(self) -> None:
        if not os.path.exists(self.resized_video_directory):
            os.makedirs(self.resized_video_directory)
        
        for video in self.source_listdir:
            print('Processing video {}'.format(video))
            curr_video_path = os.path.join(self.source_directory, video)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            vidcap = cv2.VideoCapture(curr_video_path)
            output_path = os.path.join(self.resized_video_directory, video.split('.')[0] + '_resized.avi')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, self.resize_dim)

            success, image = vidcap.read()

            count = 1

            while success:
                success, image = vidcap.read()
                try:
                    resized_image = cv2.resize(image, resize_dim)
                    out.write(resized_image)
                    count += 1
                except:
                    pass
                
                if count % 100 == 0:
                    print(count)

            vidcap.release()
            out.release()

    def run(self) -> None:
        if self.resized_video_directory == 'None':
            self.resized_listdir = self.source_listdir
            self.resized_video_directory = self.source_directory
            print('Generating optical flow.')
            self.get_optical_frame()
        else:
            print('Rescaling videos.')
            self.rescale_video()
            self.resized_listdir = os.listdir(self.resized_video_directory)
            self.resized_listdir = list(filter(lambda x: '.DS_Store' not in x, self.resized_listdir))
            self.resized_listdir.sort()
            print('Generating optical flow.')
            self.get_optical_frame()


def main():
    source_directory = '../jigsaw_dataset/Surgeon_study_videos/videos'
    # resized_video_directory = '../jigsaw_dataset/Surgeon_study_videos/resized_videos'
    resized_video_directory = 'None'
    destination_directory = '../jigsaw_dataset/Surgeon_study_videos/optical_flow'
    resize_dim = (320, 240)

    optical_flow_compute = computeOpticalFlow(source_directory = source_directory, resized_video_directory = resized_video_directory, destination_directory = destination_directory, resize_dim = resize_dim)
    optical_flow_compute.run()
    print(optical_flow_compute.source_listdir)

if __name__ == '__main__':
    main()
