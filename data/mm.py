import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data

def load_videos(parent_folder):
    # Load image.
    video_arrays = []
    print("Current Directory:", os.getcwd())
    if parent_folder.endswith(('val')):
        for dirs in os.listdir(parent_folder):
            print(dirs)
            video = []
            print(os.path.join(parent_folder,dirs))
            if dirs != '.DS_Store':
                for filename in os.listdir(os.path.join(parent_folder,dirs)):
                    if filename.endswith(('.png')):
                        image_path = os.path.join(parent_folder,dirs,filename)
                        image = Image.open(image_path)
                        image_array = np.array(image)
                        video.append(image_array)
                video_arrays.append(video)
        video_arrays = np.array(video_arrays)
        print(video_arrays.shape)
        np.save('video_arrays_val.npy', video_arrays)
    else:
        parent_folder_tmp = parent_folder+'/train'
        for dirs in os.listdir(parent_folder_tmp):
            print(dirs)
            video = []
            print(os.path.join(parent_folder,dirs))
            if dirs != '.DS_Store':
                for filename in os.listdir(os.path.join(parent_folder_tmp,dirs)):
                    if filename.endswith(('.png')):
                        image_path = os.path.join(parent_folder_tmp,dirs,filename)
                        image = Image.open(image_path)
                        image_array = np.array(image)
                        video.append(image_array)
                video_arrays.append(video)
#        parent_folder_tmp = parent_folder+'/unlabeled'
#        for dirs in os.listdir(parent_folder_tmp):
#            print(dirs)
#            video = []
#            print(os.path.join(parent_folder,dirs))
#            if dirs != '.DS_Store':
#                for filename in os.listdir(os.path.join(parent_folder_tmp,dirs)):
#                    if filename.endswith(('.png')):
#                        image_path = os.path.join(parent_folder_tmp,dirs,filename)
#                        image = Image.open(image_path)
#                        image_array = np.array(image)
#                        video.append(image_array)
#                video_arrays.append(video)
        video_arrays = np.array(video_arrays)
        print(video_arrays.shape)
        np.save('video_arrays_train.npy', video_arrays)
    return video_arrays

# load_videos('../../train_data/train')
# shape: (1000, 22, 160, 240, 3)

class MovingFrame(data.Dataset):
    def __init__(self, root, is_train, n_frames_input, n_frames_output,
                 transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingFrame, self).__init__()

        self.dataset = None
        if is_train:
            if os.path.exists('./video_arrays_train.npy'):
                self.dataset = np.load('video_arrays_train.npy')
            else:
                self.dataset = load_videos(root)
        else:
            if os.path.exists('./video_arrays_val.npy'):
                self.dataset = np.load('video_arrays_val.npy')
            else:
                self.dataset = load_videos(root)
        self.length = self.dataset.shape[1]

        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        # self.image_size_ = 64
        # self.digit_size_ = 28
        # self.step_length_ = 0.1

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        
        images = self.dataset[idx, ...]

        # if self.transform is not None:
        #     images = self.transform(images)

        # r = 1
        # w = int(64 / r)
        images = images.transpose(0, 3, 1, 2)

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        frozen = input[-1]
        # add a wall to input data
        # pad = np.zeros_like(input[:, 0])
        # pad[:, 0] = 1
        # pad[:, pad.shape[1] - 1] = 1
        # pad[:, :, 0] = 1
        # pad[:, :, pad.shape[2] - 1] = 1
        #
        # input = np.concatenate((input, np.expand_dims(pad, 1)), 1)

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()
        # print()
        # print(input.size())
        # print(output.size())

        out = [idx, output, input, frozen, np.zeros(1)]
        return out

    def __len__(self):
        return self.length
