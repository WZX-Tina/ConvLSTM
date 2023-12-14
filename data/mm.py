import numpy as np
import os
from PIL import Image
import torch
import torch.utils.data as data
import json

root_path = '/scratch/tw2672/1008_Project'

def load_videos(parent_folder):
    # Load images
    video_arrays = []
    dir_arrays = []
    print("Current Directory:", parent_folder)
    if parent_folder.endswith(('hidden')): # hidden
        for dirs in os.listdir(parent_folder):
            print(dirs)
            dir_arrays.append(dirs)
            video = []
            if dirs != '.DS_Store':
                for filename in os.listdir(os.path.join(parent_folder,dirs)):
                    if filename.endswith(('.png')):
                        image_path = os.path.join(parent_folder,dirs,filename)
                        image = Image.open(image_path)
                        image_array = np.array(image)
                        video.append(image_array)
                video_arrays.append(video)
        video_arrays = np.array(video_arrays)
        print('hidden shape', video_arrays.shape)
        np.save(root_path+'/video_arrays_hidden.npy', video_arrays)
        with open(root_path+'/dir_arrays_hidden.json','w') as f:
            json.dump(dir_arrays,f)
    elif parent_folder.endswith(('val')): # val
        for dirs in os.listdir(parent_folder):
            print(dirs)
            dir_arrays.append(dirs)
            video = []
            if dirs != '.DS_Store':
                for filename in os.listdir(os.path.join(parent_folder,dirs)):
                    if filename.endswith(('.png')):
                        image_path = os.path.join(parent_folder,dirs,filename)
                        image = Image.open(image_path)
                        image_array = np.array(image)
                        video.append(image_array)
                video_arrays.append(video)
        video_arrays = np.array(video_arrays)
        print('val shape', video_arrays.shape)
        np.save(root_path+'/video_arrays_val.npy', video_arrays)
        with open(root_path+'/dir_arrays_val.json','w') as f:
            json.dump(dir_arrays,f)
    else: # train
        parent_folder_tmp = parent_folder+'/train'
        for dirs in os.listdir(parent_folder_tmp):
            print(dirs)
            dir_arrays.append(dirs)
            video = []
            if dirs != '.DS_Store':
                for filename in os.listdir(os.path.join(parent_folder_tmp,dirs)):
                    if filename.endswith(('.png')):
                        image_path = os.path.join(parent_folder_tmp,dirs,filename)
                        image = Image.open(image_path)
                        image_array = np.array(image)
                        video.append(image_array)
                video_arrays.append(video)
        video_arrays = np.array(video_arrays)
        print('train shape', video_arrays.shape)
        np.save(root_path+'/video_arrays_train.npy', video_arrays)
        with open(root_path+'/dir_arrays_train.json','w') as f:
            json.dump(dir_arrays,f)
    return video_arrays, dir_arrays

# load_videos('../../train_data/train')
# shape: (1000, 22, 160, 240, 3)

class MovingFrame(data.Dataset):
    def __init__(self, root, is_train, is_val, n_frames_input, n_frames_output,
                 transform=None):
        
        super(MovingFrame, self).__init__()

        self.dataset = None
        if is_train: # train+unlabeled
            if os.path.exists(root_path+'/video_arrays_train.npy') and os.path.exists(root_path+'/video_arrays_unlabeled.npy') and os.path.exists(root_path+'/video_arrays_unlabeled_1.npy') and os.path.exists(root_path+'/video_arrays_unlabeled_2.npy'):
                print('loading npy')
                dataset_train = np.load(root_path+'/video_arrays_train.npy')
                dataset_unlabeled_0, dataset_unlabeled_1, dataset_unlabeled_2 = np.load(root_path+'/video_arrays_unlabeled.npy'), np.load(root_path+'/video_arrays_unlabeled_1.npy'), np.load(root_path+'/video_arrays_unlabeled_2.npy')
                with open(root_path+'/dir_arrays_train.json','r') as f:
                    video_list_train = json.load(f)
                with open(root_path+'/dir_arrays_unlabeled.json','r') as f:
                    video_list_unlabeled_0 = json.load(f)
                with open(root_path+'/dir_arrays_unlabeled_1.json','r') as f:
                    video_list_unlabeled_1 = json.load(f)
                with open(root_path+'/dir_arrays_unlabeled_2.json','r') as f:
                    video_list_unlabeled_2 = json.load(f)
                self.dataset = np.concatenate((dataset_train, dataset_unlabeled_0, dataset_unlabeled_1, dataset_unlabeled_2), axis=0)
                self.video_list = video_list_train + video_list_unlabeled_0 + video_list_unlabeled_1 + video_list_unlabeled_2
            elif os.path.exists(root_path+'/video_arrays_train.npy'):
                print('loading npy')
                self.dataset = np.load(root_path+'/video_arrays_train.npy')
                with open(root_path+'/dir_arrays_train.json','r') as f:
                    self.video_list = json.load(f)
            else:
                print('loading images')
                self.dataset, self.video_list = load_videos(root)
        elif is_val: # val
            if os.path.exists(root_path+'/video_arrays_val.npy'):
                print('loading npy')
                self.dataset = np.load(root_path+'/video_arrays_val.npy')
                with open(root_path+'/dir_arrays_val.json','r') as f:
                    self.video_list = json.load(f)
            else:
                print('loading images')
                self.dataset, self.video_list = load_videos(root)
        else: # hidden
            if os.path.exists(root_path+'/video_arrays_hidden.npy'):
                print('loading npy')
                self.dataset = np.load(root_path+'/video_arrays_hidden.npy')
                with open(root_path+'/dir_arrays_hidden.json','r') as f:
                    self.video_list = json.load(f)
            else:
                print('loading images')
                self.dataset, self.video_list = load_videos(root)
        self.length = self.dataset.shape[0]

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
        # print(images.shape)

        # if self.transform is not None:
        #     images = self.transform(images)

        # r = 1
        # w = int(64 / r)
        images = images.transpose(0, 3, 1, 2)

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = np.array([])

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
