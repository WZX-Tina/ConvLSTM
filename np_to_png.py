import numpy as np
from PIL import Image
from data.mm import MovingFrame
import os

parent_dir = "/scratch/tw2672/1008_Project/result/hidden"
root_path = '/scratch/tw2672/1008_Project'

result_array = np.load(root_path+'/video_arrays_result_hidden.npy')
print(result_array.shape)

hiddenFolder = MovingFrame(is_train=False,
                          is_val=False,
                          root=root_path+'/data/hidden',
                          n_frames_input=11,
                          n_frames_output=0,
                          )
if len(hiddenFolder.video_list) != result_array.shape[0]:
    hiddenFolder.video_list.remove('.DS_Store') 

for i in range(len(hiddenFolder.video_list)):
    path = os.path.join(parent_dir,hiddenFolder.video_list[i])
    if not os.path.exists(path):
        os.mkdir(path)
    # for j in range(11):
    #     pred_image = Image.fromarray(result_array[i,j])
    #     pred_image.save('../data/result/'+hiddenFolder.video_list[i]+'/image_'+str(j+11)+'.png')
    j = 10
    pred_image = Image.fromarray(result_array[i,j])
    pred_image.save(path+'/image_'+str(j+11)+'.png')
    print("finish saving for: ", hiddenFolder.video_list[i])