import torch
import os
from encoder import Encoder
from decoder import Decoder
from model import ED
import numpy as np
from data.mm import MovingFrame
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TIMESTAMP = "2023-12-11T00-00-00"
save_dir = './save_model/' + TIMESTAMP
root_path = '/scratch/tw2672/1008_Project'
checkpoint = 'checkpoint_2_0.000118.pth.tar'

hiddenFolder = MovingFrame(is_train=False,
                          is_val=False,
                          root=root_path+'/data/hidden',
                          n_frames_input=11,
                          n_frames_output=0,
                          )

encoder = Encoder(convgru_encoder_params[0], convgru_encoder_params[1]).cuda()
decoder = Decoder(convgru_decoder_params[0], convgru_decoder_params[1]).cuda()
net = ED(encoder, decoder)

model_info = torch.load(os.path.join(save_dir, checkpoint))
net.load_state_dict(model_info['state_dict'])

result_array = []
for i, (idx, targetVar, inputVar, _, _) in enumerate(hiddenFolder):

    inputs = torch.unsqueeze(inputVar,0).to(device)
    
    pred = net(inputs)
    pred = torch.squeeze(pred)

    pred_array = pred.detach().cpu().numpy().transpose(0,2,3,1)

    pred_array = np.expand_dims((pred_array * 255).astype(np.uint8), axis=0)
    if len(result_array) == 0:
        result_array = pred_array
    else:
        result_array = np.append(result_array, pred_array, axis=0)

    torch.cuda.empty_cache()

print('pred shape', result_array.shape)
np.save(root_path+'/video_arrays_result_hidden.npy', result_array)