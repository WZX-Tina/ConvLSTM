import torch
import os
from encoder import Encoder
from decoder import Decoder
from model import ED
import numpy as np
from data.mm import MovingFrame
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from tqdm import tqdm

# checkpoint_43_0.000931.pth.tar

TIMESTAMP = "2023-12-05T02-00-00"
save_dir = './save_model/' + TIMESTAMP

encoder = Encoder(convgru_encoder_params[0], convgru_encoder_params[1]).cuda()
decoder = Decoder(convgru_decoder_params[0], convgru_decoder_params[1]).cuda()
net = ED(encoder, decoder)
model_info = torch.load(os.path.join(save_dir, 'checkpoint_84_0.000642.pth.tar'))
net.load_state_dict(model_info['state_dict'])

hiddenFolder = MovingFrame(is_train=False,
                          is_val=False,
                          root='../data/hidden',
                          n_frames_input=11,
                          n_frames_output=0,
                          )
hiddenLoader = torch.utils.data.DataLoader(hiddenFolder,
                                          batch_size=1,
                                          shuffle=False)

# out = net(validFolder.dataset[0,:11])
# print(out.shape)

print(hiddenFolder.video_list[:10])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.8)

result_array = []
for i, (idx, targetVar, inputVar, _, _) in enumerate(hiddenFolder):
    # print(targetVar)
    # print(inputVar)
    print('i',i)
    #print(targetVar.shape)
    #print(inputVar.shape)
    print('idx',idx)

    inputs = torch.unsqueeze(inputVar,0).to(device)
    #label = targetVar.to(device)
    
    pred = net(inputs)
    
    #label = torch.squeeze(label)
    pred = torch.squeeze(pred)

    #label_array = label.detach().cpu().numpy().transpose(0,2,3,1)
    pred_array = pred.detach().cpu().numpy().transpose(0,2,3,1)

    #label_array = (label_array * 255).astype(np.uint8)
    pred_array = np.expand_dims((pred_array * 255).astype(np.uint8), axis=0)
    if len(result_array) == 0:
        result_array = pred_array
    else:
        result_array = np.append(result_array, pred_array, axis=0)

    #print(label_array.shape)

    #np.save('label.npy',label_array)
    np.save('pred.npy',pred_array)
    #print(label.shape)
    print(pred.shape)
    torch.cuda.empty_cache()

    #if i >= 5:
    #    break

print(result_array.shape)
np.save('video_arrays_result.npy', result_array)
