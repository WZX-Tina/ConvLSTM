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

TIMESTAMP = "2023-12-05T00-00-00"
save_dir = './save_model/' + TIMESTAMP

encoder = Encoder(convgru_encoder_params[0], convgru_encoder_params[1]).cuda()
decoder = Decoder(convgru_decoder_params[0], convgru_decoder_params[1]).cuda()
net = ED(encoder, decoder)
model_info = torch.load(os.path.join(save_dir, 'checkpoint_43_0.000931.pth.tar'))
net.load_state_dict(model_info['state_dict'])

validFolder = MovingFrame(is_train=False,
                          root='../data/val',
                          n_frames_input=11,
                          n_frames_output=11,
                          )
validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=1,
                                          shuffle=False)

# out = net(validFolder.dataset[0,:11])
# print(out.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_per_process_memory_fraction(0.8)

t = tqdm(validLoader,leave=False,total=2)
for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
    inputs = inputVar.to(device)
    label = targetVar.to(device)
    pred = net(inputs)
    
    label = torch.squeeze(label)
    pred = torch.squeeze(pred)

    label_array = label.detach().cpu().numpy().transpose(0,2,3,1)
    pred_array = pred.detach().cpu().numpy().transpose(0,2,3,1)

    label_array = (label_array * 255).astype(np.uint8)
    pred_array = (pred_array * 255).astype(np.uint8)

    print(label_array.shape)

    np.save('label.npy',label_array)
    np.save('pred.npy',pred_array)
    print(label.shape)
    print(pred.shape)
    torch.cuda.empty_cache()
