# ConvLSTM-Pytorch

This is an adapted version of ConvLSTM, used by Group 15 for DS-GA 1008 Deep Learning final project.

## ConvRNN cell

Implement ConvLSTM/ConvGRU cell with Pytorch. This idea has been proposed in this paper: [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214).


## Instructions
This is the first approach of our model, for video prediction. Since masks are not involved, 1k training videos and 15k unlabeled videos are used for training, and the 1k validation videos are used for saving checkpoints by validation loss. This repo only provides the necessary scripts. All data and results are already saved in scratch folder `/scratch/tw2672/1008_Project` on the Greene cluster (NOT GCP). Please see the code for specific paths.

### Training
This step is to use training and unlabeled videos for training the video prediction model and saving checkpoints. Note 100GB memory is requested and each epoch takes ~3 hours. The file is `main.py` and the command is `sbatch train.SBATCH`.

### Prediction
This step is to use the saved checkpoint to predict the future 11 frames for the hidden dataset. The prediction is saved as a `.npy` file. The file is `load_model.py` and the command is `sbatch pred.SBATCH`.

### Conversion
This step is to reconstruct the actual images from the `.npy` file. It produces the 22nd frame for each video. The file is `np_to_png.py` and the command is `sbatch conv.py`. If access is denied, the predicted frames for the hidden dataset are stored on [Google Drive](https://drive.google.com/file/d/13qWoGQx2bx9Fpk1KVrNzqAJe91AQz9BP/view?usp=drive_link).

## Dataloader Generator

The script ``data/mm.py`` is the script to generate customized MovingFrame dataloader based on our video data. 

```python
Folder = MovingFrame(is_train=False, # True for training data
                      is_val=False, # True for validation data
                      root=root_path+'/data/hidden', # path of the train/val/unlabeled/hidden data
                      n_frames_input=11,
                      n_frames_output=0, # 11 for training and 0 for prediction
                      )
```

## Citation

```
@inproceedings{xingjian2015convolutional,
  title={Convolutional LSTM network: A machine learning approach for precipitation nowcasting},
  author={Xingjian, SHI and Chen, Zhourong and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-Kin and Woo, Wang-chun},
  booktitle={Advances in neural information processing systems},
  pages={802--810},
  year={2015}
}
@inproceedings{xingjian2017deep,
    title={Deep learning for precipitation nowcasting: a benchmark and a new model},
    author={Shi, Xingjian and Gao, Zhihan and Lausen, Leonard and Wang, Hao and Yeung, Dit-Yan and Wong, Wai-kin and Woo, Wang-chun},
    booktitle={Advances in Neural Information Processing Systems},
    year={2017}
}
```