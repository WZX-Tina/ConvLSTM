import numpy as np
from PIL import Image

label = np.load('label.npy')
pred = np.load('pred.npy')
print(label.shape)
print(pred.shape)

for i in range(11):
    print(label[i])
    print('stop')
    print(pred[i])
    label_image = Image.fromarray(label[i])
    pred_image = Image.fromarray(pred[i])

    label_image.save('label_'+str(i)+'.png')
    pred_image.save(f'pred_'+str(i)+'.png')

