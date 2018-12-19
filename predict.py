from matplotlib import pyplot as plt
import cv2
import numpy as np
from model import Deeplabv3
import os

root_dir='imgs'
file_name='image4'
ext_type='.jpg'

input_file_path=os.path.join(root_dir, (file_name + ext_type))
output_file_path = os.path.join(root_dir, (file_name + '_label' + ext_type))
vis_file_path = os.path.join(root_dir, (file_name + '_vis' + ext_type))

deeplab_model = Deeplabv3(backbone='xception')
img = plt.imread(input_file_path)
w, h, _ = img.shape
ratio = 512. / np.max([w, h])
resized = cv2.resize(img, (int(ratio*h), int(ratio*w)))
resized = resized / 127.5 - 1.
pad_x = int(512 - resized.shape[0])
resized2 = np.pad(resized, ((0, pad_x), (0, 0), (0, 0)) ,mode='constant')
res = deeplab_model.predict(np.expand_dims(resized2, 0))
labels = np.argmax(res.squeeze(), -1)
output = labels[:-pad_x]
# plt.imshow(output)
# plt.waitforbuttonpress()
plt.imsave(output_file_path, output)

background = cv2.imread(input_file_path)
mask = cv2.resize(cv2.imread(output_file_path), (background.shape[1], background.shape[0]))
vis = cv2.addWeighted(background, 0.7, mask, 0.3, 0)
cv2.imwrite(vis_file_path, vis)



