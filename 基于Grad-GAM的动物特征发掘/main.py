import tkinter

import keras
import matplotlib.pyplot as plt
import tensorflow.keras as tf_keras
import tensorflow as tf
import numpy as np
from keras.layers import *
from keras.utils.vis_utils import plot_model

from API import *

model=tf_keras.applications.xception.Xception()

img_paths=['大象图片/印度象.jpg','大象图片/非洲象.jpg','大象图片/印度象2.jpg','大象图片/非洲象2.jpg',
           '鲨鱼图片/大白鲨.jpg','鲨鱼图片/虎鲨.jpg','鲨鱼图片/锤头鲨.jpg',
           '鲨鱼图片/大白鲨2.jpg','鲨鱼图片/虎鲨2.jpg','鲨鱼图片/锤头鲨2.jpg',]

for img_path in img_paths:
    img_array=get_img_array(img_path,(299,299))

    preds=model.predict(img_array)
    file_name=img_path[img_path.rindex("/")+1:] # 最后一个'/'之后的内容为文件名
    print(f'文件名：{file_name}')
    print(f'top3:\n{tf_keras.applications.xception.decode_predictions(preds,top=3)[0]}\n')

    last_conv_layer_name = 'block14_sepconv2_act'  # 将输入图像映射到最后一个卷积层的激活值
    classifier_layer_names = ['avg_pool', 'predictions']  # 整个模型的最后两层，全局池化和预测层

    heatmap=grad_cam(model,img_array,last_conv_layer_name,classifier_layer_names)

    img=tf_keras.utils.load_img(img_path)
    img=tf_keras.utils.img_to_array(img)

    superimposed_img=superimpose(heatmap,img)
    save_path = f'叠加图片/{file_name}.jpg'
    superimposed_img.save(save_path)

