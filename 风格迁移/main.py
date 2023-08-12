from keras.layers import *
from keras.utils.vis_utils import plot_model
import tensorflow.keras as tf_keras
import keras
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# tf_keras.mixed_precision.set_global_policy('mixed_float16') # 使用混合精度，前向传播用半精，反向传播用单精

def init_gpu():
    # 获取所有 GPU 设备列表
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置 GPU 显存占用为按需分配，增长式
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:

            print(e)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 屏蔽警告信息
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

init_gpu()

# 如果你没有内容图片和风格图片，可以使用下面的图片
# base_image_path=tf_keras.utils.get_file('sf.jpg',origin='https://img-datasets.s3.amazonaws.com/sf.jpg') # 内容图片
# style_reference_image_path=tf_keras.utils.get_file('starry_night.jpg',
#                                                    'https://img-datasets.s3.amazonaws.com/starry_night.jpg') # 风格图片

for base in range(1,4):
    for style in range(1,4):
        # jpg结尾的是base图片，jfif结尾的是style图片
        base_image_path=f'{base}.JPG'
        style_reference_image_path=f'{style}.jfif'

# print(base_image_path)
# print(style_reference_image_path)
        original_width,original_height=tf_keras.utils.load_img(base_image_path).size # 内容图片宽和高
        img_height=400 # 生成图片的高度
        img_width=round(original_width*img_height/original_height) # 按比例获取宽度


        def preprocess_image(image_path): # 打开图像，调整尺寸，转换为np数组
            img=tf_keras.utils.load_img(image_path,target_size=(img_height,img_width)) # 读取图片，按生成图片的比例来
            img=tf_keras.utils.img_to_array(img)
            img=np.expand_dims(img,axis=0)
            img=tf_keras.applications.vgg19.preprocess_input(img)
            return img


        def deprocess_img(img): # 将numpy数组转换为有效的图像(0~255范围）
            img=img.reshape((img_height,img_width,3))
            img[:,:,0]+=103.939
            img[:,:,1]+=116.779
            img[:,:,2]+=123.68
            # 预处理的逆操作
            img=img[:,:,::-1] # 将BGR转化为RGB，也是对预处理的逆操作
            img=np.clip(img,0,255).astype('uint8')
            return img


        model=tf_keras.applications.vgg19.VGG19(include_top=False)
        outputs_dict=dict([(layer.name,layer.output)for layer in model.layers])
        feature_extractor=keras.Model(model.inputs,outputs_dict)


        def content_loss(base_img,combination_img): # 评估内容的损失函数
            return tf.reduce_sum(tf.square(combination_img-base_img)) # 拿原始图片和生成图片的最后几个层的通道进行比较

        def gram_matrix(x):
            x=tf.transpose(x,(2,0,1)) # x是一张特征图，转置，把通道数移到最前面
            features=tf.reshape(x,(tf.shape(x)[0],-1)) # 按通道数拉平，现在每个通道都是一个很长的向量
            gram=tf.matmul(features,tf.transpose(features)) # 矩阵乘法，得到格拉姆矩阵
            # 第一个通道与第一个通道对应像素相乘，结果相加得到矩阵(1,1)位置的值
            # 简而言之就是第i个通道和第j个通道对应像素点相乘，然后把结果相加得到矩阵（i，j）位置的值
            # gram矩阵刻画了不同通道之间的相互关系，这些相互关系抓住了在某个空间尺度上的模式的统计规律，即纹理外观。
            return gram

        def style_loss(style_img,conination_img):
            S=gram_matrix(style_img) # 风格图像的格拉姆矩阵
            C=gram_matrix(conination_img) # 生成图像的格拉姆矩阵
            channels=3 # 通道数
            size=img_height*img_width # 总的像素数目
            return tf.reduce_sum(tf.square(S-C)) / (4.0*(channels**2)*(size**2)) # 该损失函数描述两图片的风格差异

        def total_variation_loss(x): # 总变差损失，使得图像具有空间连续性
            a=tf.square(x[:,:img_height-1,:img_width-1,:] - x[:,1:,:img_width-1]) # 将图片向下移动一格
            b=tf.square(x[:,:img_height-1,:img_width-1,:] - x[:,img_height-1,1:]) # 将图像向右移动一格
            return tf.reduce_sum(tf.pow(a+b,1.25))


        style_layer_names=[ # 用于风格损失
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1',
        ]

        content_layer_name='block5_conv2' # 用于内容损失

        total_variation_weight=1e-8
        style_weight=1e-6
        content_weight=2.5e-9

        def compute_loss(combination_image,base_image,style_reference_image):
            input_tensor=tf.concat([base_image,style_reference_image,combination_image],axis=0)
            # 将原图片，风格图片和目标图片组成一个批，输入模型

            features=feature_extractor(input_tensor) # 返回一个字典，存储了所有层的输出

            loss=tf.zeros(shape=()) # tf标量，损失函数初始为0
            layer_features=features[content_layer_name] # 最后一个卷积层的特征
            base_image_features=layer_features[0,:,:,:] # 取出原图片
            combination_features=layer_features[2,:,:,:] # 取出目标图片
            loss += content_weight*content_loss(base_image_features,combination_features) # 计算内容损失函数

            for layer_name in style_layer_names: # 对于每个用来计算风格损失函数的层
                layer_features=features[layer_name] # 获取该层的输出值
                style_reference_features=layer_features[1,:,:,:] # 风格图片
                combination_features=layer_features[2,:,:,:] # 生成图片
                style_loss_value=style_loss(style_reference_features,combination_features) # 计算风格损失函数
                loss += (style_weight/len(style_layer_names)) * style_loss_value # 由于有多个层用于计算，所以这里只是len()分之一

            loss += total_variation_weight*total_variation_loss(combination_image) # 添加生成图像的总变差损失
            return loss


        # @tf.function
        def compute_loss_and_grads(combination_image,base_image,style_reference_image):
            with tf.GradientTape() as tape:
                loss=compute_loss(combination_image,base_image,style_reference_image)
            grads=tape.gradient(loss,combination_image) # 梯度下降求原图片
            return loss,grads

        optimizer=tf_keras.optimizers.SGD(tf_keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=100.0,decay_steps=100,decay_rate=0.96 # 初始学习率为100，然后每100步减4%
        ))

        base_image=preprocess_image(base_image_path)
        style_reference_image=preprocess_image(style_reference_image_path)
        combination_image=tf.Variable(preprocess_image(base_image_path))


        iterations=10000
        loss_list=[]
        for i in range(1,iterations+1):
            loss,grads=compute_loss_and_grads(combination_image,base_image,style_reference_image)
            optimizer.apply_gradients([(grads,combination_image)])
            loss_list.append(loss)
            if i %100==0:
                print(f'Iteration {i}: loss={loss:.2f}')
                img=deprocess_img(combination_image.numpy())
                fname=f'生成图像{base}-{style}/第{i}轮.png'
                tf_keras.utils.save_img(fname,img)
        plt.plot(range(1,iterations+1)[100:],loss_list[100::])
        try :
            plt.savefig(f'{base}_{style}.png')
        except:
            pass
        plt.show()




