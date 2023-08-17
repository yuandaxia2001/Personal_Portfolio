import matplotlib.cm as cm
import tensorflow.keras as tf_keras
import numpy as np
import tensorflow as tf
import keras
def get_img_array(img_path,target_size):
    '''
    读取图片并进行预处理，返回1,size,size,3的张量
    :param img_path: 图像路径
    :param target_size: 图像尺寸（xception网络接收长宽为299的图片
    :return:
    '''
    img=tf_keras.utils.load_img(img_path,target_size=target_size) # 返回一张尺寸为299*299的PLT图像
    array=tf_keras.utils.img_to_array(img) # 返回一个299,299,3的float32的np数组
    array=np.expand_dims(array,axis=0) # 增加一个维度，变成1,299,299,3
    array=tf_keras.applications.xception.preprocess_input(array)
    return array

def superimpose(heatmap, img):
    '''
    叠加heatmap和img图片，可视化结果
    :param heatmap: 热力图
    :param img: 原始图片
    :return: 叠加后的图片
    '''
    # 使用jet颜色图对热力图重新着色
    jet = cm.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # 创建一张图表，使其包含重新着色的热力图
    jet_heatmap = tf_keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf_keras.utils.img_to_array(jet_heatmap)

    # 热力图和原始图像叠加，热力图的不透明度为0.4
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = tf_keras.utils.array_to_img(superimposed_img)


    return superimposed_img

def grad_cam(model,img_array,last_conv_layer_name,classifier_layer_names):
    '''
    grad_cam算法
    :param model: 模型
    :param img_array:可直接输入模型的原始图片
    :param last_conv_layer_name: 最后一个卷积层的名字
    :param classifier_layer_names: 分类器包含的层（即最后一个卷积层之后的层）
    :return: 热力图，通道数为1，像素为[0,255]的整数
    '''

    last_conv_layer=model.get_layer(last_conv_layer_name)
    last_conv_layer_model=keras.Model(model.inputs,last_conv_layer.output) # 从输入图像到最后一个卷积层激活值的映射模型

    # 创建一个模型，将最后一个卷积层的激活值映射到最终的预测类别
    classifier_input=keras.Input(shape=last_conv_layer.output_shape[1:])
    x=classifier_input
    for layer_name in classifier_layer_names:
        x=model.get_layer(layer_name)(x)
    classifier_model=keras.Model(classifier_input,x)

    # 以上将模型分成了两个部分，前面的卷积部分和后面的分类部分

    with tf.GradientTape() as tape:
        last_conv_layer_output=last_conv_layer_model(img_array) # 计算最后一个卷积层的激活值
        tape.watch(last_conv_layer_output) # 让梯度带监控该激活值

        preds=classifier_model(last_conv_layer_output) # 分类器的输入是卷积层的输出
        top_pred_index=tf.argmax(preds[0]) # 获取最大概率对应的索引
        top_class_channel=preds[:,top_pred_index]
        # 检索与最大概率类别对应的激活通道，即从1000个输出通道中，检索最大概率的那个通道

    grads=tape.gradient(top_class_channel,last_conv_layer_output)
    # 计算最大预测类别相对于最后一个卷积层的输出特征图的梯度 d(top_class_channel)/d(last_conv_layer_output)

    # 下面对梯度张量进行汇聚和重要性加权，以得到类激活热力图

    pooled_grads=tf.reduce_mean(grads,axis=(0,1,2)).numpy()
    # 这是一个向量，其中每个元素是某个通道的平均梯度强度，它量化了每个通道对最大预测类别的重要性
    last_conv_layer_output=last_conv_layer_output.numpy()[0]

    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:,:,i] *= pooled_grads[i]
    heatmap=np.mean(last_conv_layer_output,axis=-1)

    heatmap=np.maximum(heatmap,0) # 做relu处理
    heatmap/=np.max(heatmap) # 归一化

    heatmap=np.uint(255*heatmap)

    return heatmap

