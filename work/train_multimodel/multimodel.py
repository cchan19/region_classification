from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import random
import sys
import datetime
import pandas as pd
import os
import threading
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from multiprocessing import cpu_count

import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

# SE_ResNeXt

__all__ = [
    "SE_ResNeXt", "SE_ResNeXt50_32x4d", "SE_ResNeXt101_32x4d",
    "SE_ResNeXt152_32x4d"
]


class SE_ResNeXt():
    def __init__(self, layers=50):
        self.layers = layers

    def net(self, input, class_dim=1000):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)
        if layers == 50:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 6, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
                name='conv1', )
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max',
                use_cudnn=False)
        elif layers == 101:
            cardinality = 32
            reduction_ratio = 16
            depth = [3, 4, 23, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
                name="conv1", )
            conv = fluid.layers.pool2d(
                input=conv,
                pool_size=3,
                pool_stride=2,
                pool_padding=1,
                pool_type='max',
                use_cudnn=False)
        elif layers == 152:
            cardinality = 64
            reduction_ratio = 16
            depth = [3, 8, 36, 3]
            num_filters = [128, 256, 512, 1024]

            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=3,
                stride=2,
                act='relu',
                name='conv1')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=64,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv2')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=128,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv3')
            conv = fluid.layers.pool2d(
                input=conv, pool_size=3, pool_stride=2, pool_padding=1, \
                pool_type='max', use_cudnn=False)
        n = 1 if layers == 50 or layers == 101 else 3
        for block in range(len(depth)):
            n += 1
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    cardinality=cardinality,
                    reduction_ratio=reduction_ratio,
                    name=str(n) + '_' + str(i + 1))

        pool = fluid.layers.pool2d(
            input=conv, pool_type='avg', global_pooling=True, use_cudnn=False)
        drop = fluid.layers.dropout(x=pool, dropout_prob=0.5)
        stdv = 1.0 / math.sqrt(drop.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=drop,
            size=class_dim,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name='fc6_weights'),
            bias_attr=ParamAttr(name='fc6_offset'))
        return out

    def shortcut(self, input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            filter_size = 1
            return self.conv_bn_layer(
                input, ch_out, filter_size, stride, name='conv' + name + '_prj')
        else:
            return input

    def bottleneck_block(self,
                         input,
                         num_filters,
                         stride,
                         cardinality,
                         reduction_ratio,
                         name=None):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name='conv' + name + '_x1')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            groups=cardinality,
            act='relu',
            name='conv' + name + '_x2')
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 2,
            filter_size=1,
            act=None,
            name='conv' + name + '_x3')
        scale = self.squeeze_excitation(
            input=conv2,
            num_channels=num_filters * 2,
            reduction_ratio=reduction_ratio,
            name='fc' + name)

        short = self.shortcut(input, num_filters * 2, stride, name=name)

        return fluid.layers.elementwise_add(x=short, y=scale, act='relu')

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False,
            param_attr=ParamAttr(name=name + '_weights'), )
        bn_name = name + "_bn"
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def squeeze_excitation(self,
                           input,
                           num_channels,
                           reduction_ratio,
                           name=None):
        pool = fluid.layers.pool2d(
            input=input, pool_type='avg', global_pooling=True, use_cudnn=False)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=num_channels // reduction_ratio,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act='sigmoid',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale


def SE_ResNeXt50_32x4d():
    model = SE_ResNeXt(layers=50)
    return model


def SE_ResNeXt101_32x4d():
    model = SE_ResNeXt(layers=101)
    return model


def SE_ResNeXt152_32x4d():
    model = SE_ResNeXt(layers=152)
    return model

# ResNet

__all__ = [
    "ResNet", "ResNet50_vd", "ResNet101_vd", "ResNet152_vd", "ResNet200_vd"
]

class ResNet():
    def __init__(self, layers=50, is_3x3=False):
        self.layers = layers
        self.is_3x3 = is_3x3

    def net(self, input, class_dim=1000):
        is_3x3 = self.is_3x3
        layers = self.layers
        supported_layers = [20, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 20:
            depth = [2, 2, 2]
        elif layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_filters = [64, 128, 256, 512]
        if is_3x3 == False:
            conv = self.conv_bn_layer(
                input=input,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu')
        else:
            conv = self.conv_bn_layer(
                input=input,
                num_filters=32,
                filter_size=3,
                stride=2,
                act='relu',
                name='conv1_1')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=32,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv1_2')
            conv = self.conv_bn_layer(
                input=conv,
                num_filters=64,
                filter_size=3,
                stride=1,
                act='relu',
                name='conv1_3')

        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                if layers in [101, 152, 200] and block == 2:
                    if i == 0:
                        conv_name = "res" + str(block + 2) + "a"
                    else:
                        conv_name = "res" + str(block + 2) + "b" + str(i)
                else:
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    if_first=block == 0,
                    name=conv_name)

        pool = fluid.layers.pool2d(
            input=conv, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)

        out = fluid.layers.fc(
            input=pool,
            size=class_dim,
            param_attr=fluid.param_attr.ParamAttr(
                name='res_fc_weights',
                initializer=fluid.initializer.Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name='res_fc_bias'))

        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(name=bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def conv_bn_layer_new(self,
                          input,
                          num_filters,
                          filter_size,
                          stride=1,
                          groups=1,
                          act=None,
                          name=None):
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=2,
            pool_stride=2,
            pool_padding=0,
            pool_type='avg')

        conv = fluid.layers.conv2d(
            input=pool,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=1,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def shortcut(self, input, ch_out, stride, name, if_first=False):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            # if if_first:
                return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
            # else:
            #     return self.conv_bn_layer_new(
            #         input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name, if_first):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c")

        short = self.shortcut(
            input,
            num_filters * 4,
            stride,
            if_first=if_first,
            name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def ResNet20_vd():
    model = ResNet(layers=20, is_3x3=True)
    return model


def ResNet50_vd():
    model = ResNet(layers=50, is_3x3=True)
    return model


def ResNet101_vd():
    model = ResNet(layers=101, is_3x3=True)
    return model


def ResNet152_vd():
    model = ResNet(layers=152, is_3x3=True)
    return model


def ResNet200_vd():
    model = ResNet(layers=200, is_3x3=True)
    return model

# MultiModel

def MultiModel(image, visit):
    se_resnext = SE_ResNeXt()
    resnet20_vd = ResNet20_vd()
    output_image = se_resnext.net(image, class_dim=1000)
    output_visit = resnet20_vd.net(visit, class_dim=1000)
    output = fluid.layers.concat([output_image, output_visit], axis=1)
    prediction = fluid.layers.fc(input=output, size=9, act='softmax',
        param_attr=ParamAttr(name='multi_fc_weights'),
        bias_attr=ParamAttr(name='multi_fc_bias'))
    return prediction

# 数据读取代码

train_visit_max = 3160.
test_visit_max = 4779.

def train_mapper(sample):
    img_path, visit, label = sample
    
    img = Image.open(img_path)
    
    resize_size = 100
    # 统一图片大小
    img = img.resize((resize_size, resize_size), Image.ANTIALIAS)
    # 随机水平翻转
    r1 = random.random()
    if r1 > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # 随机垂直翻转
    r2 = random.random()
    if r2 > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # 随机角度翻转
    r3 = random.randint(-3, 3)
    img = img.rotate(r3, expand=False)
    
    # 把图片转换成numpy值
    img = np.array(img).astype(np.float32)
    # 转换成CHW
    img = img.transpose((2, 0, 1))
    # 转换成BGR
    img = img[(2, 1, 0), :, :] / 255.0
    
    visit /= train_visit_max
    return img, visit, label

def val_mapper(sample):
    img_path, visit, label = sample
    
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = img[(2, 1, 0), :, :] / 255.0
    
    visit /= train_visit_max
    return img, visit, label

def infer_mapper(sample):
    img_path, visit, label = sample
    
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)
    img = img.transpose((2, 0, 1))
    img = img[(2, 1, 0), :, :] / 255.0
    
    areaID = img_path.split('/')[-1].split('.')[0].split('_')[0]
    visit /= test_visit_max
    return img, visit, label, areaID

def data_reader(file_list, mapper, path_visit):
    def reader():
        with open(file_list, 'r') as f:
            lines = f.read().splitlines()
            indexes = np.arange(0, len(lines))
#             np.random.shuffle(indexes)
            for i in indexes:
                line = lines[i]
                image = line
                visit = np.load(path_visit+line.split('/')[-1].split('.')[0]+".npy")
                label = int(line.split("/")[-1].split("_")[-1].split(".")[0]) - 1
#                 yield image, visit, label
                yield mapper((image, visit, label))
                
    return reader
#     return paddle.reader.xmap_readers(mapper, reader, cpu_count(), 1024)

