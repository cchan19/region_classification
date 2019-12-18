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

from multimodel import MultiModel, train_mapper, val_mapper, infer_mapper, data_reader

import time 

# 模型训练代码

def train(train_path, valid_path, visit_path, k, num_epochs):

    # 全局参数
    config = {
        'use_gpu': True,                # 是否使用 GPU
        'image_shape': (3, 100, 100),   # Image Network 输入尺寸
        'visit_shape': (7, 26, 24),     # Visit Network 输入尺寸
        'lr': 1e-4,                     # 学习率
        'num_epochs': num_epochs,              # 训练轮数
        'model_path': 'model_fold%d' % k,          # 模型缓存路径
        'freeze_path': 'freeze_model_fold%d' % k,  # 模型固化路径
    }

    train_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        input_image = fluid.layers.data(shape=config['image_shape'], name='image')
        input_visit = fluid.layers.data(shape=config['visit_shape'], name='visit')
        label = fluid.layers.data(shape=[1], name='label', dtype='int64')
        
        with fluid.unique_name.guard():
            out = MultiModel(input_image, input_visit)
            # 获取损失函数和准确率函数
            cost = fluid.layers.cross_entropy(out, label=label)
            avg_cost = fluid.layers.mean(cost)
            acc = fluid.layers.accuracy(out, label=label)
            
            # 获取训练和测试程序
            test_program = train_prog.clone(for_test=True)
        
            # 定义优化方法
            optimizer = fluid.optimizer.AdamOptimizer(learning_rate=config['lr'],
                                                      regularization=fluid.regularizer.L2DecayRegularizer(1e-5))
            # optimizer = fluid.optimizer.SGD(learning_rate=config['lr'])
            opts = optimizer.minimize(avg_cost)
    
    # 定义一个使用GPU的执行器
    place = fluid.CUDAPlace(0) if config['use_gpu'] else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # 进行参数初始化
    exe.run(startup_prog)
    
    # 继续训练
    if os.path.isdir(config['model_path']):
        fluid.io.load_persistables(exe, config['model_path'], main_program=train_prog)
    
    print('Loading pretrained model')
    def if_exist_se_resnext(var):
        return os.path.exists(os.path.join('pretrained_model/SE_ResNeXt50_32x4d_pretrained', var.name))
    fluid.io.load_vars(exe, 'pretrained_model/SE_ResNeXt50_32x4d_pretrained', main_program=train_prog, predicate=if_exist_se_resnext)
    
    train_reader = paddle.batch(reader=data_reader(train_path, train_mapper, visit_path), batch_size=64*2)
    test_reader = paddle.batch(reader=data_reader(valid_path, val_mapper, visit_path), batch_size=128*2)
    
    # 定义输入数据维度
    feeder = fluid.DataFeeder(place=place, feed_list=[input_image, input_visit, label])
    
    train_losses = []
    train_accs = []
    best_acc = 0
    for epoch in range(config['num_epochs']):
        tic = time.time()
        for step, data in enumerate(train_reader()):
            train_loss, train_acc = exe.run(program=train_prog,
                                            feed=feeder.feed(data),
                                            fetch_list=[avg_cost, acc])
            train_losses.append(train_loss[0])
            train_accs.append(train_acc[0])
            
            # 每 100 个 batch 打印一次信息
            sys.stdout.write('\r>> Epoch %d step %d: loss %0.5f accuracy %0.5f, time: %d' %
                         (epoch, step, sum(train_losses)/len(train_losses), sum(train_accs)/len(train_accs), time.time()-tic))
            sys.stdout.flush()
        print('')
    
        # 进行测试
        test_accs = []
        test_costs = []
        for batch_id, data in enumerate(test_reader()):
            test_cost, test_acc = exe.run(program=test_program,
                                          feed=feeder.feed(data),
                                          fetch_list=[avg_cost, acc])
            test_accs.append(test_acc[0])
            test_costs.append(test_cost[0])
        # 求测试结果的平均值
        test_cost = (sum(test_costs) / len(test_costs))
        test_acc = (sum(test_accs) / len(test_accs))
        print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (step, test_cost, test_acc))
        
        # 保存模型参数
        if not os.path.isdir(config['model_path']):
            os.makedirs(config['model_path'])
        if test_acc > best_acc:
            fluid.io.save_persistables(exe, config['model_path'], main_program=train_prog)
            best_acc = test_acc


def freeze_model(k, num_epochs):

    # 全局参数
    config = {
        'use_gpu': True,                # 是否使用 GPU
        'image_shape': (3, 100, 100),   # Image Network 输入尺寸
        'visit_shape': (7, 26, 24),     # Visit Network 输入尺寸
        'lr': 1e-4,                     # 学习率
        'num_epochs': num_epochs,              # 训练轮数
        'model_path': 'model_fold%d' % k,          # 模型缓存路径
        'freeze_path': 'freeze_model_fold%d' % k,  # 模型固化路径
    }

    """ 模型固化函数 """
    freeze_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(freeze_prog, startup_prog):
        # 模型定义
        input_image = fluid.layers.data(shape=config['image_shape'], name='image')
        input_visit = fluid.layers.data(shape=config['visit_shape'], name='visit')
        model = MultiModel(input_image, input_visit)
    
    exe = fluid.Executor(fluid.CPUPlace())
    
    model_path = config['model_path']
    if os.path.isdir(model_path):
        fluid.io.load_persistables(exe, model_path, main_program=freeze_prog)
    
    # 固化模型
    fluid.io.save_inference_model(config['freeze_path'], ['image', 'visit'], model, exe, freeze_prog)


def infer(test_path, visit_path, k, num_epochs):

    mm_prob = []

    # 全局参数
    config = {
        'use_gpu': True,                # 是否使用 GPU
        'image_shape': (3, 100, 100),   # Image Network 输入尺寸
        'visit_shape': (7, 26, 24),     # Visit Network 输入尺寸
        'lr': 1e-4,                     # 学习率
        'num_epochs': num_epochs,              # 训练轮数
        'model_path': 'model_fold%d' % k,          # 模型缓存路径
        'freeze_path': 'freeze_model_fold%d' % k,  # 模型固化路径
    }

    """ 模型预测函数 """
    place = fluid.CUDAPlace(0) if config['use_gpu'] else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    
    # 加载先前固化的模型
    [inference_program, feed_target_names, fetch_list] = fluid.io.load_inference_model(dirname=config['freeze_path'], executor=exe)

    # 生成预测数据读取器
    batch_size = 256
    infer_reader = paddle.batch(reader=data_reader(test_path, infer_mapper, visit_path), batch_size=batch_size)
    
    for i, data in enumerate(infer_reader()):
        result = exe.run(inference_program, fetch_list=fetch_list, feed={
            feed_target_names[0]: np.array([d[0] for d in data]).astype('float32'),
            feed_target_names[1]: np.array([d[1] for d in data]).astype('float32') })[0]
        mm_prob.append(result)
        sys.stdout.write('\r>> Processing %d / %d' % (i, 100000//batch_size))
        sys.stdout.flush()
    print()
    prob = np.vstack(mm_prob)
    
    return prob