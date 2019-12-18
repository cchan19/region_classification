import paddle
from paddle import fluid
import numpy as np


class MLPClassifier():
    
    def __init__(self, input_dim, cls_num, hidden_layer_sizes=(256,), act='relu', lr=1e-3):
        self.input_dim = input_dim
        self.cls_num = cls_num
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.act = act
        self.main_program = None
        self.startup_program = None
        self.test_program = None
        self.avg_cost = None
        self.acc = None
        self.inp = None
        self.label = None
        self.pred = None
        self.optimizer = None
        self.best_score = 0.0
        self.init_model()
        
    def init_model(self):
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        with fluid.program_guard(main_program=self.main_program, startup_program=self.startup_program):
            # model
            self.inp = fluid.layers.data(name='input', shape=[self.input_dim], dtype='float32')
            x = self.inp
            for sz in self.hidden_layer_sizes:
                x = fluid.layers.fc(input=x, size=sz, act=self.act)
            self.pred = fluid.layers.fc(name='pred', input=x, size=self.cls_num, act='softmax')
            # loss, metrics, optimizer
            self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')
            cost = fluid.layers.cross_entropy(input=self.pred, label=self.label)
            self.avg_cost = fluid.layers.mean(cost)
            self.acc = fluid.layers.accuracy(input=self.pred, label=self.label)
            self.test_program = self.main_program.clone(for_test=True)
            self.optimizer = fluid.optimizer.Adam(learning_rate=self.lr)
            self.optimizer.minimize(self.avg_cost)
    
    def fit(self, trn_reader, val_reader, batch_size=128, epoch_num=50, use_cuda=True, model_dir='best_model'):
        trn_reader = paddle.batch(trn_reader, batch_size=batch_size)
        with fluid.program_guard(main_program=self.main_program, startup_program=self.startup_program):
            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            feeder = fluid.DataFeeder(place=place, feed_list=[self.inp, self.label])
            exe.run(self.startup_program)
            for epoch in range(epoch_num):
                # train
                ytrue = []
                ypred = []
                for dat in trn_reader():
                    avg_loss_value, trn_acc, pred, label = exe.run(self.main_program,
                                      feed=feeder.feed(dat),
                                      fetch_list=[self.avg_cost, self.acc, self.pred, self.label])
                    ytrue.append(label)
                    ypred.append(pred)
                ytrue = np.vstack(ytrue)
                ypred = np.vstack(ypred)
                ypred = np.argmax(ypred, axis=1).reshape(-1, 1)
                trn_acc_score = ((ytrue - ypred) == 0).sum() / len(ytrue)
                # eval
                val_acc_score, _, _ = self.score(val_reader, batch_size=batch_size, use_cuda=use_cuda)
                print('Epoch %d\ntrain acc: %f, valid acc: %f' % (epoch, trn_acc_score, val_acc_score))
                if val_acc_score > self.best_score:
                    self.best_score = val_acc_score
                    self.save_model(model_dir)
                    print('model saved, best score: %f' % self.best_score)
            
    def predict_prob(self, reader, batch_size=128, use_cuda=True):
        _, pred, _ = self.score(reader, batch_size=128, use_cuda=True)
        return pred
        
    def score(self, reader, batch_size=128, use_cuda=True):
        reader = paddle.batch(reader, batch_size=batch_size)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(place=place, feed_list=[self.inp, self.label])
        ypred = []
        ytrue = []
        for dat in reader():
            pred, label = exe.run(self.test_program,
                              feed=feeder.feed(dat),
                              fetch_list=[self.pred, self.label])
            ytrue.append(label)
            ypred.append(pred)
        ytrue = np.vstack(ytrue)
        ypred = np.vstack(ypred)
        ypred_ = np.argmax(ypred, axis=1).reshape(-1, 1)
        acc_score = ((ytrue - ypred_) == 0).sum() / len(ytrue)
        return acc_score, ypred, ytrue
        
    def save_model(self, model_dir):
        exe = fluid.Executor(fluid.CPUPlace())
        fluid.io.save_params(executor=exe, dirname=model_dir, main_program=self.main_program)
        
    def load_model(self, model_dir):
        exe = fluid.Executor(fluid.CPUPlace())
        fluid.io.load_params(executor=exe, dirname=model_dir, main_program=self.main_program)
        
        