import numpy as np
import math
from processdata import Encode
from numpy import *

random.seed(5)

class Train:
    def __init__(self):
        self.weight_matrix = 0
        self.grad_matrix = np.array((0,0))
        self.cumulative_matrix = 0
        self.bias = [1e-10,1e-10,1e-10,1e-10,1e-10]

    def create_weight_matrix(self,depth,width):
        self.weight_matrix = np.random.random((depth,width))
        self.grad_matrix = np.zeros((depth,width))
        self.cumulative_matrix = np.zeros((depth,width))
        # bias_list = []
        # for i in range(width):
        #     bias_list.append(1e-09)
        # self.weight_matrix[len(self.weight_matrix)-1] = np.array(bias_list)

    def train(self,parameter,train_encodes,dev_encodes):
        for i in range(parameter.iter_num):
            print('第%d轮迭代：'%(i+1))
            # outputs = self.forward(train_encodes)
            left_bound = 0
            right_boud = parameter.batch_size
            max_len = len(train_encodes)
            while left_bound<max_len:
                outputs = self.forward(train_encodes[left_bound:right_boud],parameter.batch_size)
                self.backward(train_encodes[left_bound:right_boud],outputs,parameter)
                left_bound+=parameter.batch_size
                right_boud+=parameter.batch_size
                if right_boud >= max_len:
                    right_boud = max_len-1
                self.grad_matrix = np.zeros(self.grad_matrix.shape) #每个batch一次梯度清零
            accuracy = self.eval(train_encodes,'train')
            self.eval(dev_encodes,'dev')
            train_encodes = self.encode_random(train_encodes) #随机梯度下降
            if accuracy == 1.0:
                break
            print('-------------------------')

    def forward(self, encodes,batch_size):
        result_labels = []
        for e in encodes:
            sum = np.array([0.0,0.0,0.0,0.0,0.0])
            for i in e.code_list:
                sum += self.weight_matrix[i]
            result_labels.append(self.softmax((1/batch_size)*sum))
        return result_labels

    def softmax(self,result_label):
        max = self.get_max(result_label)
        list = []
        sum = 0.0
        for i in result_label:
            sum+=math.exp(i-max)
        for i in result_label:
            list.append(math.exp(i-max)*(1/sum))
        # self.print_loss(list)
        return list

    def get_max(self,list):
        max = list[0]
        for i in range(len(list)):
            if list[i] > max:
                max = list[i]
        return max

    def get_maxIndex(self,list):
        max,index = list[0],0
        for i in range(len(list)):
            if list[i] > max:
                max,index = list[i],i
        return index

    def backward(self,encodes,outputs,parameter):
        o_labels = []
        for ec in encodes:
            # o_labels.append(np.subtract(ec.label,self.bias))
            o_labels.append(ec.label)
        value = np.array(outputs)-np.array(o_labels)
        for i,e in enumerate(encodes):
            for cl in e.code_list:
                self.grad_matrix[cl] += value[i]
                self.Adagrad(self.grad_matrix[cl],cl)
        self.weight_matrix -= (parameter.learn_rate/(parameter.kthi+np.sqrt(self.cumulative_matrix)))*self.grad_matrix #Adagrad
        # self.weight_matrix -= parameter.learn_rate*self.grad_matrix  #批梯度下降
        # for i,e in enumerate(encodes):
        #     for cl in e.code_list:
        #         self.weight_matrix[cl] -= lr * (np.array(outputs[i])-np.array(e.label))

    def eval(self,encodes,dataset_name):
        cor =0
        total= 0
        for e in encodes:
            sum = np.array([0.0,0.0,0.0,0.0,0.0])
            for cl in e.code_list:
                sum+=self.weight_matrix[cl]
            if self.get_maxIndex(self.softmax(sum)) == self.get_maxIndex(e.label):
                cor+=1
            total+=1
        if dataset_name=='dev' and cor/total > 0.39:
            print('*******')
        print(dataset_name+' accuracy:', cor/total)
        return cor/total

    def encode_random(self,o_encodes):
        index_list = []
        for i in range(len(o_encodes)):
            index_list.append(i)
        random.seed(200)
        random.shuffle(index_list)
        n_encodes = []
        for i in index_list:
            encode = Encode()
            encode.code_list = o_encodes[i].code_list
            encode.label = o_encodes[i].label
            n_encodes.append(encode)
        return n_encodes

    def print_loss(self,list):
        print('loss:',-1*math.log2(max(list)))

    def Adagrad(self,grad,index):
        self.cumulative_matrix[index] += np.square(grad)
