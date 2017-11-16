import numpy as np
import math
from processdata import Encode
from numpy import *

random.seed(5)

class APTrain:
    def __init__(self):
        self.weight_matrix = np.array((0,0))
        self.grad_matrix = np.array((0,0))
        self.average_matrix = np.array((0,0))
        self.bias = np.array([0.0,0.0,0.0,0.0,0.0])

    def create_weight_matrix(self,depth,width):
        self.weight_matrix = np.zeros((depth,width))
        self.grad_matrix = np.zeros((depth,width))
        self.average_matrix = np.zeros((depth,width))

    def train(self,parameter,train_encodes,dev_encodes):
        for i in range(parameter.ap_iter_num):
            print('第%d轮迭代：'%(i+1))
            step = denominator = parameter.ap_iter_num*len(train_encodes)
            total = cor = 0
            for encode in train_encodes:
                sum = np.array([0.0,0.0,0.0,0.0,0.0])
                for e in encode.code_list:
                    sum += self.weight_matrix[e]
                y = self.softmax(sum+self.bias)
                # if self.get_maxIndex(y) != self.get_maxIndex(encode.label):
                for e in encode.code_list:
                    self.weight_matrix[e] += np.subtract(encode.label,y)
                    self.bias += np.subtract(encode.label,y)
                    self.average_matrix[e] += (step/denominator)*np.subtract(encode.label,y)
                if self.get_maxIndex(y) == self.get_maxIndex(encode.label):
                    cor+=1
                total+=1
                step -= 1
            print('train accuarcy:',cor/total)
            self.eval(dev_encodes,'dev')
            if cor/total == 1.0:
                break
            train_encodes = self.encode_random(train_encodes)
            print('-------------------------')

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

    def eval(self,encodes,dataset_name):
        cor =0
        total= 0
        for e in encodes:
            sum = np.array([0.0,0.0,0.0,0.0,0.0])
            for cl in e.code_list:
                sum+=self.average_matrix[cl]
            if self.get_maxIndex(self.softmax(sum+self.bias)) == self.get_maxIndex(e.label):
                cor+=1
            total+=1
        if dataset_name=='dev' and cor/total > 0.38:
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


