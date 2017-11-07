import numpy as np
import math
from processdata import Encode
from numpy import *

random.seed(5)

class Train:
    def __init__(self):
        self.weight_matrix = 0

    def create_weight_matrix(self,depth,width):
        self.weight_matrix = np.random.random((depth,width))

    def train(self,train_encodes,iter_num,dev_encodes):
        for i in range(iter_num):
            print('第%d轮迭代：'%(i+1))
            rb = self.forward(train_encodes)
            self.backward(train_encodes,rb,0.01)
            accuracy = self.eval(train_encodes,'train')
            self.eval(dev_encodes,'dev')
            if accuracy == 1.0:
                break
            print('-------------------------')
        pass

    def forward(self, encodes):
        result_labels = []
        for e in encodes:
            sum = np.array([0.0,0.0,0.0,0.0,0.0])
            for i in e.code_list:
                sum += self.weight_matrix[i]
            result_labels.append(self.softmax(sum))
        return result_labels

    def softmax(self,result_label):
        max = self.get_max(result_label)
        list = []
        sum = 0.0
        for i in result_label:
            sum+=math.exp(i-max)
        for i in result_label:
            list.append(math.exp(i-max)*(1/sum))
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

    def backward(self,encodes,outputs,lr):
        for i,e in enumerate(encodes):
            for cl in e.code_list:
                self.weight_matrix[cl] -= lr * (np.array(outputs[i])-np.array(e.label))

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
        print(dataset_name+' accuracy:', cor/total)
        return cor/total

    # def create_batch_list(self,encodes,batch_size):

    def encode_random(self,o_encodes):
        index_list = []
        for i in range(len(o_encodes)):
            index_list.append(i)
        random.seed(150)
        random.shuffle(index_list)
        n_encodes = []
        for i in index_list:
            encode = Encode()
            encode.code_list = o_encodes[i].code_list
            encode.label = o_encodes[i].label
            n_encodes.append(encode)
        return n_encodes