import numpy as np
import math
from processdata import Encode
from numpy import *
import time

class APTrain:
    def __init__(self):
        self.weight_matrix = np.array((0,0))
        self.sum_weight_matrix = np.array((0,0))
        self.sum_bias = np.array([0.0,0.0,0.0,0.0,0.0])
        self.bias = np.array([0.0,0.0,0.0,0.0,0.0])
        self.last_update_weight = []
        self.update = 0

    def create_weight_matrix(self,depth,width):
        self.weight_matrix = np.zeros((depth,width))
        self.sum_weight_matrix = np.zeros((depth,width))
        self.last_update_weight = [0 for i in range(depth)]

    def train(self,parameter,train_encodes,dev_encodes,test_encodes):
        step = 1
        for i in range(parameter.ap_iter_num):
            print('第%d轮迭代：'%(i+1))
            starttime = time.time()
            total = cor = loss = left_bound = 0
            right_bound = parameter.ap_batch_size
            max_len = len(train_encodes)
            while left_bound<max_len:    #batch
                for encode in train_encodes[left_bound:right_bound]:
                    sum = np.array([0.0 for i in range(parameter.class_num)])
                    for e in encode.code_list:
                        sum += self.weight_matrix[e]
                    if self.get_maxIndex(sum) != self.get_maxIndex(encode.label):
                        # punish_vec = self.generate_punish_vec(parameter.class_num,-1,0,1,self.get_maxIndex(sum))
                        for wi in encode.code_list:
                            times = step - self.last_update_weight[wi]
                            self.sum_weight_matrix[wi] += self.weight_matrix[wi]*times
                            self.weight_matrix[wi][self.get_maxIndex(sum)] -= 1
                            # self.weight_matrix[wi] += punish_vec
                            self.sum_weight_matrix[wi] += self.weight_matrix[wi]*times
                        for wi in encode.code_list:
                            self.last_update_weight[wi] = step
                        loss += 1
                    else:
                        cor+=1
                    total+=1
                    step += 1
                left_bound += parameter.batch_size
                right_bound += parameter.batch_size
                if right_bound >= max_len:
                    right_bound = max_len - 1
            for i in range(parameter.depth):
                times = step - self.last_update_weight[i]
                self.sum_weight_matrix[i] += self.weight_matrix[i] * times
                self.last_update_weight[i] = step
            print('训练时间：',time.time()-starttime)
            print('train accuarcy:',cor/total)
            print('loss:',loss )
            self.eval(dev_encodes, 'dev',parameter)
            # self.eval(test_encodes, 'test')
            if cor/total == 1.0:
                break
            train_encodes = self.encode_random(train_encodes)
        print('-------------------------')

    def generate_punish_vec(self,num,punish_num,rand_a,rand_b,punish_index):
        # if punish_num>rand_a or punish_num>rand_b:
        #     print('punish_num should be greater than other_num')
        #     return None
        # random.seed(33)
        punish_vec = []
        for i in range(num):
            if i == punish_index:
                punish_vec.append(punish_num)
            else:
                punish_vec.append(random.uniform(rand_a,rand_b))
        return punish_vec

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

    def eval(self,encodes,dataset_name,parameter):
        cor =0
        total= 0
        for e in encodes:
            sum = np.array([0.0 for i in range(parameter.class_num)])
            for cl in e.code_list:
                sum+=self.sum_weight_matrix[cl]
            if self.get_maxIndex(sum) == self.get_maxIndex(e.label):
                cor+=1
            total+=1
        if dataset_name=='dev' and cor/total > 0.4:
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

    def loss(self,list):
        return -1*math.log2(max(list))