import numpy as np
import math
from processdata import Encode
from numpy import *

class APTrain:
    def __init__(self):
        self.weight_matrix = np.array((0,0))
        self.sum_weight_matrix = np.array((0,0))
        self.last_update_weight = []

    def create_weight_matrix(self,depth,width):
        self.weight_matrix = np.zeros((depth,width))
        self.sum_weight_matrix = np.zeros((depth,width))
        self.last_update_weight = [0 for i in range(depth)]

    def train(self,parameter,train_encodes,dev_encodes):
        step = 0
        for i in range(parameter.ap_iter_num):
            print('第%d轮迭代：'%(i+1))
            total = cor = loss = left_bound = 0
            right_bound = parameter.ap_batch_size
            max_len = len(train_encodes)
            while left_bound<max_len:    #batch
                for encode in train_encodes[left_bound:right_bound]:
                    sum = np.array([0.0,0.0,0.0,0.0,0.0])
                    for e in encode.code_list:
                        sum += self.weight_matrix[e]
                    if self.get_maxIndex(sum) != self.get_maxIndex(encode.label):
                        for wi in encode.code_list:
                            times = step - self.last_update_weight[wi]
                            self.sum_weight_matrix[wi] += self.weight_matrix[wi]*times
                            self.weight_matrix[wi][self.get_maxIndex(sum)] -= 1
                            self.sum_weight_matrix[wi] += self.weight_matrix[wi]
                            self.last_update_weight[wi] = step
                        loss +=1
                    else:
                        cor+=1
                    total+=1
                    step += 1
                left_bound += parameter.batch_size
                right_bound += parameter.batch_size
                if right_bound >= max_len:
                    right_bound = max_len - 1

            # for encode in train_encodes:
            #     sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            #     for e in encode.code_list:
            #         sum += self.weight_matrix[e]
            #     if self.get_maxIndex(sum) != self.get_maxIndex(encode.label):
            #         for wi in encode.code_list:
            #             times = step - self.last_update_weight[wi]
            #             self.sum_weight_matrix[wi] += self.weight_matrix[wi] * times
            #             self.weight_matrix[wi][self.get_maxIndex(sum)] -= 1
            #             self.sum_weight_matrix[wi] += self.weight_matrix[wi]
            #             self.last_update_weight[wi] = step
            #         loss += 1
            #     else:
            #         cor += 1
            #     total += 1
            #     step += 1
            print('train accuarcy:',cor/total)
            print('loss:',loss )
            self.eval(dev_encodes, 'dev')
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
                sum+=self.sum_weight_matrix[cl]
            if self.get_maxIndex(self.softmax(sum)) == self.get_maxIndex(e.label):
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
        random.seed(100)
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