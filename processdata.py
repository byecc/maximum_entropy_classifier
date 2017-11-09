import re
import pickle
import os
import random

random.seed(5)
class Instance:
    def __init__(self):
        self.sentence = ''
        self.label = -1

    def show(self):
        print(self.sentence,' ',self.label)

class Code:
    def __init__(self):
        self.code_list = []
        self.label = []

    def show(self):
        print(self.code_list,' ',self.label)

class Read_data:
    def __init__(self):
        self.result = []

    def clean_str(self,string):
        """
                Tokenization/string cleaning for all datasets except for SST.
                Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def process_file(self,path):
        result = []
        with open(path,'r') as f:
            for o in f.readlines():
                info = o.strip().split('|||')
                inst = Instance()
                inst.sentence = self.clean_str(info[0])
                inst.label = self.clean_str(info[1])
                result.append(inst)
        self.result =  result

    def create_ngram_dict(self):
        if os.path.exists('ngram_dict.pkl'):
            return pickle.load(open('ngram_dict.pkl','rb'))
        else:
            ngram_list = []
            for r in self.result:
                s = r.sentence.split(' ')
                for i in range(len(s)):
                    ngram_list.append('unigram='+s[i])
                for i in range(len(s)-1):
                    ngram_list.append('bigram='+s[i]+'#'+s[i+1])
                for i in range(len(s)-2):
                    ngram_list.append('trigram='+s[i]+'#'+s[i+1]+'#'+s[i+2])
            ngram_dict = {}
            ngram_list = list(set(ngram_list))
            for i in range(len(ngram_list)):
                ngram_dict[ngram_list[i]] = i
            ngram_dict['-unknown-']=len(ngram_list)
            pickle.dump(ngram_dict,open('ngram_dict.pkl','wb'))
            return ngram_dict

class Encode:
    def __init__(self):
        pass

    def encode(self,data_result,dict):
        encodes = []
        for r in data_result:
            e = Code()
            s = r.sentence.split(' ')
            for i in range(len(s)):
                if 'unigram='+s[i] in dict.keys():
                    e.code_list.append(dict['unigram='+s[i]])
                else:
                    e.code_list.append(dict['-unknown-'])
            for i in range(len(s)-1):
                if 'bigram='+s[i]+'#'+s[i+1] in dict.keys():
                    e.code_list.append(dict['bigram='+s[i]+'#'+s[i+1]])
                else:
                    e.code_list.append(dict['-unknown-'])
            for i in range(len(s)-2):
                if 'trigram='+s[i]+'#'+s[i+1]+'#'+s[i+2] in dict.keys():
                    e.code_list.append(dict['trigram='+s[i]+'#'+s[i+1]+'#'+s[i+2]])
                else:
                    e.code_list.append(dict['-unknown-'])
            if r.label == '0':
                e.label = [1,0,0,0,0]
            elif r.label == '1':
                e.label = [0,1,0,0,0]
            elif r.label == '2':
                e.label = [0,0,1,0,0]
            elif r.label == '3':
                e.label = [0,0,0,1,0]
            elif r.label == '4':
                e.label = [0,0,0,0,1]
            encodes.append(e)
        return encodes

