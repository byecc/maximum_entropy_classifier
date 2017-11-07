from processdata import *
from train import *

r = Read_data()
r.process_file('data/raw.clean.train')
ngram_dict = r.create_ngram_dict()
r_dev = Read_data()
r_dev.process_file('data/raw.clean.dev')
# print(len(ngram_dict))
train_encoder = Encode()
train_encodes = train_encoder.encode(r.result,ngram_dict)
dev_encodes = train_encoder.encode(r_dev.result,ngram_dict)
# for e in encodes:
#     e.show()
train = Train()
train_encodes = train.encode_random(train_encodes)
train.create_weight_matrix(len(ngram_dict),5)
train.train(train_encodes,512,dev_encodes)
