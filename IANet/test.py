import os
import time
import numpy
import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from .transformer.model import ACLFTransformer
from .transformer.optimal import get_std_opt
from .transformer.train import Batch
from .transformer.lossfunction import LossCompute
from evaluation import test_bleu_rouge, normalize_txt
from dataloader import loadVocab, load_data_tensor

EPOCHS = 40
BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
MAX_LEN = 16
LOWERCASE = True
tqdm_print_step = 1


weight_path = '.pkl'
epoch = 31

base_path = 'data/Quora/nltk/ACL'
vocab_path = 'data/Quora/nltk/quora.ntlk.8000'
test_path = os.path.join(base_path, 'test.csv')


cur_time = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())

test_txt_save_path = 'test/'

token2idx, idx2token = loadVocab(vocab_path)
vocab_size = len(token2idx)
start_id = token2idx.get('<s>')
mask_id = token2idx.get('[mask]')

# data loader
# load_data_tensor return: inputs_pd, mid_label_pd, mid_indexs, labels_pd
val_data = load_data_tensor(test_path, vocab_path, MAX_LEN, lowercase=LOWERCASE, is_val=True, mode='Stagger')
val_df = pd.read_csv(test_path)
val_tgt_string = val_df['sen2'].apply(lambda x: x.lower()).to_list()  # for bleu, rouge test
val_tgt_string = normalize_txt(val_tgt_string, token2idx=token2idx, idx2token=idx2token, UNK_TAG='[unk]')
# val_tgt_string = [i.split(" ") for i in val_tgt_string]

val_data = TensorDataset(*val_data)
val_dataloader = DataLoader(val_data, batch_size=VAL_BATCH_SIZE)

# model
model = ACLFTransformer(d_model=512, vocab=vocab_size, concat=False)
model_dict = torch.load(weight_path)
model.load_state_dict(model_dict)
model.cuda()
measure_dic = test_bleu_rouge(val_dataloader, model, o_target=val_tgt_string,
                              idx2token=idx2token, epoch=epoch, max_len=MAX_LEN,
                              start_symbol=start_id, save_path=test_txt_save_path, mask_id=mask_id)