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
from .transformer.lossfunction import LossCompute, LabelSmoothing
from .evaluation import test_bleu_rouge, normalize_txt
from .dataloader import loadVocab, load_data_tensor

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

EPOCHS = 40
BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
MAX_LEN = 16
LOWERCASE = True
D_MODEL = 512
tqdm_print_step = 1

task_name = 'quora_ISNet'
log_path = './runs'
base_path = 'data/Quora/nltk/ACL'
vocab_path = 'data/Quora/nltk/quora.ntlk.8000'
train_path = os.path.join(base_path, 'train.csv')
val_path = os.path.join(base_path, 'test.csv')
# test_path = os.path.join(base_path, 'test.csv')

cur_time = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
model_save_path = 'model_save/' + task_name
test_txt_save_path = 'test_out/' + task_name + '/'
test_txt_stagger = 'test_out/' + task_name + '/stagger_'

if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

if not os.path.exists(test_txt_save_path):
    os.mkdir(test_txt_save_path)

writer = SummaryWriter(os.path.join(log_path, task_name))
token2idx, idx2token = loadVocab(vocab_path)
vocab_size = len(token2idx)
start_id = token2idx.get('<s>')

# data loader
# load_data_tensor return: inputs_pd, mid_label_pd, mid_indexs, labels_pd
train_data = load_data_tensor(train_path, vocab_path, MAX_LEN, is_debug=False, mode='upper_bound')
val_data = load_data_tensor(val_path, vocab_path, MAX_LEN, is_debug=False, mode='ISNet_X')
val_data_stagger = load_data_tensor(val_path, vocab_path, MAX_LEN, is_debug=False, mode='Stagger')

val_df = pd.read_csv(val_path)
val_tgt_string = val_df['sen2'].apply(lambda x: x.lower()).to_list()  # for bleu, rouge test
val_tgt_string = normalize_txt(val_tgt_string, token2idx, idx2token, '[unk]', max_len=16)
val_src_string = val_df['sen1'].apply(lambda x: x.lower()).to_list()  # for src_bleu, key_acc test
# val_indexes = [[int(i) for i in j.strip().split(" ")] for j in val_df.s_index.to_list()]

train_data = TensorDataset(*train_data)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

val_data = TensorDataset(*val_data)
val_dataloader = DataLoader(val_data, sampler=None, batch_size=VAL_BATCH_SIZE)

val_data_stagger = TensorDataset(*val_data_stagger)
val_dataloader_stagger = DataLoader(val_data_stagger, sampler=None, batch_size=VAL_BATCH_SIZE)
dataloader = {'train': train_dataloader, 'val': val_dataloader, 's_tagger': val_dataloader_stagger}
num_of_batch = len(train_dataloader)

# model
model = ACLFTransformer(d_model=D_MODEL, vocab=vocab_size, concat=False)

model = model.cuda()

# seq2seq_criterion = nn.CrossEntropyLoss(ignore_index=0)
seq2seq_criterion = nn.CrossEntropyLoss(ignore_index=0)  # [pad]: 0
seq2seq_criterion.cuda()
model_opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

best_loss = float("inf")
global_step = 0


def run_onece(model, batch1, batch2):
    # model forward: (src, src2, tgt, src_mask, src2_mask, tgt_mask)
    model_opt.zero_grad()
    final_out = model.forward(src=batch1.src, src2=batch2.src, tgt=batch1.trg, src_mask=batch1.src_mask,
                              src2_mask=batch2.src_mask, tgt_mask=batch1.trg_mask)
    s2s_loss_ = seq2seq_criterion(final_out.view(-1, final_out.size(-1)),
                                  batch1.trg_y.contiguous().view(-1))

    if phase == 'train':
        s2s_loss_.backward()
        model_opt.step()
    return s2s_loss_


for epoch in range(EPOCHS):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        elif phase == 'val':
            model.eval()
        total_tokens, total_s2s_loss, total_mid_loss = 0, 0, 0
        with tqdm(total=len(dataloader[phase])) as pbar:
            for i, data in enumerate(dataloader[phase]):
                src1, src2, tgt = data
                src1, src2, tgt = src1.cuda(), src2.cuda(), tgt.cuda()
                batch_1 = Batch(src=src1, trg=tgt)
                batch_2 = Batch(src=src2, trg=tgt)
                if phase == 'val':
                    with torch.no_grad():
                        s2s_loss = run_onece(model, batch_1, batch_2)
                else:
                    s2s_loss = run_onece(model, batch_1, batch_2)

                total_s2s_loss += s2s_loss
                # total_tokens += (batch_1.ntokens + batch_2.ntokens)

                # s2s_loss = s2s_loss / (batch_1.ntokens + batch_2.ntokens)

                if phase == 'train':
                    global_step += 1
                    writer.add_scalar('s2s_loss', s2s_loss, global_step)

                if i % tqdm_print_step == 0:
                    pbar.update(tqdm_print_step)
                    pbar.set_description(f's2s: {s2s_loss:.4f}')

        epoch_loss = total_s2s_loss / len(dataloader[phase])
        if phase == 'val' and (epoch_loss < best_loss or epoch % 10 == 0) and epoch > 5:
            print("save model")
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'{model_save_path}/epoch{epoch}_loss_{epoch_loss}.pkl')
        if phase == 'val':
            print('ISNet + X:')
            measure_dic = test_bleu_rouge(dataloader['val'], model, o_target=val_tgt_string,
                                          idx2token=idx2token, epoch=epoch, max_len=MAX_LEN,
                                          start_symbol=start_id, save_path=test_txt_save_path)
            print('ISNet + S')
            measure_dic_stagger = test_bleu_rouge(dataloader['s_tagger'], model, o_target=val_tgt_string,
                                                  idx2token=idx2token, epoch=epoch, max_len=MAX_LEN,
                                                  start_symbol=start_id, save_path=test_txt_stagger)
            writer.add_scalars('Bleu_Rouge', measure_dic, epoch)
            for name, value in measure_dic.items():
                writer.add_scalar(name, value, epoch)
