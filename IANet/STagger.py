import torch
import torch.nn as nn
import os
import time
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from .dataloader import loadVocab, load_tagger_data


class STagger(nn.Module):
    def __init__(self, vocab=8000):
        super(STagger, self).__init__()
        self.embed = nn.Embedding(vocab, 300)
        self.LSTM = nn.LSTM(input_size=300, hidden_size=512, num_layers=2, proj_size=2)

    def forward(self, x):
        x = self.embed(x)
        output, (hn, cn) = self.LSTM(x)
        output = output
        return output  # batch len class_num:2


def binary_acc(preds, labels, save=False):
    if save:

        s_preds = torch.softmax(preds, dim=2)
        s_preds = s_preds[:, 1:-1, :]
        s_preds = torch.argmax(s_preds, dim=2)
        predicts = []
        for i in s_preds:
            temp = []
            for idx, j in enumerate(i.float()):
                if j == 1:
                    temp.append(idx)
            if not temp:
                temp = [-1]
            temp = " ".join([str(i) for i in temp])
            predicts.append(temp)
    preds = preds.reshape(-1, 2)
    preds = torch.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)
    labels = labels.flatten()
    label_not_ingore = labels * (labels != -100)
    positive_correct = torch.sum((labels * preds) == 1)
    positive_num = torch.sum(labels == 1)
    recall = positive_correct / positive_num if positive_num != 0 else 1

    correct_num = torch.sum(preds == labels).float()
    igore_num = torch.sum(labels == -100).float()
    acc = correct_num / (len(labels) - igore_num)
    # preds.shape=(16, 2) labels.shape=torch.Size([16, 1])
    if save:
        return acc, recall, predicts
    return acc, recall


EPOCHS = 60
BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
MAX_LEN = 16
LOWERCASE = True
D_MODEL = 512
tqdm_print_step = 1

task_name = 'Stagger_mscoco'
log_path = './runs'
base_path = 'data/mscoco/ACL/new'
vocab_path = '/data/mscoco/new/lowercase.mscoco.11000'
train_path = os.path.join(base_path, 'train_flatten.csv')
val_path = os.path.join(base_path, 'test.csv')
test_path = os.path.join(base_path, 'test.csv')

cur_time = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
model_save_path = 'model_save/' + task_name
test_txt_save_path = 'test_out/' + task_name + '/'

if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

if not os.path.exists(test_txt_save_path):
    os.mkdir(test_txt_save_path)

writer = SummaryWriter(os.path.join(log_path, cur_time))
token2idx, idx2token = loadVocab(vocab_path)
vocab_size = len(token2idx)
start_id = token2idx.get('<s>')

train_data = load_tagger_data(train_path, token2idx=token2idx, max_length=MAX_LEN)
val_data = load_tagger_data(val_path, token2idx=token2idx, max_length=MAX_LEN)
train_data = TensorDataset(*train_data)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

val_data = TensorDataset(*val_data)
val_dataloader = DataLoader(val_data, sampler=None, batch_size=VAL_BATCH_SIZE)
dataloader = {'train': train_dataloader, 'val': val_dataloader}
num_of_batch = len(train_dataloader)

model = STagger(vocab=vocab_size)
model.cuda()
ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0]).cuda(), ignore_index=-100)  #
optimizer = Adam(model.parameters(), lr=2e-5, betas=(0.9, 0.98), eps=1e-9)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def evaluate(model, save=True):
    avg_acc, avg_rec = [], []
    model.eval()
    predicts = []
    with torch.no_grad():
        for data in val_dataloader:
            src, tgt = data
            src, tgt = src.cuda(), tgt.cuda()
            output = model(src)
            acc, rec, pred = binary_acc(output, tgt, save=True)
            avg_acc.append(acc.item())
            avg_rec.append(rec.item())
            predicts += pred
    avg_acc = np.array(avg_acc).mean()
    avg_rec = np.array(avg_rec).mean()
    if save:
        return avg_acc, avg_rec, predicts
    return avg_acc, avg_rec


best_loss = float("inf")



def train(model, optimizer, global_step):
    avg_loss, avg_acc, avg_rec = [], [], []
    model.train()
    with tqdm(total=len(train_dataloader)) as pbar:
        for step, batch in enumerate(train_dataloader):

            global_step += 1
            src, tgt = batch
            src, tgt = src.cuda(), tgt.cuda()
            output = model(src)

            output_ = output.reshape(-1, output.size(-1))
            tgt_ = tgt.contiguous().view(-1)
            loss = ce_loss(output.reshape(-1, output.size(-1)),
                           tgt.contiguous().view(-1))
            avg_loss.append(loss.item())
            acc, rec = binary_acc(output, tgt)

            pbar.update(1)
            pbar.set_description(f'loss {loss:.4f}  acc: {acc:.4f} rec: {rec:.4f}')
            writer.add_scalar('loss', loss, global_step)
            writer.add_scalar('acc', acc, global_step)
            writer.add_scalar('rec', rec, global_step)

            avg_acc.append(acc.item())
            avg_rec.append(rec.item())
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    avg_rec = np.array(avg_rec).mean()
    return avg_loss, avg_acc, avg_rec, global_step



global_step = 0
max_acc = 0
max_rec = 0
total_metric = 0
for epoch in range(EPOCHS + 1):
    train_loss, train_acc, tran_rec, global_step = train(model, optimizer, global_step)
    print('epoch={},train_acc={}，Recall={}, loss={}'.format(epoch, train_acc, tran_rec, train_loss))

    test_acc, test_rec, predicts = evaluate(model, save=True)
    writer.add_scalar('val_epoch_acc', test_acc, epoch)
    print("epoch={},val_acc={},recall={}".format(epoch, test_acc, test_rec))
    if epoch > 20 and total_metric < test_acc + test_rec:
        print("save model:")
        total_metric = test_acc + test_rec
        predict_df = pd.DataFrame()
        predict_df['predicts'] = predicts
        predict_df.to_csv(test_txt_save_path + f'/epoch_{epoch}.csv')
        torch.save(model.state_dict(), f'{model_save_path}/epoch{epoch}_acc_{test_acc}.bin')
