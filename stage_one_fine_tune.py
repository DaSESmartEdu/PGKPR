import random
import pandas as pd
import sys
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from sklearn.utils import shuffle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

import os
import time
import datetime

SEED = 123
BATCH_SIZE = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
EPSILON = 1e-8
epochs = 5

MODE_MSCOCO = True  # if quora, set False
task_name = 'mscoco_cls'
data_path = ".data/mscoco/is_paraphrase/"

log_path = '.runs/mscoco/cls_runs/' + task_name
model_save_path = 'model_save/cls_model_save/' + task_name

if not os.path.exists(log_path):
    os.mkdir(log_path)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
writer = SummaryWriter(log_path)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


#  直接分词
def c_tokenizer(sen1, sen2, max_length):
    sen1, sen2 = sen1.split(" ")[:max_length], sen2.split(" ")[:max_length]
    # pad: 0
    sen1, sen2 = tokenizer.convert_tokens_to_ids(sen1), tokenizer.convert_tokens_to_ids(sen2)
    max_length_aux = 2 * max_length + 3
    pd_length = max_length_aux - len(sen1) - len(sen2) - 3
    input_id = [tokenizer.cls_token_id] + sen1 + [tokenizer.sep_token_id] + sen2 + [tokenizer.sep_token_id] + [
        0] * pd_length
    token_type_id = [0] * (len(sen1) + 2) + [1] * (len(sen2) + 1) + [0] * pd_length
    attention_mask = [int(i != 0) for i in input_id]

    return input_id, token_type_id, attention_mask


def custom_tokenizer(sen1, sen2, max_length):
    input_ids, token_type_ids, attention_masks = [], [], []
    for x, y in zip(sen1, sen2):
        input_id, token_type_id, attention_mask = c_tokenizer(x, y, max_length)
        input_ids.append(input_id), token_type_ids.append(token_type_id), attention_masks.append(attention_mask)
    return {'input_ids': torch.tensor(input_ids), 'token_type_ids': torch.tensor(token_type_ids),
            'attention_mask': torch.tensor(attention_masks)}



def readfile(filename):
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()
        return content


def get_dataset(df, tokenizer, shuffle=False, if_custom_token=False):
    if if_custom_token:
        tokens = custom_tokenizer(df.sen1.to_list(), df.sen2.to_list(), max_length=18)
    else:
        tokens = tokenizer(df.sen1.to_list(), df.sen2.to_list(), return_tensors="pt", padding=True, max_length=51,
                           truncation=True)
    a = df.is_duplicate.to_list()
    data = TensorDataset(tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask'],
                         torch.LongTensor(df.is_duplicate.to_list()))
    sampler = RandomSampler(data) if shuffle else None
    dataloader = DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE)
    print([tokenizer.cls_token_id] + [tokenizer.sep_token_id] + [tokenizer.sep_token_id])
    return dataloader


def data_loader(path, tokenizer, random_state=666, if_custom_token=False):
    if MODE_MSCOCO:  # data loader for mscoco
        train = pd.read_csv(path + 'train.csv')
        val = pd.read_csv(path + 'val.csv')
        for i in train.columns:
            if i in ['sen1', 'sen2']:
                train[i] = train[i].apply(lambda x: x.lower())
                val[i] = val[i].apply(lambda x: x.lower())
        train = shuffle(train)
        val = shuffle(val)
    else:  # data loader for quora
        sample_nums = [120000, 6000]
        df = pd.read_csv(path)
        df = df.dropna()
        for i in df.columns:
            if i in ['sen1', 'sen2', 'sen1', 'sen2']:
                df[i] = df[i].apply(lambda x: x.lower())
        train = df.sample(n=sample_nums[0], random_state=random_state)
        df = df.drop(index=train.index)
        df_positive = df[df["is_duplicate"] == 1]
        df_negtive = df[df["is_duplicate"] == 0]
        df_positive = df_positive.sample(n=int(sample_nums[1] / 2))
        df_negtive = df_negtive.sample(n=int(sample_nums[1] / 2))
        val = pd.concat([df_negtive, df_positive])
    return get_dataset(train, tokenizer, if_custom_token=if_custom_token), get_dataset(val, tokenizer,
                                                                                       if_custom_token=if_custom_token)


train_dataloader, val_dataloader = data_loader(path=data_path, tokenizer=tokenizer, if_custom_token=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def binary_acc(preds, labels):  # preds.shape=(16, 2) labels.shape=torch.Size([16, 1])
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()  # eq里面的两个参数的shape=torch.Size([16])
    acc = correct.sum().item() / len(correct)
    return acc


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(model, optimizer, global_step):
    t0 = time.time()
    avg_loss, avg_acc = [], []

    model.train()
    with tqdm(total=len(train_dataloader)) as pbar:
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            b_input_ids, b_input_types, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), \
                                                                 batch[2].long().to(device), batch[3].long().to(device)

            output = model(b_input_ids, token_type_ids=b_input_types, attention_mask=b_input_mask, labels=b_labels,
                           return_dict=True)
            loss, logits = output[0], output[1]

            avg_loss.append(loss.item())
            acc = binary_acc(logits, b_labels)

            pbar.update(1)
            pbar.set_description(f'loss {loss:.4f}  acc: {acc:.4f}')
            writer.add_scalar('loss', loss, global_step)
            writer.add_scalar('acc', acc, global_step)

            avg_acc.append(acc)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    return avg_loss, avg_acc, global_step


def evaluate(model):
    avg_acc = []
    model.eval()

    with torch.no_grad():
        for batch in val_dataloader:
            b_input_ids, b_input_types, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), \
                                                                 batch[2].long().to(device), batch[3].long().to(device)

            output = model(b_input_ids, token_type_ids=b_input_types, attention_mask=b_input_mask)

            acc = binary_acc(output[0], b_labels)
            avg_acc.append(acc)
    avg_acc = np.array(avg_acc).mean()
    return avg_acc


# evaluate and save
global_step = 0
max_acc = 0
for epoch in range(epochs + 1):
    train_loss, train_acc, global_step = train(model, optimizer, global_step)
    print('epoch={},train acc={}, loss={}'.format(epoch, train_acc, train_loss))
    test_acc = evaluate(model)
    writer.add_scalar('val_epoch_acc', test_acc, epoch)
    print("epoch={}, train acc={}".format(epoch, test_acc))

    if test_acc > max_acc:
        max_acc = test_acc
        print("save model:")
        torch.save(model.state_dict(), f'{model_save_path}/epoch{epoch}_acc_{test_acc}.bin')
