import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


model_weight_path = 'you_model_weight_path'

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 0
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

model.load_state_dict(torch.load(model_weight_path))

SEED = 8
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print('device:', device)

bert_embedding = model.bert.embeddings.to(device)

values = bert_embedding.word_embeddings.weight.data
shape = bert_embedding.word_embeddings.weight.data.shape


def convert_text_to_token(tokenizer, sentence, limit_size=126):
    tokens = tokenizer.encode(sentence[:limit_size])
    if len(tokens) < limit_size + 2:
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens


def get_gradients(embeddings, model, attention_mask):
    embeddings.retain_grad()
    output = model(inputs_embeds=embeddings, token_type_ids=None,
                   attention_mask=attention_mask)
    res = output[0] if len(output) < 2 else output[1]
    label_index = torch.argmax(res, dim=1)


    label = res[:, label_index][:, 0]
    label.backward(torch.full(label.shape, 1.0).to(device), retain_graph=True)
    grads = embeddings.grad
    return grads, output


def get_scores(grad):
    """
    grade: shape [batch, sequencelength]
    """
    res = grad
    max_ = np.max(res, axis=1)
    min_ = np.min(res, axis=1)
    min_ = min_.reshape((min_.shape[0], 1))
    max_ = max_.reshape((max_.shape[0], 1))
    return (res - min_) / (max_ - min_).round(4)


def merge(token, score):
    res = []
    num_sum = 1  # num of a merged token
    for i, j in zip(token, score):
        if i == '[CLS]':
            continue
        if i == '[SEP]':
            if len(res) > 0:
                res[-1][0] /= num_sum
            break
        if len(i) > 2 and i[:2] == '##':
            res[-1][1] += i[2:]
            res[-1][0] += j
            num_sum += 1
        else:
            if len(res) > 0:
                res[-1][0] /= num_sum
            num_sum = 1
            res.append([j, i])
    score_ = [i[0] for i in res]
    score_ = np.array(score_)
    if score_.min() != score_.max():
        score_ = (score_ - score_.min()) / (score_.max() - score_.min())
    score_ = [str(i) for i in score_]
    word_ = [i[1] for i in res]
    return " ".join(score_), " ".join(word_)



def data_loader(path, mode='xy'):
    df = pd.read_csv(path)
    for i in df.columns:
        if i in ['question1', 'question2', 'sen1', 'sen2']:
            df[i] = df[i].apply(lambda x: x.lower())
    if mode == 'xx':
        tokens = tokenizer(df.question1.to_list(), df.question2.to_list(),
                           return_tensors="pt", padding=True, max_length=51, truncation=True)
    else:
        tokens = tokenizer(df.question1.to_list(), df.question1.to_list(),
                           return_tensors="pt", padding=True, max_length=51, truncation=True)

    data = TensorDataset(tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask'])
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=BATCH_SIZE)
    return dataloader


BATCH_SIZE = 128

data_type = ['train', 'val', 'test']


def run_dataset(in_path, out_path, mode='xy'):
    cur_dataloader = data_loader(path=in_path, mode=mode)
    res_scores, res_words = [], []
    for batch in tqdm(cur_dataloader):
        b_input_ids, b_input_types, b_input_mask = batch[0].long().to(device), batch[1].long().to(device), \
                                                   batch[2].long().to(device)
        pred_grads = []
        model.zero_grad()
        embeddings = bert_embedding(b_input_ids, token_type_ids=b_input_types)
        pred_grad, outputs = get_gradients(embeddings, model, b_input_mask)
        pred_grad = pred_grad.cpu()
        pred_grads.append(pred_grad.detach().numpy())
        pred_grads = np.mean(pred_grads, 0)
        scores = np.sqrt((pred_grads ** 2).sum(axis=-1))
        # scores = get_scores(scores)

        words = [tokenizer.convert_ids_to_tokens(i) for i in b_input_ids]

        for w, s in zip(words, scores):
            t_scores, t_words = merge(w, s)
            res_scores.append(t_scores)
            res_words.append(t_words)

    df = pd.read_csv(in_path)
    if mode == 'xy':
        df['bert_sen1'] = res_words
    if mode == 'xy':
        df['xy_scores'] = res_scores
    elif mode == 'xx':
        df['xx_scores'] = res_scores
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    in_path = 'data/mscoco/'
    out_path = 'data/mscoco/'
    # run_dataset(in_path=in_path + 'val.csv', out_path=out_path + 'val.csv', mode='xy')
    for i in ['val.csv', 'train_flatten.csv', 'test.csv']:
        run_dataset(in_path=in_path + i, out_path=out_path + i, mode='xy')

    for i in ['val.csv', 'train_flatten.csv', 'test.csv']:
        run_dataset(in_path=out_path + i, out_path=out_path + i, mode='xx')
