import random
import pandas as pd
import numpy as np
import torch

PAD_TAG = '[pad]'  # 0
UNK_TAG = '[unk]'
START_TAG = '<s>'
END_TAG = '</s>'
SEP_TAG = '<sep>'
MASK_TAG = '[mask]'


def ger_label_index(label, l):
    label = [int(i) for i in label]
    b = np.zeros(l, dtype=int)
    # b = [-100] * l
    for i in label:
        if i != -1 and i < l:
            b[i] = 1
    return list(b)


def load_tagger_data(path, token2idx, max_length=16):
    df = pd.read_csv(path)
    inputs_pd, labels_pd = [], []
    sen1, label = df['sen1'].apply(lambda x: x.lower()).to_list(), df['s_index'].to_list()
    for x, y in zip(sen1, label):
        x, y = x.split(" ")[:max_length], y.split(" ")[:max_length]
        y = ger_label_index(y, len(x))
        y = [-100] + y + [-100]
        x = [START_TAG] + x + [END_TAG]
        x = [token2idx.get(i, token2idx[UNK_TAG]) for i in x]

        x = x + [0] * (max_length + 2 - len(x))
        y = y + [-100] * (max_length + 2 - len(y))

        inputs_pd.append(x)
        labels_pd.append(y)
    return torch.LongTensor(inputs_pd), torch.LongTensor(labels_pd)


def loadVocab(vocab_fpath, oov_mark=False):
    ## 第0个id 默认为 pad
    vocab = [PAD_TAG, UNK_TAG, START_TAG, END_TAG, SEP_TAG, MASK_TAG]
    vocab += [line.split()[0] for line in open(vocab_fpath, 'r', encoding="utf8").read().splitlines()]

    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    assert idx2token[0] == PAD_TAG
    return token2idx, idx2token


def load_text(path, maxlen=16, lowercase=True, is_debug=False, is_val=False, mode='ISNet_X'):
    df = pd.read_csv(path)
    if lowercase:
        for i in df.columns:
            if i in ['question1', 'question2', 'sen1', 'sen2', 'first', 'second', 'self_first', 'self_second',
                     'stagger_first', 'stagger_second']:
                df[i] = df[i].apply(lambda x: x.lower())
    if is_debug:
        df = df[:1]

    tgt_sen = [i.strip().split(" ")[:maxlen] for i in df.sen2.to_list()]
    labels = [[START_TAG] + x + [END_TAG] for x in tgt_sen]

    is_val_dic = {'upper_bound': ['first', 'second'], 'ISNet_X': ['self_first', 'self_second'],
                  'Stagger': ['stagger_first', 'stagger_second']}
    inputs_name = is_val_dic[mode]
    src_1 = [i.strip().split(" ")[:maxlen] for i in df[inputs_name[0]].to_list()]
    src_2 = [i.strip().split(" ")[:maxlen] for i in df[inputs_name[1]].to_list()]
    input_1 = [[START_TAG] + x + [END_TAG] for x in src_1]
    input_2 = [[START_TAG] + x + [END_TAG] for x in src_2]
    return input_1, input_2, labels


def load_data_tensor(path, vocabPath, maxlen=17, lowercase=True, is_debug=False, is_val=False,
                     mode='ISNet_X') -> object:
    inputs_1_pd, inputs_2_pd, labels_pd = [], [], []
    token2idx, _ = loadVocab(vocabPath)

    input1, input2, labels = load_text(path, maxlen, lowercase=lowercase, is_val=is_val, mode=mode)
    for x, y, z in zip(input1, input2, labels):
        x = [token2idx.get(t, token2idx[UNK_TAG]) for t in x]
        y = [token2idx.get(t, token2idx[UNK_TAG]) for t in y]
        z = [token2idx.get(t, token2idx[UNK_TAG]) for t in z]
        x = x + [0] * (maxlen + 2 - len(x))  # 默认0为 pad
        y = y + [0] * (maxlen + 2 - len(y))  # 默认0为 pad
        z = z + [0] * (maxlen + 2 - len(z))  # 默认0为 pad
        inputs_1_pd.append(x)
        inputs_2_pd.append(y)
        labels_pd.append(z)
    return torch.tensor(inputs_1_pd), torch.tensor(inputs_2_pd), torch.LongTensor(labels_pd)


def convert_id_to_token(tokens, idx2token, cut=False, end_symbl='</s>'):
    tokens = tokens.numpy().tolist()
    result = [idx2token[i] for i in tokens]
    result.append(end_symbl)  # 防止结果没有</s>
    if cut:
        result = result[1:]
        index = result.index(end_symbl)
        result = result[:index]
    return [str(i) for i in result]
