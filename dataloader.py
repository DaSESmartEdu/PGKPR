import random
import pandas as pd
import numpy as np
import torch

PAD_TAG = '[pad]'  # 0
UNK_TAG = '[unk]'
UN_KEY_TAG = '[unkey]'
START_TAG = '<s>'
END_TAG = '</s>'
SEP_TAG = '<sep>'
# OOV_MAK = [f'[oov_{i}]' for i in range(50)]
POS_TAG = ['[VB]',
           '[NNS]',
           '[RBR]',
           '[DT]',
           '[IN]',
           '[WRB]',
           '[VBZ]',
           '[NNP]',
           '[CD]',
           '[(]',
           '[NN]',
           '[VBG]',
           '[``]',
           '[PRP$]',
           "['']",
           '[VBD]',
           '[PDT]',
           '[CC]',
           '[POS]',
           '[LS]',
           '[$]',
           '[JJ]',
           '[.]',
           '[,]',
           '[#]',
           '[RB]',
           '[RP]',
           '[:]',
           '[VBP]',
           '[MD]',
           '[PRP]',
           '[)]',
           '[VBN]',
           '[FW]',
           '[SYM]']


def loadVocab(vocab_fpath, padTag=PAD_TAG, oov_mark=False, lowercase=True):
    vocab = [PAD_TAG, UNK_TAG, START_TAG, END_TAG, SEP_TAG, UN_KEY_TAG] + [i.lower() for i in POS_TAG]
    if not lowercase:
        vocab = [PAD_TAG, UNK_TAG, START_TAG, END_TAG, SEP_TAG, UN_KEY_TAG] + [i for i in POS_TAG]
    # if oov_mark:
    #     vocab += OOV_MAK
    vocab += [line.split()[0] for line in open(vocab_fpath, 'r', encoding="utf8").read().splitlines()]

    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    assert idx2token[0] == PAD_TAG
    return token2idx, idx2token


def ger_index_array(index, l, maxlen):
    a = np.ones(l + 2)
    b = np.zeros(l + 1)
    # b = np.array([5] * (l + 1))  # un_key_tag: 5
    for i in index:
        if i != -1 and i < l:
            b[i] = 1
    b[-1] = 1
    c = np.zeros(2 * maxlen - 2 * l)
    a = np.append(a, b)
    return np.append(a, c)


def unkey_tager(index, sen):
    res = [UN_KEY_TAG] * len(sen)
    for i in index:
        if i != -1 and i < len(sen):
            res[i] = sen[i]
    return res


def load_text(path, maxlen=25, lowercase=True, isdebug=False, switch_pos=True, reverse=False):
    df = pd.read_csv(path)

    if lowercase:
        for i in df.columns:
            if i in ['question1', 'question2', 'sen1', 'sen2', 'pos1', 'pos2', 's_sen1', 's_pos1', 's_sen2', 's_pos2']:
                df[i] = df[i].apply(lambda x: x.lower())
    if isdebug:
        df = df[:1]

    tgt_sen = [i.strip().split(" ")[:maxlen] for i in df.sen2.to_list()]
    labels = [[START_TAG] + x + [END_TAG] for x in tgt_sen]
    #     tgt_pos = [i.strip().split(" ")[:maxlen] for i in df.pos2.to_list()]
    if not switch_pos:
        src_sen = [i.strip().split(" ")[:maxlen] for i in df.sen1.to_list()]
        inputs = [[START_TAG] + x + [END_TAG] for x in src_sen]
        return inputs, labels

    s_src_sen = [i.strip().split(" ")[:maxlen] for i in df.s_sen1.to_list()]
    s_src_pos = [i.strip().split(" ")[:maxlen] for i in df.s_pos1.to_list()]
    if reverse:
        inputs = [[START_TAG] + x + [SEP_TAG] + y + [END_TAG] for x, y in zip(s_src_sen, s_src_pos)]
    else:
        inputs = [[START_TAG] + x + [SEP_TAG] + y + [END_TAG] for x, y in zip(s_src_pos, s_src_sen)]

    src_sen = [i.strip().split(" ")[:maxlen] for i in df.sen1.to_list()]
    src_pos = [i.strip().split(" ")[:maxlen] for i in df.pos1.to_list()]

    mid_indexs = [str(i).split(" ") for i in df.s_index.to_list()]
    mid_indexs = [[int(i) for i in j if int(i) < maxlen] for j in mid_indexs]
    src_sen_unkey_targed = [unkey_tager(i, j) for i, j in zip(mid_indexs, src_sen)]

    mid_indexs = [ger_index_array(i, len(j), maxlen) for i, j in zip(mid_indexs, src_sen)]

    mid_label = [[START_TAG] + x + [SEP_TAG] + y + [END_TAG] for x, y in zip(src_pos, src_sen_unkey_targed)]

    # tgt_sen = [i.strip().split(" ")[:maxlen] for i in df.sen2.to_list()]
    #     tgt_pos = [i.strip().split(" ")[:maxlen] for i in df.pos2.to_list()]
    # labels = tgt_sen

    return inputs, mid_label, mid_indexs, labels


def load_text_con(path, maxlen=25, lowercase=True, isdebug=False, switch_pos=True, reverse=False):
    df = pd.read_csv(path)

    if lowercase:
        for i in df.columns:
            if i in ['question1', 'question2', 'sen1', 'sen2', 'pos1', 'pos2', 's_sen1', 's_pos1', 's_sen2', 's_pos2']:
                df[i] = df[i].apply(lambda x: x.lower())
    if isdebug:
        df = df[:1]

    tgt_sen = [i.strip().split(" ")[:maxlen] for i in df.sen2.to_list()]
    labels = [[START_TAG] + x + [END_TAG] for x in tgt_sen]

    s_src_sen = [i.strip().split(" ")[:maxlen] for i in df.s_sen1.to_list()]
    s_src_pos = [i.strip().split(" ")[:maxlen] for i in df.s_pos1.to_list()]

    inputs = [[START_TAG] + x + [SEP_TAG] + y + [END_TAG] for x, y in zip(s_src_pos, s_src_sen)]

    src_sen = [i.strip().split(" ")[:maxlen] for i in df.sen1.to_list()]
    src_pos = [i.strip().split(" ")[:maxlen] for i in df.pos1.to_list()]

    label_sen = [i.strip().split(" ")[:maxlen] for i in df.sen2.to_list()]
    label_pos = [i.strip().split(" ")[:maxlen] for i in df.pos2.to_list()]

    mid_indexs = [str(i).split(" ") for i in df.s_index.to_list()]
    mid_indexs = [[int(i) for i in j if int(i) < maxlen] for j in mid_indexs]
    src_sen_unkey_targed = [unkey_tager(i, j) for i, j in zip(mid_indexs, src_sen)]

    mid_label = [[START_TAG] + x + [SEP_TAG] + y + [END_TAG] for x, y in zip(src_pos, src_sen_unkey_targed)]

    # label_with_pos = [[START_TAG] + x + [SEP_TAG] + y + [END_TAG] for x, y in zip(label_sen, label_pos)]
    label_with_pos = [[START_TAG] + y + [SEP_TAG] + x + [END_TAG] for x, y in zip(label_sen, label_pos)]

    return inputs, mid_label, labels, label_with_pos


def load_data_tensor(path, vocabPath, maxlen=25, lowercase=True, switch_pos=True, reverse=False):
    inputs_pd, mid_label_pd, labels_pd = [], [], []
    token2idx, _ = loadVocab(vocabPath)
    if not switch_pos:
        inputs, labels = load_text(path, maxlen, lowercase=lowercase, switch_pos=switch_pos, reverse=reverse)
        for x, y in zip(inputs, labels):
            x = [token2idx.get(t, token2idx[UNK_TAG]) for t in x]
            y = [token2idx.get(t, token2idx[UNK_TAG]) for t in y]
            x = x + [0] * (maxlen + 2 - len(x))  # 默认0为 pad
            y = y + [0] * (maxlen + 2 - len(y))  # 默认0为 pad
            inputs_pd.append(x)
            labels_pd.append(y)
        return torch.tensor(inputs_pd), torch.LongTensor(labels_pd)

    inputs, mid_label, mid_indexs, labels = load_text(path, maxlen, lowercase=lowercase, reverse=reverse)
    padded_length = maxlen * 2 + 3
    for x, y, z, k in zip(inputs, mid_label, labels, mid_indexs):
        x = [token2idx.get(t, token2idx[UNK_TAG]) for t in x]
        y = [token2idx.get(t, token2idx[UNK_TAG]) for t in y]
        z = [token2idx.get(t, token2idx[UNK_TAG]) for t in z]

        x = x + [0] * (padded_length - len(x))  # 默认0为 pad
        y = y + [0] * (padded_length - len(y))  # 默认0为 pad
        z = z + [0] * (padded_length - len(z))

        inputs_pd.append(x)
        mid_label_pd.append(y)
        labels_pd.append(z)
        assert len(x) == len(k)
    return torch.tensor(inputs_pd), torch.LongTensor(mid_label_pd), \
           torch.LongTensor(mid_indexs), torch.LongTensor(labels_pd)


def load_data_tensor_con(path, vocabPath, maxlen=25, lowercase=True, switch_pos=True, reverse=False):
    inputs_pd, mid_label_pd, labels_pd, label_with_pos_pd = [], [], [], []
    token2idx, _ = loadVocab(vocabPath)

    inputs, mid_label, labels, label_with_pos = load_text_con(path, maxlen, lowercase=lowercase, reverse=reverse)
    padded_length = maxlen * 2 + 3
    for x, y, z, k in zip(inputs, mid_label, labels, label_with_pos):
        x = [token2idx.get(t, token2idx[UNK_TAG]) for t in x]
        y = [token2idx.get(t, token2idx[UNK_TAG]) for t in y]
        z = [token2idx.get(t, token2idx[UNK_TAG]) for t in z]
        k = [token2idx.get(t, token2idx[UNK_TAG]) for t in k]

        x = x + [0] * (padded_length - len(x))  # pad:0
        y = y + [0] * (padded_length - len(y))  # pad:0
        z = z + [0] * (padded_length - len(z))
        k = k + [0] * (padded_length - len(k))

        inputs_pd.append(x)
        mid_label_pd.append(y)
        labels_pd.append(z)
        label_with_pos_pd.append(k)
        assert len(x) == len(k)
    return torch.tensor(inputs_pd), torch.LongTensor(mid_label_pd), \
           torch.LongTensor(labels_pd), torch.LongTensor(label_with_pos_pd)


def convert_id_to_token(tokens, idx2token, cut=False, end_symbl='</s>'):
    tokens = tokens.numpy().tolist()
    result = [idx2token[i] for i in tokens]
    result.append(end_symbl)
    if cut:
        result = result[1:]
        index = result.index(end_symbl)
        result = result[:index]
    return result


# for pure transforner
def load_data(df_path, maxlen=16, token2idx=None, lowercase=True):
    df = pd.read_csv(df_path)
    inputs_pd, labels_pd = [], []
    if lowercase:
        src = df['sen1'].apply(lambda x: x.lower()).to_list()
        tgt = df['sen2'].apply(lambda x: x.lower()).to_list()
    else:
        src = df['sen1'].to_list()
        tgt = df['sen2'].to_list()
    src = [[START_TAG] + i.strip().split(" ")[:maxlen] + [END_TAG] for i in src]
    tgt = [[START_TAG] + i.strip().split(" ")[:maxlen] + [END_TAG] for i in tgt]
    for x, y in zip(src, tgt):
        x = [token2idx.get(t, token2idx[UNK_TAG]) for t in x]
        y = [token2idx.get(t, token2idx[UNK_TAG]) for t in y]
        x = x + [0] * (maxlen + 2 - len(x))  # 默认0为 pad
        y = y + [0] * (maxlen + 2 - len(y))  # 默认0为 pad
        inputs_pd.append(x)
        labels_pd.append(y)
    return torch.tensor(inputs_pd), torch.LongTensor(labels_pd)


def load_data_contrast(df_path, maxlen=15, token2idx=None, use_key_test=False):
    df = pd.read_csv(df_path)
    inputs_pd, labels_pd, key_pd, un_key_pd = [], [], [], []
    src = df['sen1'].apply(lambda x: x.lower()).to_list()
    tgt = df['sen2'].apply(lambda x: x.lower()).to_list()
    key = df['s_sen1'].apply(lambda x: x.lower()).to_list()
    un_key = df['s_pos1'].apply(lambda x: x.lower()).to_list()

    src = [[START_TAG] + i.strip().split(" ")[:maxlen] + [END_TAG] for i in src]
    tgt = [[START_TAG] + i.strip().split(" ")[:maxlen] + [END_TAG] for i in tgt]
    key = [[START_TAG] + i.strip().split(" ")[:maxlen] + [END_TAG] for i in key]
    un_key = [[START_TAG] + i.strip().split(" ")[:maxlen] + [END_TAG] for i in un_key]
    for x, y, z, k in zip(src, tgt, key, un_key):
        x = [token2idx.get(t, token2idx[UNK_TAG]) for t in x]
        y = [token2idx.get(t, token2idx[UNK_TAG]) for t in y]
        z = [token2idx.get(t, token2idx[UNK_TAG]) for t in z]
        k = [token2idx.get(t, token2idx[UNK_TAG]) for t in k]
        x = x + [0] * (maxlen + 2 - len(x))  # 默认0为 pad
        y = y + [0] * (maxlen + 2 - len(y))  # 默认0为 pad
        z = z + [0] * (maxlen + 2 - len(z))  # 默认0为 pad
        k = k + [0] * (maxlen + 2 - len(k))  # 默认0为 pad
        inputs_pd.append(x)
        labels_pd.append(y)
        key_pd.append(z)
        un_key_pd.append(k)
    if use_key_test:
        return torch.LongTensor(key_pd), torch.LongTensor(labels_pd), torch.LongTensor(key_pd), torch.LongTensor(un_key_pd)
    return torch.tensor(inputs_pd), torch.LongTensor(labels_pd), torch.LongTensor(key_pd), torch.LongTensor(un_key_pd)
