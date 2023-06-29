import nltk
import os
from tqdm import tqdm
import json
import pandas as pd
import swifter
from vocab import Vocab
import random


def get_pos(x):
    x = x.split(" ")
    return " ".join(['[' + i[1] + ']' for i in nltk.pos_tag(x)])


def split(df, sample_nums=[100000, 4000, 20000], random_state=666):
    """
    sample_num: [train, val, test]
    """
    train = df.sample(n=sample_nums[0], random_state=random_state)
    df = df.drop(index=train.index)
    val = df.sample(n=sample_nums[1], random_state=random_state)
    df = df.drop(index=val.index)
    test = df.sample(n=sample_nums[2], random_state=random_state)
    return train, val, test


def split_quora(data_path, out_path):
    df = pd.read_csv(data_path, [100000, 4000, 20000], 666)
    df['sen1'] = df['question1'].apply(lambda x: " ".join(nltk.word_tokenize(x)))
    df['sen2'] = df['question2'].apply(lambda x: " ".join(nltk.word_tokenize(x)))
    df['pos1'] = df['sen1'].swifter.apply(get_pos)
    df['pos2'] = df['sen2'].swifter.apply(get_pos)
    df.to_csv(out_path + 'total.csv')
    dataset = split(df)
    for d, n in zip(dataset, ['train.csv', 'val.csv', 'test.csv']):
        d.to_csv(out_path + n, index=False)


def split_mscoco(datapath, out_path):
    def mscoco_read_json(file_path, bleu_baseline=False):
        print("Reading mscoco raw data .. ")
        print("  data path: %s" % file_path)
        with open(file_path, "r") as fd:
            data = json.load(fd)

        print("%d sentences in total" % len(data["annotations"]))

        # aggregate all sentences of the same images
        image_idx = set([d["image_id"] for d in data["annotations"]])
        paraphrases = {}
        for im in image_idx: paraphrases[im] = []
        for d in tqdm(data["annotations"]):
            im = d["image_id"]
            sent = d["caption"]
            paraphrases[im].append(sent)
        sentence_sets = [paraphrases[im] for im in paraphrases if len(paraphrases[im]) == 5]
        return sentence_sets

    data = mscoco_read_json(datapath, bleu_baseline=True)
    df = pd.DataFrame(columns=[f'question{j}' for j in range(1, 6)])
    for idx, i in enumerate([f'question{j}' for j in range(1, 6)]):
        d = [t[idx] for t in data]
        df[i] = d

    data = df.copy()
    for i in df.columns:
        data[i] = df[i].swifter.apply(lambda x: " ".join(nltk.word_tokenize(x)))

    for i in range(1, 6):
        data[f'pos{i}'] = data[f'question{i}'].swifter.apply(get_pos)
    data.to_csv(out_path + 'total.csv', index=False)
    data['sen1'] = data['question1']
    data['sen2'] = data['question2']
    dataset = split(data, sample_nums=[93000, 4000, 20000], random_state=666)

    for idx, i in enumerate(['train.csv', 'val.csv', 'test.csv']):
        dataset[idx].to_csv(out_path + i, index=False)
    train = dataset[0]
    first = train.copy()
    for i in range(2, 6):
        t = first.copy()
        t['sen1'] = t['question1']
        t['sen2'] = t[f'question{i}']
        train = train.append(t)
    train = train.reset_index()
    train = train.drop(columns=['index'])
    train.to_csv(out_path + 'train_flatten.csv', index=False)


def get_vocab(base_path, dataset='mscoco'):
    vocab = Vocab()
    vocab_size_dic = {'quora': 8000, 'mscoco': 1100}
    vocab_size = vocab_size_dic[dataset]

    def read_data_from_list(data, lowercase=True):
        for num, sen in enumerate(data):
            for word in sen.strip().split(" "):
                if lowercase:
                    vocab.add_word(word.lower())
                else:
                    vocab.add_word(word)
        print('End', num)

    if dataset == 'mscoco':
        df = pd.read_csv(base_path + 'train.csv')
        for i in range(1, 6):
            read_data_from_list(df[f'question{i}'].to_list(), lowercase=True)
    elif dataset == 'quora':
        df = pd.read_csv(base_path + 'total.csv')
        read_data_from_list(df['sen1'].to_list(), lowercase=True)
        read_data_from_list(df['sen2'].to_list(), lowercase=True)
    vocab.save_vocab(base_path + f'{dataset}.vocab')
    vocab = Vocab()
    vocab.load_vocab_from_file(base_path + f'{dataset}vocab')
    vocab.limit_vocab_length(vocab_size)
    vocab.save_vocab_for_nmt(base_path + f'dataset.{str(vocab_size)}')


def ger_stage_one_mscoco(inpath, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    def sample_neg(df, i, n):
        smps = df.drop(index=i).sample(n=n)
        sen1 = [df.loc[i].sen1] * n
        sen2 = smps.sen2.to_list()
        return sen1, sen2

    def sample_neg_df(df, n):
        res = pd.DataFrame(columns=df.columns)
        sen1, sen2 = [], []
        for i in tqdm(df.index):
            t1, t2 = sample_neg(df, i, n)
            sen1 += t1
            sen2 += t2
        res['sen1'] = sen1
        res['sen2'] = sen2
        res['is_duplicate'] = [0] * len(sen1)
        return res

    def is_parapharase_ger(df, neg_n=4):
        df['is_duplicate'] = [1] * df.shape[0]
        net = sample_neg_df(df, n=neg_n)
        pos = df.copy()
        pos['sen2'] = pos['sen1']
        df = df.append(pos)
        df = df.append(net)
        return df

    drop_col = ['question1', 'question2', 'pos1', 'pos2', 'question3', 'question4', 'question5']
    val = pd.read_csv(inpath + 'val.csv').drop(columns=drop_col)
    train = pd.read_csv(inpath + 'train.csv').drop(columns=drop_col)
    val = is_parapharase_ger(val, neg_n=2)
    train = is_parapharase_ger(train, neg_n=4)
    val.to_csv(out_path + 'val.csv', index=False)
    train.to_csv(out_path + 'train.csv', index=False)









