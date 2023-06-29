import io
from tqdm import tqdm
import pandas as pd
import numpy as np


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def Rank_xi(x):
    x_ = [(x[i], i) for i in range(len(x))]
    x_ = sorted(x_)
    x_ = [(x_[i][1], i) for i in range(len(x_))]
    x_ = sorted(x_)
    return [x_[i][1] for i in range(len(x_))]


## similarity of two sentences
def Familiar(x, y):
    #     x, y = list(x), list(y)
    x, y = vectornize(x), vectornize(y)
    #     assert len(x) == len(y)
    D = len(x)
    r_x, r_y = Rank_xi(x), Rank_xi(y)

    res = 0
    for i in range(D):
        res = res + pow((r_x[i] - r_y[i]), 2)
    return 1 - (6 * res / (D * (pow(D, 2) - 1)))


## get vec of a sentence
def vectornize(asen):
    res = []
    asen = asen.strip().split(" ")
    for i in asen:
        res.append(fastText2.get(i, [-1.0] * 300))  # return min
    res = np.array(res)
    res = res.max(axis=0)
    return list(res)


#  importance of a word
def G_xi_Two(x_idx, x, y, if_abs=False):
    x_ = [x[i] for i in range(len(x)) if i != x_idx]
    f_x_y = Familiar(" ".join(x), " ".join(y))
    if if_abs:
        return abs((f_x_y - Familiar(" ".join(x_), " ".join(y))) / f_x_y)
    return (f_x_y - Familiar(" ".join(x_), " ".join(y))) / f_x_y


def PIS(x, y, alha=0.1, MASK='[MASK]', return_f_result=False, if_abs=False):
    x_p, x_s = [], []
    f_result, key_index = [], []
    x = x.strip().split(" ")
    y = y.strip().split(" ")
    for i in range(len(x)):
        g_xi_x_y = G_xi_Two(i, x, y, if_abs=if_abs)
        if (g_xi_x_y > alha):
            x_p.append(x[i])
            key_index.append(i)
            if not x_s or (x_s and x_s[-1] != MASK):
                x_s.append(MASK)
        else:
            x_s.append(x[i])
            if not x_p or (x_p and x_p[-1] != MASK):
                x_p.append(MASK)
        f_result.append((x[i], g_xi_x_y))
    if return_f_result:
        if not key_index:
            key_index = ['-1']
        return x_p, x_s, f_result, " ".join([str(i) for i in key_index])
    return x_p, x_s


def self_PSI(df):
    primarys, seconds, key_indexs = [], [], []
    for i in tqdm(range(df.shape[0])):
        prim, secon, _, key_index = PIS(df.iloc[i]['sen1'].lower(), df.iloc[i]['sen1'].lower(), return_f_result=True)
        primarys.append(" ".join(prim))
        seconds.append(" ".join(secon))
        key_indexs.append(key_index)
    df['self_first'] = primarys
    df['self_second'] = seconds
    df['key_indexs'] = key_indexs
    return df


def get_norm_score(sen1, sen2):
    _, _, score, _ = PIS(sen1.lower(), sen2.lower(), return_f_result=True)
    score = [i[1] for i in score]
    score = [float(i) for i in score]
    score = np.array(score)
    if score.min() == score.max():
        score = score - score.min()
    else:
        score = (score - score.min()) / (score.max() - score.min())
    score = list(score)
    score = [str(i) for i in score]
    return " ".join(score)




fastText2 = load_vectors('crawl-300d-2M.vec')

a = 'What are the best games you can play with just paper '
b = 'Which games can you play on paper in your free time with your friends '
c = 'What can I do to become a professional chess player '
d = 'Should animals be used for testing medicines and drugs '


print(Familiar(a.lower(), b.lower()))

src = 'How are baby elephants called ?'.lower()
tgt = 'What is a name for a baby elephant ?'.lower()

x = PIS(src, tgt, return_f_result=True)
