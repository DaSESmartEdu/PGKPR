from Data.switchpos import SwitchPOS
import random
import os
import pandas as pd


def kw_by_posibility(df, large=True, mode='xy', random_seed=666):
    random.seed(random_seed)
    res = []
    dfsens = df.bert_sen1
    if mode == "xy":
        dfscores = df.xy_scores
    else:
        dfscores = df.xx_scores
    for sen, score in zip(dfsens, dfscores):
        score = [float(i) for i in score.strip().split(" ")]
        sen = sen.strip().split(" ")
        assert len(score) == len(sen)
        kw = []
        for w, s in zip(sen, score):
            if large == False:
                s = 1 - s
            if s > random.random():
                kw.append(w)
        res.append(kw)
    return res


def get_stage_two_data(inpath, outpath, dataset='mscoco', mode='xx'):
    base_path = inpath
    out_path = outpath
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    if dataset == 'mscoco':
        data_name = ['val.csv', 'test.csv', 'train_flatten.csv']
    elif dataset == 'quora':
        data_name = ['val.csv', 'test.csv', 'train.csv']
    else:
        raise

    for i in data_name:
        df = pd.read_csv(base_path + i)
        sps = SwitchPOS(base_path + i)
        print(f'{i} : mode {mode}')
        kw = kw_by_posibility(df, large=True, mode=mode)
        kw_index = []
        for x, y in zip(df.sen1.to_list(), kw):
            kw_index.append(sps.word_to_index(x, y))
        sps.set_s_index(kw_index)
        sps.to_csv(out_path + i)
        if 'train' in data_name:
            kw = kw_by_posibility(df, large=True, mode='xx')
            kw_index = []
            for x, y in zip(df.sen1.to_list(), kw):
                kw_index.append(sps.word_to_index(x, y))
            sps.set_s_index(kw_index)
            sps.to_csv(out_path + 'xx_' + i)


if __name__ == '__main__':
    in_path_quora = './data/quora/'
    out_path_quora_xx = './data/quora/stg_two_xx/'
    out_path_quora_xy = './data/quora/stg_two_xy/'

    in_path_mscoco = './data/mscoco/'
    out_path_mscoco_xx = './data/mscoco/stg_two_xx/'
    out_path_mscoco_xy = './data/mscoco/stg_two_xy/'

    get_stage_two_data(in_path_quora, out_path_quora_xx, mode='xx')
    get_stage_two_data(in_path_quora, out_path_quora_xy, mode='xy')

    get_stage_two_data(in_path_mscoco, out_path_mscoco_xx, mode='xx')
    get_stage_two_data(in_path_mscoco, out_path_mscoco_xx, mode='xy')
