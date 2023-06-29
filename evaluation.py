import torch
from torch.autograd import Variable
from Model.transformer.model import subsequent_mask
from Model.transformer.train import Batch
# from Model.transformer.eval import greedy_decode
from nltk.translate.bleu_score import corpus_bleu
from dataloader import convert_id_to_token
from rouge import Rouge
import rouge
import pandas as pd
import os
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer


def greedy_decode(model, src, src_mask, max_len=25, start_symbol=2):
    memory, _ = model.encode(src, src_mask)
    batch_size = src.shape[0]
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        # next_word = next_word.data
        # next_pos = next_pos.data
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
    return ys


def test_bleu_rouge(dataloader, model, o_src, o_target, key_index, idx2token, epoch,
                    max_len=17, start_symbol=2, save_path=None, switch_input=True, oov_mark=False, task='quro'):

    model.eval()
    test_src, test_tgt = [], []
    sens = []  # predict tensor, has <unk> token
    for data in dataloader:
        if switch_input:
            src, _, _, tgt = data
        else:
            src, tgt = data
        test_src += src
        test_tgt += tgt
        batch = Batch(src.cuda(), tgt.cuda())
        sens_ = greedy_decode(model, batch.src, batch.src_mask, max_len=max_len, start_symbol=start_symbol)
        sens += sens_.cpu()

    sen_ = []  # predict string, has <unk> token
    for x in sens:
        t = convert_id_to_token(x, idx2token=idx2token, cut=True)
        # t = [i for i in t if i != '<unk>']
        sen_.append(t)

    de = TreebankWordDetokenizer()
    decode_txt = [de.detokenize(i) for i in sen_]
    if save_path:
        df = pd.DataFrame()
        # df['predict'] = decode_txt
        df['predict'] = [" ".join(i) for i in sen_]
        df.to_csv(save_path + f'test_tgt{epoch}.csv', index=False)
    if task == 'mscoco':
        measure_dic = measure_bleu_rouge_mscoco(predict=sen_, target=o_target, src=o_src,
                                                key_index=key_index, decode_predict=decode_txt)
    else:
        measure_dic = measure_bleu_rouge(predict=sen_, target=o_target, src=o_src,
                                         key_index=key_index, decode_predict=decode_txt)
    return measure_dic


def keywords_acc(predict, src, indexes):
    """
    src: string
    predict: list of words
    """
    # src = src.strip().split(" ")
    predict = set(predict)
    keywords = set([src[i] for i in indexes if i < len(src)])
    same = predict & keywords
    if len(keywords) == 0:
        return 1
    return len(same) / len(keywords)


def keywords_acc_avg(predicts, srcs, indexes):
    assert len(predicts) == len(indexes) == len(srcs)
    len_, result = len(predicts), 0
    for p, s, i in zip(predicts, srcs, indexes):
        result += keywords_acc(p, s, i)
    return result / len_


def test_bleu_rouge_for_transformer(dataloader, model, o_src, o_target, idx2token,
                                    epoch, max_len=15, start_symbol=2, save_path=None, task='quora'):
    model.eval()
    test_src, test_tgt = [], []
    sens = []  # predict tensor, has <unk> token
    for data in dataloader:
        src, tgt = data
        test_src += src
        test_tgt += tgt
        batch = Batch(src.cuda(), tgt.cuda())
        sens_ = greedy_decode(model, batch.src, batch.src_mask, max_len=max_len, start_symbol=start_symbol)
        sens += sens_.cpu()

    sen_ = []  # predict string, has <unk> token
    for x in sens:
        t = convert_id_to_token(x, idx2token=idx2token, cut=True)
        # t = [i for i in t if i != '<unk>']
        sen_.append(t)

    de = TreebankWordDetokenizer()
    decode_txt = [de.detokenize(i) for i in sen_]
    if save_path:
        temp = [" ".join(i) for i in sen_]
        df = pd.DataFrame()
        df['predict'] = temp
        df.to_csv(save_path + f'test_tgt{epoch}.csv', index=False)

    if task == 'mscoco':
        measure_dic = measure_bleu_rouge_mscoco(predict=sen_, target=o_target, src=o_src,
                                                key_index=None, decode_predict=decode_txt)
    else:
        measure_dic = measure_bleu_rouge(predict=sen_, target=o_target, src=o_src,
                                         key_index=None, decode_predict=decode_txt)

    return measure_dic


def test_bleu_rouge_contrast(dataloader, model, o_src, o_target, key_index, idx2token, epoch,
                             max_len=17, start_symbol=2, save_path=None, switch_input=True, oov_mark=False):
    model.eval()
    test_src, test_tgt = [], []
    sens = []  # predict tensor, has <unk> token
    tgts = []
    for data in dataloader:
        src, tgt, _, _ = data
        test_src += src
        test_tgt += tgt
        batch = Batch(src.cuda(), tgt.cuda())
        sens_ = greedy_decode(model, batch.src, batch.src_mask, max_len=max_len, start_symbol=start_symbol)
        sens += sens_.cpu()
        tgts += tgt.cpu()

    sen_, tgts_ = [], []  # predict string, has <unk> token
    for x in sens:
        t = convert_id_to_token(x, idx2token=idx2token, cut=True)
        # t = [i for i in t if i != '<unk>']
        sen_.append(t)
    for x in tgts:
        t = convert_id_to_token(x, idx2token=idx2token, cut=True)
        # t = [i for i in t if i != '<unk>']
        tgts_.append(t)
    if oov_mark:
        oov_marked_src, oov_marked_tgt = [], []

    # test_tgt_ = [[(convert_id_to_token(i, idx2token=idx2token, cut=True))] for i in test_tgt]
    # reference = test_tgt_
    de = TreebankWordDetokenizer()
    decode_txt = [de.detokenize(i) for i in sen_]

    if save_path:
        temp = [" ".join(i) for i in sen_]
        df = pd.DataFrame()
        df['predict'] = temp
        df.to_csv(save_path + f'test_tgt{epoch}.csv', index=False)

    measure_dic = measure_bleu_rouge(predict=sen_, target=tgts_, src=o_src,
                                     key_index=key_index, decode_predict=decode_txt)
    return measure_dic


def normalize_txt(txt, token2idx, idx2token, UNK_TAG, max_len=16):
    def sen_token2id(s):
        return [token2idx.get(i, token2idx[UNK_TAG]) for i in s]

    def sen_id2token(s):
        return [idx2token[i] for i in s]

    txt = [i.strip().split(" ") for i in txt]
    txt = [sen_token2id(i) for i in txt]
    txt = [sen_id2token(i) for i in txt]
    txt = [i[:max_len] for i in txt]
    return txt


def measure_bleu_rouge(predict, target, src, key_index=None, decode_predict=None):
    sen_ = predict

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)
    nltk_token_target = [" ".join(i) for i in target]
    rouge_scores = evaluator.get_scores([" ".join(i) for i in sen_], nltk_token_target)
    # measure_dic = rouge.get_scores([" ".join(i) for i in sen_], nltk_token_target, avg=True)  # (predict, reference)
    measure_dic = {}
    bleu_reference = [[i.split(" ")] for i in nltk_token_target]
    measure_dic["bleu_1"] = corpus_bleu(bleu_reference, sen_, weights=(1, 0, 0, 0))
    measure_dic["bleu_2"] = corpus_bleu(bleu_reference, sen_, weights=(0.5, 0.5, 0, 0))
    measure_dic["bleu_3"] = corpus_bleu(bleu_reference, sen_, weights=(0.33, 0.33, 0.34, 0))
    measure_dic["bleu_4"] = corpus_bleu(bleu_reference, sen_, weights=(0.25, 0.25, 0.25, 0.25))
    measure_dic['rouge-1'] = rouge_scores['rouge-1']['f']
    measure_dic['rouge-2'] = rouge_scores['rouge-2']['f']
    measure_dic['rouge-l'] = rouge_scores['rouge-l']['f']


    # measure_dic['bleu'] = corpus_bleu(bleu_reference, sen_)  # (reference, predict)
    measure_dic['src_bleu'] = corpus_bleu([[i.split(" ")] for i in src], sen_)
    # if key_index:
    #     measure_dic['key_acc'] = keywords_acc_avg(sen_, src, key_index)  # (predict, src, key_index)
    for key, value in measure_dic.items():
        print(f'{key}: {value:.4f} ', end='')
    print(' ')
    return measure_dic


def measure_bleu_rouge_mscoco(predict, target, src, key_index=None, decode_predict=None):
    sen_ = predict
    re_for_rouge = []
    for i in target:
        rouge_ = []
        for j in i:
            rouge_.append(" ".join(j))
        re_for_rouge.append(rouge_)
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)
    # prouge = PyRouge(rouge_n=(1, 2), rouge_l=True, rouge_s=True)
    # rouge_score = prouge.evaluate_tokenized(sen_, target)  # hypotheses, references

    rouge_score = evaluator.get_scores([" ".join(i) for i in sen_], re_for_rouge)
    # measure_dic = rouge.get_scores([" ".join(i) for i in sen_], re_for_rouge, avg=True)  # (predict, reference)
    measure_dic = {}
    bleu_reference = target
    measure_dic["bleu_1"] = corpus_bleu(bleu_reference, sen_, weights=(1, 0, 0, 0))
    measure_dic["bleu_2"] = corpus_bleu(bleu_reference, sen_, weights=(0.5, 0.5, 0, 0))
    measure_dic["bleu_3"] = corpus_bleu(bleu_reference, sen_, weights=(0.33, 0.33, 0.34, 0))
    measure_dic["bleu_4"] = corpus_bleu(bleu_reference, sen_, weights=(0.25, 0.25, 0.25, 0.25))
    measure_dic['rouge-1'] = rouge_score['rouge-1']['f']
    measure_dic['rouge-2'] = rouge_score['rouge-2']['f']
    measure_dic['rouge-l'] = rouge_score['rouge-l']['f']

    # measure_dic['bleu'] = corpus_bleu(bleu_reference, sen_)  # (reference, predict)
    measure_dic['src_bleu'] = corpus_bleu([[i.split(" ")] for i in src], sen_)
    # if key_index:
    #     measure_dic['key_acc'] = keywords_acc_avg(sen_, src, key_index)  # (predict, src, key_index)
    for key, value in measure_dic.items():
        print(f'{key}: {value:.4f} ', end='')
    print(' ')
    return measure_dic
