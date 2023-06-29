import torch
from torch.autograd import Variable
from .transformer.model import subsequent_mask
from .transformer.train import Batch
from nltk.translate.bleu_score import corpus_bleu
from dataloader import convert_id_to_token
import rouge
import pandas as pd
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer


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


def greedy_decode(model, src_1, src_2, src_mask, src_mask2, max_len=18, start_symbol=2, mask_id=5):
    # encode: src, src_mask, src2, src_mask2
    memory, memory2 = model.encode(src_1, src_mask, src_2, src_mask2)
    # memory = model.linear1(memory)
    # memory2 = model.linear2(memory2)
    # memory = torch.cat([memory, memory], dim=2)
    # memory2 = torch.cat([memory2, memory2], dim=2)
    batch_size = src_1.shape[0]
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(src_1.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, memory2, src_mask2,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src_1.data)))
        prob = model.generator(out[:, -1])
        prob = torch.softmax(prob, dim=1)
        prob[:, mask_id] = 0
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
    return ys


def test_bleu_rouge(dataloader, model, o_target, idx2token, epoch,
                    max_len=17, start_symbol=2, save_path=None, mask_id=5, task='quora'):

    model.eval()
    # test_src, test_tgt = [], []
    sens = []  # predict tensor, has <unk> token
    for data in dataloader:
        src_1, src_2, tgt = data
        # test_src += src
        # test_tgt += tgt
        batch_1 = Batch(src_1.cuda(), tgt.cuda())
        batch_2 = Batch(src_2.cuda(), tgt.cuda())
        sens_ = greedy_decode(model, batch_1.src, batch_2.src, batch_1.src_mask,
                              batch_2.src_mask, max_len=max_len, start_symbol=start_symbol, mask_id=mask_id)
        sens += sens_.cpu()

    sen_ = []
    for x in sens:
        t = convert_id_to_token(x, idx2token=idx2token, cut=True)
        # t = [i for i in t if i != '<unk>']
        sen_.append(t)

    de = TreebankWordDetokenizer()
    decode_txt = [de.detokenize(i) for i in sen_]
    print("predict:")
    for i in sen_[:10]:
        print(" ".join(i))
    if save_path:
        df = pd.DataFrame()
        # df['predict'] = decode_txt
        df['predict'] = [" ".join(i) for i in sen_]
        df.to_csv(save_path + f'test_tgt{epoch}.csv', index=False)
    if task == 'mscoco':
        measure_dic = measure_bleu_rouge_mscoco(predict=sen_, target=o_target, src=None,
                                                key_index=None, decode_predict=decode_txt)
    else:
        measure_dic = measure_bleu_rouge(predict=sen_, target=o_target, src=None,
                                         key_index=None, decode_predict=decode_txt)
    return measure_dic


def measure_bleu_rouge(predict, target, src, key_index=None, decode_predict=None):
    sen_ = predict

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)
    nltk_token_target = [" ".join(i) for i in target]
    rouge_scores = evaluator.get_scores([" ".join(i) for i in sen_], nltk_token_target)
    # measure_dic = rouge.get_scores([" ".join(i) for i in sen_], nltk_token_target, avg=True)  # (predict, reference)
    measure_dic = {}
    measure_dic['rouge-1'] = rouge_scores['rouge-1']['f']
    measure_dic['rouge-2'] = rouge_scores['rouge-2']['f']
    measure_dic['rouge-l'] = rouge_scores['rouge-l']['f']
    bleu_reference = [[i.split(" ")] for i in nltk_token_target]
    measure_dic["bleu_1"] = corpus_bleu(bleu_reference, sen_, weights=(1, 0, 0, 0))
    measure_dic["bleu_2"] = corpus_bleu(bleu_reference, sen_, weights=(0.5, 0.5, 0, 0))
    measure_dic["bleu_3"] = corpus_bleu(bleu_reference, sen_, weights=(0.33, 0.33, 0.34, 0))
    measure_dic["bleu_4"] = corpus_bleu(bleu_reference, sen_, weights=(0.25, 0.25, 0.25, 0.25))

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

    rouge_score = evaluator.get_scores([" ".join(i) for i in sen_], re_for_rouge)

    measure_dic = {}
    bleu_reference = target
    measure_dic["bleu_1"] = corpus_bleu(bleu_reference, sen_, weights=(1, 0, 0, 0))
    measure_dic["bleu_2"] = corpus_bleu(bleu_reference, sen_, weights=(0.5, 0.5, 0, 0))
    measure_dic["bleu_3"] = corpus_bleu(bleu_reference, sen_, weights=(0.33, 0.33, 0.34, 0))
    measure_dic["bleu_4"] = corpus_bleu(bleu_reference, sen_, weights=(0.25, 0.25, 0.25, 0.25))
    measure_dic['rouge-1'] = rouge_score['rouge-1']['f']
    measure_dic['rouge-2'] = rouge_score['rouge-2']['f']
    measure_dic['rouge-l'] = rouge_score['rouge-l']['f']

    for key, value in measure_dic.items():
        print(f'{key}: {value:.4f} ', end='')
    print(' ')
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
