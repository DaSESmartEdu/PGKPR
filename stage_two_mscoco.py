import os
import time
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from Model.SPTransformerFine import PGKPR
from Model.transformer.optimal import get_std_opt, NoamOpt
from Model.transformer.train import Batch
from Model.transformer.lossfunction import LabelSmoothing, LossCompute
from evaluation import test_bleu_rouge, normalize_txt
from dataloader import loadVocab, load_data_tensor_con

EPOCHS = 15
BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
MAX_LEN = 16
LOWERCASE = True
D_MODEL = 512
tqdm_print_step = 1
MID_LOSS = True
SWITCH_INPUT = True
DOUBLE_POSITION = False


if_MID_LOSS = True
if_CON_LOSS = True

# weight
MID_WEIGHT = 1.0  # Reconstruction Loss
CON_WEIGHT = 1.0  # contrastive Loss
S2S_WEIGHT = 1.0  # s2s Loss


task_name = 'mscoco'
log_path = './grade_runs'

vocab_path = 'data/mscoco/quora.8000'

train_path = 'data/mscoco/stage_two_xy/train.csv'
train_path_xx = 'data/mscoco/stage_two_xx/train.csv'

val_path = 'data/mscoco/stage_two_xx/train.csv'
val_path_xy = 'data/mscoco/stage_two_xy/test.csv'

cur_time = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())

model_save_path = 'grade_model_save/' + task_name
test_txt_save_path = 'test_out/' + task_name + '/'



class SPLoss:
    def __init__(self, vocab_size, mid_weight=0.5, con_loss_weight=0.5, con_temperature=0.1, opt=None, if_con=True, if_mid=True):
        self.mid_weight = mid_weight
        self.if_con_loss = if_con
        self.if_mid_loss = if_mid

        self.con_loss_weight = con_loss_weight
        self.con_temperature = con_temperature

        self.vocab_size = vocab_size
        self.seq2seq_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.con_loss_criterion = ContrastiveLoss(temperature=con_temperature)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.opt = opt

    def __call__(self, mid_output, mid_label,  # mid loss
                 sen, sen_label,  # s2s loss
                 src_encode_feature=None, tgt_encode_feature=None, src_tokes=None, tgt_with_pos_tokens=None,  # con_loss
                 train=True):
        if self.opt:
            self.opt.zero_grad()
        # mid_label = mid_index.view(-1) * mid_label.view(-1)
        mid_label = mid_label.view(-1)
        if self.if_mid_loss:
            mid_loss = self.ce_loss(mid_output.view(-1, self.vocab_size), mid_label)
        else:
            mid_loss = 0
        s2s_loss = self.seq2seq_criterion(sen.view(-1, sen.size(-1)),
                                          sen_label.contiguous().view(-1))
        if self.if_con_loss:
            con_loss = self.con_loss_criterion(src_encode_feature, tgt_encode_feature, src_tokes, tgt_with_pos_tokens)
        else:
            con_loss = 0
        total_loss = s2s_loss + self.mid_weight * mid_loss + self.con_loss_weight * con_loss
        if self.opt is not None and train:
            total_loss.backward()
            self.opt.step()

        return mid_loss, s2s_loss.data, con_loss, total_loss


class ContrastiveLoss:
    def __init__(self, temperature=0.1):
        self.temperature = temperature
        self.similarity_fun = nn.CosineSimilarity(dim=-1)

    @staticmethod
    def _sentence_embedding(encoder_out, tokens):
        # encoder_output = encoder_out.transpose(0, 1)
        encoder_output = encoder_out
        mask = (tokens != 0)  # pad: 0
        mask_ = mask.unsqueeze(-1)
        encoder_embedding = (encoder_output * mask_).sum(dim=1) / mask.float().sum(dim=1).unsqueeze(
            -1)  # [batch, hidden_size]
        return encoder_embedding

    @staticmethod
    def item(tensor):
        if hasattr(tensor, "item"):
            return tensor.item()
        if hasattr(tensor, "__getitem__"):
            return tensor[0]
        return tensor

    def __call__(self, encoder_out1, encoder_out2, src_tokens, tgt_tokens):

        encoder_embedding1 = ContrastiveLoss._sentence_embedding(encoder_out1, src_tokens)  # [batch, hidden_size]
        encoder_embedding2 = ContrastiveLoss._sentence_embedding(encoder_out2, tgt_tokens)  # [batch, hidden_size]

        batch_size = encoder_embedding2.shape[0]
        feature_dim = encoder_embedding2.shape[1]
        anchor_feature = encoder_embedding1
        contrast_feature = encoder_embedding2

        anchor_dot_contrast = self.similarity_fun(anchor_feature.expand((batch_size, batch_size, feature_dim)),
                                                  torch.transpose(
                                                      contrast_feature.expand((batch_size, batch_size, feature_dim)),
                                                      0, 1))
        loss = -nn.LogSoftmax(0)(torch.div(anchor_dot_contrast, self.temperature)).diag().sum()
        src_ntokens = ContrastiveLoss.item((src_tokens[:, 1:] != 0).int().sum().data)
        # loss = loss * src_ntokens / src_tokens.size(0)
        loss = loss / src_tokens.size(0)
        return loss



if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)
if not os.path.exists(test_txt_save_path):
    os.mkdir(test_txt_save_path)

writer = SummaryWriter(os.path.join(log_path, task_name))
token2idx, idx2token = loadVocab(vocab_path, lowercase=LOWERCASE)
vocab_size = len(token2idx)
start_id = token2idx.get('<s>')

# data loader
# load_data_tensor return: inputs_pd, mid_label_pd, mid_indexs, labels_pd
train_data = load_data_tensor_con(train_path, vocab_path, MAX_LEN, lowercase=LOWERCASE, switch_pos=SWITCH_INPUT)
train_data_xx = load_data_tensor_con(train_path_xx, vocab_path, MAX_LEN, lowercase=LOWERCASE, switch_pos=SWITCH_INPUT)

val_data = load_data_tensor_con(val_path, vocab_path, MAX_LEN, lowercase=LOWERCASE, switch_pos=SWITCH_INPUT)
val_data_xy = load_data_tensor_con(val_path_xy, vocab_path, MAX_LEN, lowercase=LOWERCASE, switch_pos=SWITCH_INPUT)

val_df = pd.read_csv(val_path)

if LOWERCASE:
    for i in [f'question{i}' for i in range(2, 6)]:
        val_df[i] = val_df[i].apply(lambda x: x.lower())
targets = np.array(val_df[[f'question{i}' for i in range(2, 6)]]).tolist()
val_tgt_string = [normalize_txt(i, token2idx=token2idx, idx2token=idx2token, UNK_TAG='[unk]') for i in targets]
# val_tgt_string = normalize_txt(val_tgt_string, token2idx=token2idx, idx2token=idx2token, UNK_TAG='[unk]')

val_src_string = val_df['sen1'].apply(lambda x: x.lower()).to_list()  # for src_bleu, key_acc test
val_src_string = normalize_txt(val_src_string, token2idx=token2idx, idx2token=idx2token, UNK_TAG='[unk]')
val_src_string = [" ".join(i) for i in val_src_string]
val_indexes = [[int(i) for i in j.strip().split(" ")] for j in val_df.s_index.to_list()]

train_data = TensorDataset(*train_data)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

train_data_xx = TensorDataset(*train_data_xx)
train_sampler_xx = RandomSampler(train_data_xx)
train_dataloader_xx = DataLoader(train_data_xx, sampler=train_sampler_xx, batch_size=BATCH_SIZE)

val_data = TensorDataset(*val_data)
val_dataloader = DataLoader(val_data, batch_size=VAL_BATCH_SIZE)

val_data_xy = TensorDataset(*val_data_xy)
val_dataloader_xy = DataLoader(val_data_xy, batch_size=VAL_BATCH_SIZE)

dataloader = {'train': train_dataloader, 'val': val_dataloader, 'train_xx': train_dataloader_xx}
num_of_batch = len(train_dataloader)

# model
model = PGKPR(d_model=D_MODEL, vocab=vocab_size, mid_loss=MID_LOSS, double_pe=DOUBLE_POSITION,
                          shared_emb=True)

# model_opt = get_std_opt(model.parameters())
model_opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
# model_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98))
model = model.cuda()

# scheduler = torch.optim.lr_scheduler.StepLR(model_opt, step_size=20, gamma=0.1)

loss_compute = SPLoss(vocab_size=vocab_size,
                      mid_weight=MID_WEIGHT, con_loss_weight=CON_WEIGHT,
                      opt=model_opt,
                      if_con=if_CON_LOSS,
                      if_mid=if_MID_LOSS)
loss_compute.ce_loss.cuda()
loss_compute.seq2seq_criterion.cuda()

best_loss = float("inf")
global_step = 0


def run_onece(model, batch):
    _, src_encode_f, mid_out, final_out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask, )
    tgt_encode_f = None
    if if_CON_LOSS:
        _, tgt_encode_f = model.encode(tgt_with_pos, (tgt_with_pos != 0).unsqueeze(-2))
    mid_loss_, s2s_loss_, con_loss_, loss = loss_compute(mid_out, mid_labels,
                                                         final_out, batch.trg_y,
                                                         src_encode_f, tgt_encode_f, batch.src, tgt_with_pos,
                                                         train=('train' in phase))
    return mid_loss_, s2s_loss_, con_loss_, loss


for epoch in range(EPOCHS):
    # scheduler.step()
    print(f'\n EPOCH {epoch}')
    val_t = 0
    for phase in ['train', 'val', 'train_xx', 'val']:
        if 'train' in phase:
            model.train()
        elif phase == 'val':
            val_t += 1
            model.eval()
        total_loss = 0
        total_s2s, total_mid, total_con_loss = 0, 0, 0
        with tqdm(total=len(dataloader[phase])) as pbar:
            for i, data in enumerate(dataloader[phase]):
                src, mid_labels, tgt, tgt_with_pos = data
                src, mid_labels, tgt, tgt_with_pos = src.cuda(), mid_labels.cuda(), tgt.cuda(), tgt_with_pos.cuda()

                batch = Batch(src, tgt)
                if phase == 'val':
                    with torch.no_grad():
                        mid_loss, s2s_loss, con_loss, loss = run_onece(model, batch)
                else:
                    mid_loss, s2s_loss, con_loss, loss = run_onece(model, batch)

                total_loss += loss
                total_s2s += s2s_loss
                total_mid += mid_loss
                total_con_loss += con_loss
                # s2s_loss = s2s_loss / batch.ntokens
                # cur_loss = mid_loss + s2s_loss
                if phase == 'train':
                    global_step += 1
                    writer.add_scalar('loss', loss, global_step)
                    writer.add_scalar('s2s_loss', s2s_loss, global_step)
                    writer.add_scalar('con_loss', con_loss, global_step)
                    writer.add_scalar('mid_loss', mid_loss, global_step)

                if i % tqdm_print_step == 0:
                    pbar.update(tqdm_print_step)
                    pbar.set_description(
                        f'mid: {mid_loss:.4f}  con: {con_loss:.4f}  s2s: {s2s_loss:.4f}  t: {loss:.4f}')

        epoch_loss = total_loss / len(dataloader[phase])
        epoch_s2s = total_s2s / len(dataloader[phase])
        epoch_mid = total_mid / len(dataloader[phase])
        epoch_con = total_con_loss / len(dataloader[phase])
        print(f"{phase} epoch loss: {epoch_loss},  s2s {epoch_s2s}, mid {epoch_mid}, con {epoch_con}")
        if phase == 'val' and (epoch_loss < best_loss or epoch % 10 == 0) and epoch > 5:
            print("save model")
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'{model_save_path}/epoch{epoch}_loss_{epoch_loss}.pkl')
        if phase == 'val':
            # if not os.path.exists(model_save_path):
            #     os.mkdir(model_save_path)
            print("xx:")
            measure_dic = test_bleu_rouge(dataloader[phase], model, o_src=val_src_string,
                                          o_target=val_tgt_string, key_index=val_indexes,
                                          idx2token=idx2token, epoch=epoch, max_len=MAX_LEN,
                                          start_symbol=start_id, save_path=test_txt_save_path + str(val_t) + '_',
                                          switch_input=SWITCH_INPUT, task='mscoco')
            print("xy:")
            measure_dic_2 = test_bleu_rouge(val_dataloader_xy, model, o_src=val_src_string,
                                            o_target=val_tgt_string, key_index=val_indexes,
                                            idx2token=idx2token, epoch=epoch, max_len=MAX_LEN,
                                            start_symbol=start_id, save_path=test_txt_save_path + str(val_t) + '_xy_',
                                            switch_input=SWITCH_INPUT, task='mscoco')
            writer.add_scalars('Bleu_Rouge', measure_dic, epoch)
            for name, value in measure_dic.items():
                writer.add_scalar(name, value, epoch)
