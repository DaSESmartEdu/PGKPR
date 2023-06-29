from .transformer.model import *
from .transformer.lossfunction import LabelSmoothing
import torch


class PGKPR(nn.Module):
    def __init__(self, d_model, vocab, nhead=8, num_encoder=6, num_decoder=6,
                 d_ff=2048, dropout=0.1, mid_loss=True, double_pe=False,
                 shared_emb=False):
        super(PGKPR, self).__init__()
        self.is_mid_loss = mid_loss
        self.d_model = d_model
        self.shared_emb = shared_emb
        self.src_embed_pos = nn.Sequential(Embeddings(d_model, vocab),
                                           PositionalEncoding(d_model, dropout, double=double_pe))
        if not shared_emb:
            self.tgt_embed_pos = nn.Sequential(Embeddings(d_model, vocab),
                                               PositionalEncoding(d_model, dropout, double=double_pe))
        self.encoder = Encoder(num_layers=num_encoder, d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout, return_layer=6)

        self.mid_cls = BertClassification(vocab_size=vocab, hidden_size=d_model, layer_norm_eps=1e-12)
        self.decoder = Decoder(num_layers=num_decoder, d_model=d_model, nhead=nhead, d_ff=d_ff, dropout=dropout)
        self.generator = Generator(d_model, vocab)
        self.parameters_init()

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        encode_out, pre_encode_out = self.encode(src, src_mask)
        x = self.decode(encode_out, src_mask, tgt, tgt_mask)
        return encode_out, pre_encode_out, self.mid_cls(encode_out), self.generator(x)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed_pos(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        if self.shared_emb:
            return self.decoder(self.src_embed_pos(tgt), memory, src_mask, tgt_mask)
        return self.decoder(self.tgt_embed_pos(tgt), memory, src_mask, tgt_mask)

    def parameters_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)


# From BERT huggingface
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size=512, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertClassification(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, layer_norm_eps=1e-12):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size=hidden_size, layer_norm_eps=layer_norm_eps)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SPLoss:
    def __init__(self, vocab_size, mid_weight=0.5, opt=None):
        self.mid_weight = mid_weight
        self.vocab_size = vocab_size
        self.seq2seq_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.opt = opt

    def __call__(self, mid_output, mid_label, mid_index, sen, sen_label, norm, train=True):
        if self.opt:
            self.opt.zero_grad()
        # mid_label = mid_index.view(-1) * mid_label.view(-1)
        mid_label = mid_label.view(-1)
        mid_loss = self.ce_loss(mid_output.view(-1, self.vocab_size), mid_label)
        s2s_loss = self.seq2seq_criterion(sen.view(-1, sen.size(-1)),
                                          sen_label.contiguous().view(-1))
        total_loss = s2s_loss + self.mid_weight * mid_loss
        if self.opt is not None and train:
            total_loss.backward()
            self.opt.step()
        return mid_loss, s2s_loss.data, total_loss



