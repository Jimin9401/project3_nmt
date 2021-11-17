import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, DistilBertModel, ElectraModel

# from transformers.configuration_bert import BertConfig
import numpy as np
from .transformer import Decoder, DecoderLayer


def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss


class Seq2Seq(nn.Module):
    def __init__(self, tgt_vocab_size, n_layer=6, pad_idx=0, device="cuda"):
        super(Seq2Seq, self).__init__()
        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.encoder = BertModel.from_pretrained("bert-base-uncased")

        d_model = self.config.hidden_size
        num_head = self.config.num_attention_heads

        d_ff = self.config.intermediate_size
        dropout_rate = self.config.hidden_dropout_prob

        self.device = device
        self.decoder = Decoder(tgt_vocab_size, num_head, d_model, d_ff, n_layer=n_layer, dropout=dropout_rate)
        self.lm_head = nn.Linear(d_model, tgt_vocab_size)
        self.pad_idx = pad_idx

        electra_model = ElectraModel.from_pretrained("monologg/koelectra-base-discriminator")
        target_embedding_matrix = electra_model.embeddings.word_embeddings.weight.data
        self.set_target_embedding(target_embedding_matrix)

    def set_target_embedding(self, trg_embedding_matrix):
        self.decoder.embeddings.word_embedding.weight.data = trg_embedding_matrix
        self.lm_head.weight.data = self.decoder.embeddings.word_embedding.weight.data

    def forward(self, input_ids, attention_mask, tgt_ids):
        src_hidden = self.encoder(input_ids)["last_hidden_state"]

        tgt_size = tgt_ids.size(1)
        tgt_mask = self.subsequent_mask(tgt_size).to(self.device)

        out = self.decoder(tgt_ids, src_hidden, attention_mask, tgt_mask)

        return self.lm_head(out)

    def subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

        return torch.from_numpy(subsequent_mask) == 0

    def generate(self, memory, attention_mask, tgt_ids):
        tgt_size = tgt_ids.size(1)
        tgt_mask = self.subsequent_mask(tgt_size).to(self.device)
        out = self.decoder(tgt_ids, memory, attention_mask, tgt_mask)

        return self.lm_head(out)[:, -1]
