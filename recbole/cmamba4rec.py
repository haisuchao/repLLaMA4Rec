"""
CMamba4Rec — CNN + Mamba Sequential Recommender

Kiến trúc: mỗi layer xử lý sequence qua Conv1d (local pattern) rồi Mamba SSM
(long-range dependency), kết hợp attention pooling để tổng hợp output.

Chạy:
  python run_recbole.py recbole.cmamba4rec.CnnMamba4RecV2 beauty
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from mamba_ssm import Mamba
from recbole.model.abstract_recommender import SequentialRecommender


class CMamba4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(CMamba4Rec, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["n_layers"]
        self.dropout_prob = config["dropout_prob"]

        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        self.cnn_size = config["cnn_size"]
        self.max_seq_length = config["max_seq_length"]

        self.multi_intent = config["multi_intent"]
        if self.multi_intent:
            self.beta = config["beta"]

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                max_seq_length=self.max_seq_length,
                cnn_size=self.cnn_size,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])

        # Attention pooling
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_prob)

        self.temperature = math.sqrt(self.hidden_size)
        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        item_emb = self.item_embedding(item_seq)
        item_emb = self.LayerNorm(self.dropout(item_emb))

        mamba_output = item_emb
        for layer in self.mamba_layers:
            mamba_output = layer(mamba_output)

        # Attention pooling: combine last hidden state with all positions
        mask = item_seq.gt(0).unsqueeze(2).expand_as(mamba_output)
        ht = self.gather_indexes(mamba_output, item_seq_len - 1)
        q1 = self.a_1(mamba_output)
        q2 = self.a_2(ht)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        alpha = self.v_t(mask * torch.sigmoid(q1 + q2_expand))
        c_local = torch.sum(alpha.expand_as(mamba_output) * mamba_output, 1)

        output = self.LayerNorm(self.dropout(c_local))
        return output

    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        gather_index = torch.clamp(gather_index, min=0)
        return output.gather(dim=1, index=gather_index).squeeze(1)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        test_item_emb = self.item_embedding.weight
        scores = torch.matmul(output, test_item_emb.transpose(0, 1)) / self.temperature
        return self.loss_fct(scores, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        score = torch.matmul(output, test_item_emb.transpose(0, 1))
        return score / self.temperature

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(output, test_items_emb.transpose(0, 1))
        return scores / self.temperature


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, max_seq_length, cnn_size, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.CNN = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=cnn_size,
                stride=1,
                padding="same",
            ),
            nn.ReLU(),
        )

        self.ds_gate = DSGate(d_model, cnn_size)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor):
        # CNN → Mamba
        hidden_states = self.CNN(input_tensor.permute(0, 2, 1)).permute(0, 2, 1)
        hidden_states = self.mamba(hidden_states)

        if self.num_layers == 1:
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)

        hidden_states = self.ffn(hidden_states)
        return hidden_states


class DSGate(nn.Module):
    def __init__(self, d_model, cnn_size, dropout=0.2):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=cnn_size, stride=1, padding="same")
        self.linear = nn.Linear(d_model, d_model)
        self.selective_gate_sig = nn.Sequential(nn.Sigmoid(), nn.Linear(d_model, d_model))
        self.selective_gate_si = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model))
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.dense(input_tensor)
        hidden_states = self.conv1d(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
        hidden_states = self.linear(hidden_states)
        hidden_states = self.selective_gate_si(hidden_states) + self.selective_gate_sig(hidden_states)
        return self.dropout(hidden_states)


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_2(self.dropout(self.activation(self.w_1(input_tensor))))
        return self.LayerNorm(self.dropout(hidden_states) + input_tensor)


class FilterLayer(nn.Module):
    """Frequency-domain filtering (optional — không dùng trong config mặc định)."""
    def __init__(self, max_seq_length, hidden_size, dropout_prob):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(1, max_seq_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02
        )
        self.out_dropout = nn.Dropout(dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")
        x = x * torch.view_as_complex(self.complex_weight)
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm="ortho")
        return self.LayerNorm(self.out_dropout(sequence_emb_fft) + input_tensor)


class HighwayNetwork(nn.Module):
    """Highway gating giữa input và hidden (optional)."""
    def __init__(self, d_model):
        super().__init__()
        self.highway_fn = nn.Linear(d_model * 2, d_model)

    def forward(self, input_tensor, hidden_states):
        gate = torch.sigmoid(self.highway_fn(torch.cat([input_tensor, hidden_states], dim=-1)))
        return gate * input_tensor + (1 - gate) * hidden_states


class SelectiveGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.selective_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Dropout(0.3),
        )

    def forward(self, input_tensor, hidden_states):
        return hidden_states * self.selective_gate(input_tensor)


class GLU(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        output = self.w1(input_tensor) * torch.sigmoid(self.w2(input_tensor))
        return self.LayerNorm(self.dropout(output) + input_tensor)
