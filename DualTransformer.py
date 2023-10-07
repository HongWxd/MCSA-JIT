import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        M = query.shape[1]
        values_array = np.shape(np.array(values.detach().numpy()))
        tensor_length = len(values_array)
        # print(values.shape)

        # split embedding into self.heads pieces
        if tensor_length == 3:
            value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
            # print("reshape之前：", values.shape)
            values = values.reshape(N, value_len, self.heads, self.head_dim)
            keys = keys.reshape(N, key_len, self.heads, self.head_dim)
            queries = query.reshape(N, query_len, self.heads, self.head_dim)
            # print("reshape之后：", values.shape)
        elif tensor_length == 4:
            value_len, key_len, query_len = values.shape[2], keys.shape[2], query.shape[2]
            # print("reshape之前：", values.shape)
            values = values.reshape(N, M, value_len, self.heads, self.head_dim)
            keys = keys.reshape(N, M, key_len, self.heads, self.head_dim)
            queries = query.reshape(N, M, query_len, self.heads, self.head_dim)
            # print("reshape之后：", values.shape)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        if tensor_length == 3:
            energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
            # print("energy:", energy.shape)
            # queries shape: (N, query_len, heads, heads_dim)
            # keys shape : (N, key_len, heads, heads_dim)
            # energy shape: (N, heads, query_len, key_len)
        elif tensor_length == 4:
            energy = torch.einsum("nmqhd,nmkhd->nmhqk", queries, keys)
            # queries shape: (N, M, query_len, heads, heads_dim)
            # keys shape : (N, M, key_len, heads, heads_dim)
            # energy shape: (N, M, heads, query_len, key_len)


        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        if tensor_length == 3:
            out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
            # attention shape: (N, heads, query_len, key_len)
            # values shape: (N, value_len, heads, heads_dim)
            # (N, query_len, heads, head_dim)
        elif tensor_length == 4:
            out = torch.einsum("nmhql, nmlhd->nmqhd", [attention, values]).reshape(N, M, query_len, self.heads * self.head_dim)
            # attention shape: (N, M, heads, query_len, key_len)
            # values shape: (N, M, value_len, heads, heads_dim)
            # (N, M, query_len, heads, head_dim)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # self.word_embedding = src_vocab_size
        self.position_embedding = nn.Embedding(max_length, embed_size)# max_length=100

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_array = np.shape(np.array(x))
        # print(x_array)
        tensor_length = len(x_array)# 求出tensor的维度个数, tensor[.. , ..]就是2,tensor[.. , .. , ..]就是3
        if tensor_length == 2:
            # print(x.shape)
            N, seq_length = x.shape
            positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
            # print(positions.shape)
            out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
            # out = self.dropout(x + positions)
        elif tensor_length == 3:# 传入code的时候是进入这个分支
            # print(x.shape)
            N, seq_length, M = x.shape
            positions = torch.arange(0, M).expand(N, seq_length, M).to(self.device)#
            # print(positions)
            out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
            # print(x + positions)
            # out = self.dropout(x + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DualTransformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            src_pad_idx,
            embed_size=16,# 改这里embedding_dim
            num_layers=1,
            forward_expansion=4,
            heads=1,
            dropout=0,
            device="cuda",
            max_length=512
    ):
        super(DualTransformer, self).__init__()
        self.encoder1 = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = (src != self.src_pad_idx)
        # (N, 1, 1, src_len)


    def forward(self, src1):
        src_mask1 = self.make_src_mask(src1)

        enc_src1 = self.encoder1(src1, src_mask1)
        return enc_src1


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    src_pad1_idx = 0
    src_pad2_idx = 0
    src_vocab1_size = 10
    src_vocab2_size = 10

    model = DualTransformer(src_vocab1_size, src_pad1_idx, device=device).to(device)

    print(model)
    # print(x.max().item())
    print(x)
    out = model(x)
    # print(out)
