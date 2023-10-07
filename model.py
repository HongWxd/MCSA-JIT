import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class DeepJIT(nn.Module):
    def __init__(self, args):
        super(DeepJIT, self).__init__()
        self.args = args
        V_msg = args.vocab_msg # 100011
        V_code = args.vocab_code # 100011
        Dim = args.embedding_dim #
        Class = args.class_num
        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes

        # Transformer Encoder Layers
        self.encoder_layer = TransformerEncoderLayer(d_model=Dim, nhead=1, dim_feedforward=4)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=1)

        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_file = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks))) for K in Ks])
        self.dropout = nn.Dropout(args.dropout_keep_prob)

        if args.cam:
            self.fc1 = nn.Linear(len(Ks) * Co, args.hidden_units)  # hidden units
        else:
            self.fc1 = nn.Linear(2 * len(Ks) * Co, args.hidden_units)  # hidden units

        self.fc2 = nn.Linear(args.hidden_units, Class)
        self.sigmoid = nn.Sigmoid()

    def forward_msg(self, x, convs):
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return x

    def forward_code(self, x, convs_line, convs_hunks):
        n_batch, n_file = x.shape[0], x.shape[1]
        x = x.reshape(n_batch * n_file, x.shape[2], x.shape[3])

        # Transformer Encoder for Code
        x = x.transpose(0,1) #(seq_len, batch_size, input_dim)
        x = self.transformer_encoder(x)
        x = x.transpose(0,1)

        x = self.forward_msg(x=x, convs=convs_line)
        x = x.reshape(n_batch, n_file, self.args.num_filters * len(self.args.filter_sizes))
        x = self.forward_msg(x=x, convs=convs_hunks)
        return x

    def forward(self, msg, code):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x_msg = self.embed_msg(msg)

        # Transformer Encoder
        x_msg = x_msg.transpose(0,1) #(seq_len, batch_size, input_dim)
        x_msg = self.transformer_encoder(x_msg)
        x_msg = x_msg.transpose(0,1)

        x_msg = self.forward_msg(x=x_msg, convs=self.convs_msg)
        x_code = self.embed_code(code)

        x_code = self.forward_code(x=x_code, convs_line=self.convs_code_line, convs_hunks=self.convs_code_file)

        x = torch.cat((x_msg, x_code), dim=1)
        x = self.dropout(x)
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).squeeze(1)
        return out

