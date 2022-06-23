import torch.nn as nn
import torch.nn.functional as F





class Decoder(nn.Module):

    def __init__(self, vocab: dict, embed_dim: int, hidden_dim: int, num_layers: int, dropout: int, bidirectional: bool):
        super().__init__()
        vocab_size = len(vocab)
        # TODO: Add padding stuff with padding_idx=vocab["=PAD="]
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.dense = nn.Linear(hidden_dim, vocab_size)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.)
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0.)

    def forward(self, x, state, *args):
        x = self.dropout(self.embedding(x))
        output, state = self.rnn(x, state)
        output = self.dense(output)
        return output, state
