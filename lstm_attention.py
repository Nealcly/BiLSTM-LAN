import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from attention import multihead_attention


class LSTM_attention(nn.Module):
    ''' Compose with two layers '''

    def __init__(self,lstm_hidden,bilstm_flag,data):
        super(LSTM_attention, self).__init__()

        self.lstm = nn.LSTM(lstm_hidden * 4, lstm_hidden, num_layers=1, batch_first=True, bidirectional=bilstm_flag)
        #self.slf_attn = multihead_attention(data.HP_hidden_dim,num_heads = data.num_attention_head, dropout_rate=data.HP_dropout)
        self.slf_attn = multihead_attention(data.HP_hidden_dim, num_heads=data.num_attention_head,dropout_rate=data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_dropout)
        #gpu
        self.lstm =self.lstm.cuda()
        self.slf_attn = self.slf_attn.cuda()


    def forward(self,lstm_out,label_embs,word_seq_lengths,hidden):

        lstm_out = pack_padded_sequence(input=lstm_out, lengths=word_seq_lengths.cpu().numpy(), batch_first=True)
        lstm_out, hidden = self.lstm(lstm_out, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.droplstm(lstm_out.transpose(1, 0))
        # label_embs (18 * 10 * 200)
        attention_label = self.slf_attn(lstm_out, label_embs, label_embs)
        # 10x52x200
        lstm_out = torch.cat([lstm_out, attention_label], -1)
        return lstm_out
