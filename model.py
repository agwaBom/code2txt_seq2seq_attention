import torch.nn.functional as F
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    
    def forward(self, input, hidden):
        # reshapes x into size (a,b,...)        
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=None):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        # nn.Embedding -  simple lookup table that stores embeddings of a fixed dictionary and size. 
        # This module is often used to store word embeddings and retrieve them using indices. 
        # The input to the module is a list of indices, and the output is the corresponding word embeddings.
        """
        https://wikidocs.net/64779
        # an Embedding module containing 10 tensors of size 3
        embedding = nn.Embedding(10, 3)
        
        # a batch of 2 samples of 4 indices each
        input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
        embedding(input)
        
        tensor([[[-0.0251, -1.6902,  0.7172],
                [-0.6431,  0.0748,  0.6969],
                [ 1.4970,  1.3448, -0.9685],
                [-0.3677, -2.7265, -0.1685]],

                [[ 1.4970,  1.3448, -0.9685],
                [ 0.4362, -0.4004,  0.9400],
                [-0.6431,  0.0748,  0.6969],
                [ 0.9124, -2.3616,  1.1151]]])
        """
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # nn.Linear - Applies a linear transformation to the incoming data: y = xA^T + b
        """
        m = nn.Linear(20, 30)(input, output)
        input = torch.randn(128, 20)()
        output = m(input)
        print(output.size())
        -> torch.Size([128, 30])
        """
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(0), 
                                encoder_outputs.unsqueeze(0))
        
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)