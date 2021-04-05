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
        """
        input
        tensor([3], device='cuda:0')

        hidden (256개)
        tensor([[[ 0.1931, -0.0288, -0.0403, -0.0782, -0.0759, -0.2417,  0.3216,          -0.4941, -0.4487, -0.0596,  0.1550, -0.1158,  0.2394,  0.3644,           0.3349,  0.3393, -0.0818, -0.0335,  0.1329,  0.1974, -0.0626,           0.0375, -0.3542,  0.1328,  0.3296, -0.1835,  0.2873, -0.0031,           0.1817,  0.2797, -0.1199,  0.3443, -0.1583, -0.3683, -0.3138,          -0.0136, -0.5000,  0.0608, -0.0383,  0.1510,  0.1993, -0.2647,           0.3436, -0.1021, -0.2588,  0.1638, -0.0438,  0.3178, -0.2639,           0.2243,  0.0793,  0.3553, -0.3028,  0.2559,  0.2128, -0.2685,          -0.2793, -0.1913, -0.0073, -0.2530,  0.2138, -0.5338,  0.0322,           0.3420, -0.2666,  0.3883,  0.0535,  0.3029, -0.2116, -0.2974,           0.0872,  0.1116,  0.2034,  0.1717,  0.0933, -0.2494,  0.4260,          -0.3147, -0.2608,  0.0082,  0.0157,  0.0964, -0.2405, -0.0040,          -0.2402,  0.3808,  0.1771, -0.0247,  0.2049,  0.0212, -0.1014,          -0.0458, -0.1931,  0.2626,  0.2163, -0.1463,  0.1133,  0.0399,          -0.2775, -0.1638,  0.2581, -0.0090,  0.0109,  0.3343, -0.4982,           0.0713, -0.2934, -0.1198,  0.3202,  0.1690, -0.2773,  0.1554,          -0.1708, -0.1260,  0.1729,  0.1523,  0.1639,  0.0776,  0.0333,           0.3246,  0.0604,  0.0633, -0.1435,  0.1616,  0.1365,  0.1694,          -0.2313, -0.0587, -0.0597,  0.1756, -0.2054,  0.1090, -0.2551,          -0.0911,  0.2672, -0.0025,  0.2719,  0.0765,  0.0883,  0.0236,           0.2918, -0.2043,  0.1080,  0.3130, -0.1346,  0.0191,  0.3372,           0.4595,  0.1192,  0.1517,  0.1028,  0.1144, -0.4471,  0.0987,           0.2017,  0.0036,  0.1711, -0.2400, -0.2502, -0.3395,  0.3819,          -0.1459,  0.3566, -0.2038,  0.1443,  0.0873, -0.0578, -0.0245,           0.3531,  0.3128,  0.0834, -0.2180,  0.1291, -0.2935,  0.4090,           0.4194, -0.2653, -0.2735, -0.3946,  0.3825, -0.2396, -0.1795,          -0.0225,  0.0021,  0.4348, -0.0416, -0.1758,  0.1181,  0.3384,           0.3017,  0.0969,  0.1371, -0.0163,  0.0869, -0.1147, -0.0856,          -0.2915, -0.3586,  0.1203, -0.0363,  0.2689, -0.0331, -0.2817,          -0.1523,  0.0764,  0.3766, -0.1820,  0.3974,  0.2686, -0.0332,           0.0817,  0.2498, -0.1333,  0.1835,  0.3197, -0.3336,  0.2052,          -0.2035,  0.2009, -0.0993, -0.2765, -0.1228,  0.2220, -0.0018,          -0.4078, -0.1367, -0.1378,  0.1921,  0.0706,  0.2740, -0.1825,           0.0357, -0.2728, -0.1928,  0.2595, -0.1162,  0.1628, -0.1224,           0.4244, -0.0110,  0.3433,  0.4062,  0.3113,  0.2726,  0.3988,           0.0855, -0.1330, -0.0093,  0.2987,  0.2149, -0.2012,  0.1241,           0.1815,  0.2956, -0.0747, -0.1438]]], device='cuda:0',       grad_fn=<CudnnRnnBackward>)
        
        embedding (4425(총 나온 단어) * 256)
        
        embedded (256)
        tensor([[[ 0.8885, -0.0961, -0.7002, -0.1999, -0.2634,  0.9593, -0.1256,           0.0499,  0.0189, -0.4807,  2.4351,  0.7059,  1.0537,  0.9353,          -0.4983,  0.0220,  0.0341,  2.4951, -0.5960,  1.2247, -0.0156,          -1.2919,  1.4617, -0.1274,  0.2366,  0.1003, -0.6368, -0.2061,          -0.3106,  0.3114,  0.1426, -0.4900,  0.6764, -0.0051,  1.2040,          -0.8939, -0.7196, -0.2894, -0.3408, -1.4736, -1.8235,  1.2494,          -1.5781,  0.2868, -0.4215,  1.2247,  0.0036,  1.4790,  0.4270,           1.0105, -0.6667, -0.1692,  2.2667,  0.4114, -2.3078, -1.0802,           1.2731, -0.1389, -0.2142, -1.3385, -0.8061,  1.3973, -1.2232,           0.7822,  0.6195, -1.1140, -0.4704,  0.3074, -0.6485,  0.5587,           0.2631,  1.5717,  0.6097,  0.9290,  1.0176, -0.2082,  0.1057,          -1.7231,  0.1151, -0.3630,  0.8289,  0.5722, -0.8692,  0.6555,          -0.2110,  0.3437,  0.0408,  0.6743, -0.6477, -0.5974,  0.5264,          -1.3432, -0.6185, -0.0714,  0.1547,  0.0439,  1.9239, -2.1174,           2.1032, -0.1412, -0.6889,  0.7481,  0.2759, -0.2264,  0.6403,          -0.5904,  0.5931,  0.4658,  0.2066, -2.1063, -0.8445,  0.4511,           0.6273,  0.8610,  0.5864, -0.0321, -0.4809,  1.4605,  0.6252,          -0.1945, -1.3077,  0.0296, -0.1741,  0.6584, -0.2214,  0.5092,          -0.2067,  1.0253,  1.3700,  2.1507, -1.4866,  0.3922, -0.7639,          -0.2858, -0.4689,  1.5681, -0.2830, -0.8501, -0.0921,  1.1747,           1.4002,  0.4907,  0.8655, -1.3227, -1.7787,  0.2620,  1.7175,          -1.9246,  0.1490, -0.4389, -0.6584, -1.7546, -0.6660,  0.0505,           0.3226, -0.5061, -1.2241,  0.0803, -1.8450, -0.7473,  1.1966,          -0.3188,  1.0422,  0.9710,  1.5088, -0.1134,  0.8894,  0.6917,          -2.1927,  1.5304, -0.1040, -1.4913,  0.6837, -0.5278,  0.4795,           0.3407, -1.2775, -1.5434, -0.9406, -0.3505,  2.4522, -0.3173,           0.8418, -0.3744,  0.0695, -0.7981,  0.7904,  1.8918,  0.2914,          -0.5488, -0.2571, -1.2029,  2.3693, -1.1028,  0.2151, -0.4624,           0.6509, -1.2003,  0.8305,  0.8493, -0.2810,  0.2630,  0.7366,           0.3423, -1.9288, -0.9910, -0.0703, -0.9898,  0.0550,  0.4205,          -1.5314, -0.2499, -0.2558,  0.2397,  0.6245, -1.3767, -0.0919,          -0.1982,  0.9909, -1.1371,  2.0547,  2.6003, -0.4960, -0.3714,          -0.2789, -0.3203, -1.1862,  0.3563,  1.8802, -3.4941,  0.5351,           0.9225,  0.5017, -1.6749, -1.0240,  0.1032, -1.3663,  0.1710,          -1.2704, -0.0780,  0.8917, -1.2638, -1.2230, -0.7155, -0.3164,           0.1330, -0.0717, -0.3405, -1.2592, -0.2874,  0.9140, -0.9497,           3.1641,  1.3118,  0.9558, -0.4777]]], device='cuda:0',       grad_fn=<ViewBackward>)
        """
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
