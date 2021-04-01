# It's basically the equivalent of "I guess you're not ready for this, but your kids are gonna love it". 
from __future__ import unicode_literals, print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import preprocessing as p
import model as m
import helper as h
import evaluation as e

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import random
import time

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for i in range(input_length):
        """
        input_tensor
        tensor([[ 2],        [ 3],        [ 4],        [ 4],        [ 5],        [ 6],        [ 7],        [ 8],        [ 9],        [10],        [ 5],        [ 4],        [ 8],        [11],        [ 4],        [12],        [13],        [14],        [15],        [16],        [17],        [18],        [19],        [ 8],        [ 9],        [20],        [21],        [22],        [ 4],        [23],        [11],        [ 4],        [ 5],        [12],        [ 6],        [ 6],        [24],        [ 4],        [ 4],        [ 8],        [16],        [17],        [18],        [19],        [ 8],        [ 9],        [20],        [21],        [24],        [ 4],        [25],        [26],        [27],        [28],        [24],        [ 4],        [ 1]], device='cuda:0')
        
        encoder_hidden
        tensor([[[ 0.1931, -0.0288, -0.0403, -0.0782, -0.0759, -0.2417,  0.3216,          
                -0.4941, -0.4487, -0.0596,  0.1550, -0.1158,  0.2394,  0.3644,          
                0.3349,  0.3393, -0.0818, -0.0335,  0.1329,  0.1974, -0.0626,           0.0375, -0.3542,  0.1328,  0.3296, -0.1835,  0.2873, -0.0031,           0.1817,  0.2797, -0.1199,  0.3443, -0.1583, -0.3683, -0.3138,          -0.0136, -0.5000,  0.0608, -0.0383,  0.1510,  0.1993, -0.2647,           0.3436, -0.1021, -0.2588,  0.1638, -0.0438,  0.3178, -0.2639,           0.2243,  0.0793,  0.3553, -0.3028,  0.2559,  0.2128, -0.2685,          -0.2793, -0.1913, -0.0073, -0.2530,  0.2138, -0.5338,  0.0322,           0.3420, -0.2666,  0.3883,  0.0535,  0.3029, -0.2116, -0.2974,           0.0872,  0.1116,  0.2034,  0.1717,  0.0933, -0.2494,  0.4260,          -0.3147, -0.2608,  0.0082,  0.0157,  0.0964, -0.2405, -0.0040,          -0.2402,  0.3808,  0.1771, -0.0247,  0.2049,  0.0212, -0.1014,          -0.0458, -0.1931,  0.2626,  0.2163, -0.1463,  0.1133,  0.0399,          -0.2775, -0.1638,  0.2581, -0.0090,  0.0109,  0.3343, -0.4982,           0.0713, -0.2934, -0.1198,  0.3202,  0.1690, -0.2773,  0.1554,          -0.1708, -0.1260,  0.1729,  0.1523,  0.1639,  0.0776,  0.0333,           0.3246,  0.0604,  0.0633, -0.1435,  0.1616,  0.1365,  0.1694,          -0.2313, -0.0587, -0.0597,  0.1756, -0.2054,  0.1090, -0.2551,          -0.0911,  0.2672, -0.0025,  0.2719,  0.0765,  0.0883,  0.0236,           0.2918, -0.2043,  0.1080,  0.3130, -0.1346,  0.0191,  0.3372,           0.4595,  0.1192,  0.1517,  0.1028,  0.1144, -0.4471,  0.0987,           0.2017,  0.0036,  0.1711, -0.2400, -0.2502, -0.3395,  0.3819,          -0.1459,  0.3566, -0.2038,  0.1443,  0.0873, -0.0578, -0.0245,           0.3531,  0.3128,  0.0834, -0.2180,  0.1291, -0.2935,  0.4090,           0.4194, -0.2653, -0.2735, -0.3946,  0.3825, -0.2396, -0.1795,          -0.0225,  0.0021,  0.4348, -0.0416, -0.1758,  0.1181,  0.3384,           0.3017,  0.0969,  0.1371, -0.0163,  0.0869, -0.1147, -0.0856,          -0.2915, -0.3586,  0.1203, -0.0363,  0.2689, -0.0331, -0.2817,          -0.1523,  0.0764,  0.3766, -0.1820,  0.3974,  0.2686, -0.0332,           0.0817,  0.2498, -0.1333,  0.1835,  0.3197, -0.3336,  0.2052,          -0.2035,  0.2009, -0.0993, -0.2765, -0.1228,  0.2220, -0.0018,          -0.4078, -0.1367, -0.1378,  0.1921,  0.0706,  0.2740, -0.1825,           0.0357, -0.2728, -0.1928,  0.2595, -0.1162,  0.1628, -0.1224,           0.4244, -0.0110,  0.3433,  0.4062,  0.3113,  0.2726,  0.3988,           0.0855, -0.1330, -0.0093,  0.2987,  0.2149, -0.2012,  0.1241,           0.1815,  0.2956, -0.0747, -0.1438]]], device='cuda:0',       grad_fn=<CudnnRnnBackward>)
                """
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[p.SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[i])
            decoder_input = target_tensor[i]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[i])
            if decoder_input.item() == p.EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_epoch, n_iters, max_length, print_every=100, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [p.tensorsFromPair(input_lang, output_lang, pairList[i]) for i in range(0, len(pairList))]
    criterion = nn.NLLLoss()

    for i in range(1, n_epoch + 1):
        # one epoch
        print(i, " Epoch...")

        for iter in tqdm(range(1, n_iters + 1)):
            """
            training_pair : 0
            tensor([[     2],        
                    [113078],        
                    [   198],        
                    [    26],        
                    [ 20807],        
                    [  1446],        [  3248],        [   477],        [ 69937],        [    18],        [   198],        [    18],        [ 10403],        [ 11969],        [     9],        [  3248],        [   231],        [    99],        [  3248],        [    18],        [113079],        [113080],        [    26],        [  1890],        [ 20807],        [  1446],        [  3248],        [    18],        [113079],        [113081],        [    26],        [   130],        [ 20807],        [  1446],        [    59],        [    26],        [ 19879],        [   148],        [    49],        [    36],        [  3248],        [    18],        [113079],        [113082],        [    26],        [ 19879],        [ 20807],        [  1446],        [    14],        [   417],        [  1328],        [    16],        [    73],        [    26],        [113083],        [  3248],        [    18],        [113084],        [   417],        [  1328],        [ 20807],        [  1446],        [  3248],        [    18],        [113085],        [ 11969],        [    18],        [    99],        [   255],        [ 20813],        [     9],        [  3248],        [   231],        [ 11969],        [    24],        [  1087],        [  3248],        [     1]], device='cuda:0')

            training_pair : 1
            tensor([[ 910],        
                    [6135],        
                    [   8],        
                    [   1]]EOS, device='cuda:0')
            """
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            """
            # Access dictionary using []
            print(training_pair[0][0].item())
            print(training_pair[0][1].item())
            print(len(training_pair[0]))
            print([input_lang.index2word[training_pair[0][i].item()] for i in range(0, len(training_pair[0]))])
            print([output_lang.index2word[training_pair[1][i].item()] for i in range(0, len(training_pair[1]))])
            """

            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)
            print_loss_total += loss
            """
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('Time elapsed : %s \tTime left : %d \tPercentage: %d%% \tml_loss : %.4f' % (h.timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))
            """

        # Print Status
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print('Time elapsed : %s \tPercentage: %d%% \tml_loss : %.4f' % (h.timeSince(start, i / n_epoch),
                                            i / n_epoch * 100, print_loss_avg))
        # [input & target] pair [0][0]-src [0][1]-tgt
        hypothesis_list, reference_list = make_hypothesis_reference(encoder, decoder, pairList, max_length)
        bleu, rouge_l, meteor, precision, recall, f1 = e.eval_accuracies(hypothesis_list, reference_list)
        print("\nbleu : %f\trouge_l : %f\tmeteor : %f\tprecision : %f\trecall : %f\tf1 : %f\t" %(bleu, rouge_l, meteor, precision, recall, f1))
        print("\n")
        # Random Evaluation
        evaluateRandomly(encoder, decoder, max_length)


def make_hypothesis_reference(encoder, decoder, pairList, max_length):
    hypothesis_list, reference_list = dict(), dict()
    for i in range(0, len(pairList)):
        output_words, _ = evaluate(encoder, decoder, pairList[i][0], max_length=max_length)

        hypothesis_list[i] = [' '.join(output_words)]
        reference_list[i] = [pairList[i][1]]

    return hypothesis_list, reference_list

def evaluate(encoder, decoder, sentence, max_length):
    with torch.no_grad():
        input_tensor = p.tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i],
                                                     encoder_hidden)
            encoder_outputs[i] += encoder_output[0, 0]

        decoder_input = torch.tensor([[p.SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for i in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[i] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == p.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:i + 1]

def evaluateRandomly(encoder, decoder, max_length ,n=3):
    for i in range(n):
        pair = random.choice(pairList)
        print('input:\t', pair[0])
        print('target:\t', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], max_length=max_length)
        output_sentence = ' '.join(output_words)
        print('output:\t', output_sentence)
        print('')

if __name__ == '__main__':
    n_epoch = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_lang, output_lang, pairList, max = p.prepareData('code', 'text', False)
    hidden_size = 256
    # input_lang.n_words = 총 input 데이터에서 나온 단어의 개수
    # ouput_lang.n_words = 총 tgt 데이터에서 나온 단어의 개수
    encoder1 = m.EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = m.AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=max).to(device)
        
    trainIters(encoder1, attn_decoder1, n_epoch, len(pairList), print_every=len(pairList), max_length=max)

    evaluateRandomly(encoder1, attn_decoder1, max_length=max)
