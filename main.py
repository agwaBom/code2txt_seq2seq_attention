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
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [p.tensorsFromPair(input_lang, output_lang, pairList[i]) for i in range(0, len(pairList))]
    criterion = nn.NLLLoss()

    for i in range(1, n_epoch + 1):
        # one epoch
        print(i, " Epoch...")

        for iter in range(1, n_iters + 1):
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
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (h.timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        # [input & target] pair [0][0]-src [0][1]-tgt
        hypothesis_list, reference_list = make_hypothesis_reference(encoder, decoder, pairList, max_length)
        bleu, rouge_l, meteor, precision, recall, f1 = e.eval_accuracies(hypothesis_list, reference_list)
        print("\nbleu : %f\trouge_l : %f\tmeteor : %f\tprecision : %f\trecall : %f\tf1 : %f\t" %(bleu, rouge_l, meteor, precision, recall, f1))
        print("\n")
        # Random Evaluation
        evaluateRandomly(encoder, decoder, max_length)

    h.showPlot(plot_losses)

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

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


if __name__ == '__main__':
    n_epoch = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_lang, output_lang, pairList, max = p.prepareData('code', 'text', False)
    hidden_size = 256
    encoder1 = m.EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = m.AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=max).to(device)
        
    trainIters(encoder1, attn_decoder1, n_epoch, len(pairList), print_every=100, max_length=max)

    evaluateRandomly(encoder1, attn_decoder1, max_length=max)
    """
    output_words, attentions = evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
    plt.matshow(attentions.numpy())
    """
    evaluateAndShowAttention("def expand probes probes defaults expected probes {/}for probe name probe test in six iteritems probes if probe name not in expected probes keys expected probes[probe name] {}probe defaults probe test pop 'defaults' {/} for test name test details in six iteritems probe test test defaults test details pop 'defaults' {/} expected test details deepcopy defaults expected test details update probe defaults expected test details update test defaults expected test details update test details if test name not in expected probes[probe name] keys expected probes[probe name][test name] expected test detailsreturn expected probes")
    evaluateAndShowAttention("@pytest fixture scope u'session' def celery config return {/}")
