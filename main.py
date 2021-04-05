# It's basically the equivalent of "I guess you're not ready for this, but your kids are gonna love it". 
from __future__ import unicode_literals, print_function, division
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import preprocessing as p
import models.seq2seq_attn as m
import models.bilstm as bilstm
import helper as h
import evaluation as e

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import random
import time

writer = SummaryWriter('runs/experiment3')

teacher_forcing_ratio = 0.5

def validate(input_tensor, target_tensor, encoder, decoder, max_length, criterion):
    with torch.no_grad():
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        loss = 0
        for i in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
            encoder_outputs[i] = encoder_output[0, 0]
        
        decoder_input = torch.tensor([[p.SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        for i in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[i])
            if decoder_input.item() == p.EOS_token:
                break

    return loss.item() / target_length

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, max_length, criterion):
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

def trainIters(encoder, decoder, n_epoch, max_length, learning_rate=0.01):
    start = time.time()
    print_train_loss_total = 0  # Reset every print_every
    print_valid_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [p.tensorsFromPair(train_input_lang, train_output_lang, train_pairList[i]) for i in range(0, len(train_pairList))]
    validation_pairs = [p.tensorsFromPair(valid_input_lang, valid_output_lang, valid_pairList[i]) for i in range(0, len(valid_pairList))]
    test_pairs = [p.tensorsFromPair(test_input_lang, test_output_lang, test_pairList[i]) for i in range(0, len(test_pairList))]
 
    criterion = nn.NLLLoss()

    for i in range(1, n_epoch + 1):
        # one epoch
        print(i, " Epoch...")

        # Training loop
        print("Training loop")
        for j in tqdm(range(1, len(train_pairList) + 1)):
            encoder1.train()
            attn_decoder1.train()
            
            training_pair = training_pairs[j - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            train_loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, max_length, criterion)
            print_train_loss_total += train_loss

            if j % 10 == 0:
                writer.add_scalar('training loss', train_loss / 10, (i-1) * len(train_pairList) + (j-1))

        # Validation loop
        print("Validation loop")
        for j in tqdm(range(1, len(valid_pairList) + 1)):
            validation_pair = validation_pairs[j - 1]
            input_tensor = validation_pair[0]
            target_tensor = validation_pair[1]

            encoder1.eval()
            attn_decoder1.eval()

            val_loss = validate(input_tensor, target_tensor, encoder,
                        decoder, max_length, criterion)
            print_valid_loss_total += val_loss

            if j % 10 == 0:
                writer.add_scalar('validation loss', val_loss / 10, (i-1) * len(valid_pairList) + (j-1))

        # Print train loss Status
        print_train_loss_avg = print_train_loss_total / len(train_pairList)
        print_train_loss_total = 0
        print('Time elapsed : %s \tPercentage: %d%% \tml_loss : %.4f' % (h.timeSince(start, i / n_epoch),
                                            i / n_epoch * 100, print_train_loss_avg))

        # [input & target] pair [0][0]-src [0][1]-tgt
        train_hypothesis_list, train_reference_list = make_hypothesis_reference(encoder, decoder, train_pairList, train_input_lang, max_length)
        valid_hypothesis_list, valid_reference_list = make_hypothesis_reference(encoder, decoder, valid_pairList, valid_input_lang, max_length)

        bleu, rouge_l, _, precision, recall, f1 = e.eval_accuracies(train_hypothesis_list, train_reference_list)
        print("\nTraining Data Evaluation")
        print("\nbleu : %f\trouge_l : %f\tprecision : %f\trecall : %f\tf1 : %f\t" %(bleu, rouge_l, precision, recall, f1))
        print("\n")
        # Random Evaluation

        evaluateRandomly(encoder, decoder, train_pairList, train_input_lang, max_length)
        bleu, rouge_l, _, precision, recall, f1 = e.eval_accuracies(valid_hypothesis_list, valid_reference_list)
        print("\nValidation Data Evaluation")
        print("\nbleu : %f\trouge_l : %f\tprecision : %f\trecall : %f\tf1 : %f\t" %(bleu, rouge_l, precision, recall, f1))
        print("\n")

        # Random Evaluation
        evaluateRandomly(encoder, decoder, valid_pairList, valid_input_lang, max_length)

def make_hypothesis_reference(encoder, decoder, pairList, input_lang, max_length):
    hypothesis_list, reference_list = dict(), dict()
    for i in range(0, len(pairList)):
        output_words, _ = evaluate(encoder, decoder, pairList[i][0], input_lang, max_length=max_length)

        hypothesis_list[i] = [' '.join(output_words)]
        reference_list[i] = [pairList[i][1]]

    return hypothesis_list, reference_list

def evaluate(encoder, decoder, sentence, input_lang, max_length):
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
                decoded_words.append(train_output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:i + 1]

def evaluateRandomly(encoder, decoder, dataList, input_lang ,max_length ,n=2):
    for i in range(n):
        pair = random.choice(dataList)
        print('input:\t', pair[0])
        print('target:\t', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, max_length=max_length)
        output_sentence = ' '.join(output_words)
        print('output:\t', output_sentence)
        print('')

if __name__ == '__main__':
    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data_dir parameter
    train_src_dir = "./data/python/debug/train/code.original_subtoken"
    train_tgt_dir = "./data/python/debug/train/javadoc.original"

    valid_src_dir = "./data/python/debug/valid/code.original_subtoken"
    valid_tgt_dir = "./data/python/debug/valid/javadoc.original"

    test_src_dir = "./data/python/debug/test/code.original_subtoken"
    test_tgt_dir = "./data/python/debug/test/javadoc.original"

    total_src_dir = "./data/python/debug/total/code.original_subtoken"
    total_tgt_dir = "./data/python/debug/total/javadoc.original"

    print("#### Reading Train data\n")
    train_input_lang, train_output_lang, train_pairList, train_max = p.prepareData('code', 'text', train_src_dir, train_tgt_dir, False)
    print("#### Reading Validation data\n")
    valid_input_lang, valid_output_lang, valid_pairList, valid_max = p.prepareData('val_code', 'val_text', valid_src_dir, valid_tgt_dir, False)
    print("#### Reading Test data\n")
    test_input_lang, test_output_lang, test_pairList, test_max = p.prepareData('test_code', 'test_text', test_src_dir, test_tgt_dir, False)
    print("#### total data")
    total_input_lang, total_output_lang, _, max = p.prepareData('a', 'a', total_src_dir, total_tgt_dir, False)


    # hyperparameter
    hidden_size = 256
    n_epoch = 100
    
    # input_lang.n_words = 총 input 데이터에서 나온 단어의 개수
    # ouput_lang.n_words = 총 tgt 데이터에서 나온 단어의 개수
    encoder1 = m.EncoderRNN(total_input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = m.AttnDecoderRNN(hidden_size, total_output_lang.n_words, dropout_p=0.1, max_length=max).to(device)    

    trainIters(encoder1, attn_decoder1, n_epoch, max_length=max)

    # Tensorboard Network Visualizer
    # writer.add_graph(encoder1)
    # writer.add_graph(attn_decoder1)
    writer.close()
