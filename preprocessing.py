import unicodedata
import re
import torch
from tqdm import tqdm 
from model import device
from io import open

"""
The full process for preparing the data is:

1. Read text file and split into lines, split lines into pairs
2. Normalize text, filter by length and content
3. Make word lists from sentences in pairs
"""
SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # SOS and EOS를 고려해서 2부터 시작.

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else: 
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)


def unicodeToAscii(unicode):
    return ''.join(c for c in unicodedata.normalize('NFD', unicode) if unicodedata.category(c) != 'Mn')

# Lowercase, trim, and remove non-letter characters
def normalizeString(str):
    str = unicodeToAscii(str.lower().strip())
    # Support for regular expressions
    str = re.sub(r"([.!?])", r" \1", str)
    str = re.sub(r"[^a-zA-Z.!?]+", r" ", str)
    return str

def readLangs(lang1, lang2, src_dir, tgt_dir, reverse=False):
    print("Reading Source Code...")
    source = open(src_dir).read().strip().split('\n')
    print("Reading Target...")
    target = open(tgt_dir).read().strip().split('\n')
    print("Pairing source and target")
    pairs = [[i, j]for i, j in zip(source, target)]

    max = 0
    for i in range(0, len(pairs)):
        if len(pairs[i][0].split(' ')) > max:
            max = len(pairs[i][0].split(' '))
    max = max + 1

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        
    return input_lang, output_lang, pairs, max


def prepareData(lang1, lang2, src_dir, tgt_dir,reverse=False):
    input_lang, output_lang, pairList, max = readLangs(lang1, lang2, src_dir, tgt_dir, reverse)
    print("Read %s sentence pairs\n" % len(pairList))
    print("Counting words...")
    for pair in tqdm(pairList):
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words: ")
    print(input_lang.name, input_lang.n_words, output_lang.name, output_lang.n_words, "\n")
    print("max_sentence : ", max)
    return input_lang, output_lang, pairList, max


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
