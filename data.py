import os
import torch
import itertools
import numpy as np

from os.path import join

from torch.autograd import Variable

EOS = '<eos>'
UNK = '<unk>'


def tokenize(str_, add_bos=False, add_eos=False):
    words = []
    if add_bos:
        words += [EOS]
    words += str_.split()
    if add_eos:
        words += [EOS]
    return words


class Dictionary(object):
    def __init__(self):
        self.word2idx = {UNK:0}
        self.idx2word = [UNK]

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    @property
    def bos_id(self):
        return self.word2idx[EOS]

    @property
    def eos_id(self):
        return self.word2idx[EOS]

    @property
    def unk_id(self):
        return self.word2idx[UNK]

    def tokenize_file(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = tokenize(line, add_eos=True)
                tokens += len(words)
                for word in words:
                    self.add_word(word)
        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                if not line.strip():
                    continue
                words = tokenize(line, add_eos=True)
                for word in words:
                    ids[token] = self.word2idx[word]
                    token += 1
        return ids

    def words_to_ids(self, words, cuda):
        tt = torch.cuda if cuda else torch
        ids = tt.LongTensor(len(words))
        for i, word in enumerate(words):
            ids[i] = self.word2idx.get(word, self.word2idx[UNK])
        return ids


class Subset(object):
    yields_sentences = False

    def __init__(self, dictionary, path, cuda, rng=None):
        del rng  # Unused in this iterator
        self.tokens = dictionary.tokenize_file(path)
        self.eos_id = dictionary.eos_id
        self.unk_id = dictionary.unk_id

        self.cuda = cuda

        self._last_bsz = None

    def batchify(self, bsz):
        if self._last_bsz == bsz:
            return self._last_batched_data

        data = self.tokens
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        if self.cuda:
            data = data.cuda()

        self._last_bsz = bsz
        self._last_batched_data = data
        return data

    def get_num_batches(self, bsz, bptt):
        data = self.batchify(bsz)
        return int(np.ceil((1.0 * data.size(0) - 1.0) / bptt))

    def iter_epoch(self, bsz, bptt, evaluation=False):
        data = self.batchify(bsz)
        data_len = data.size(0)
        for i in range(0, data_len - 1, bptt):
            seq_len = min(bptt, data_len - 1 - i)
            x = Variable(data[i:i+seq_len], volatile=evaluation)
            y = Variable(data[i+1:i+1+seq_len])
            yield x, y, None


class SubsetBySentence():
    yields_sentences = True

    def __init__(self, dictionary, path, cuda, rng=None):
        raw_data = dictionary.tokenize_file(path).numpy()
        self.eos_id = dictionary.eos_id
        self.unk_id = dictionary.unk_id
        raw_data = np.split(raw_data, np.where(raw_data == self.eos_id)[0] + 1)
        last = raw_data.pop()
        assert last.shape[0] == 0
        # Add 1 to sentence length because we want to hav both an eos
        # at the beginning and at the end.
        max_sentence_len = max(sent.shape[0] for sent in raw_data) + 1
        padded_data = np.zeros(
            (len(raw_data), max_sentence_len+1), dtype='int64') + -1
        for i, sent in enumerate(raw_data):
            padded_data[i, 1:sent.shape[0]+1] = sent
        padded_data[:, 0] = self.eos_id
        tokens = torch.from_numpy(padded_data)

        self.sentences = padded_data
        self.cuda = cuda
        self.rng = np.random.RandomState(rng)

    def get_num_batches(self, bsz, bptt=None):
        del bptt  # unused
        #return int(np.ceil(1.0 * self.sentences.shape[0] / bsz))
        return int(1.0 * self.sentences.shape[0] / bsz) # always return batch of full size (`bsz`)

    def iter_epoch(self, bsz, bptt=None, evaluation=False, ):
        del bptt  # unused
        num_sentences = self.sentences.shape[0]

        sentences = self.sentences

        if not evaluation:
            sentences = np.array(sentences)
            self.rng.shuffle(sentences)

        for i in range(0, num_sentences-bsz, bsz): # always return batch of full size (`bsz`)
            batch = sentences[i:i+bsz]
            seq_lens = (batch != -1).sum(1)

            seq_lens_idx = np.argsort(-seq_lens)
            seq_lens = seq_lens[seq_lens_idx]
            batch = batch[seq_lens_idx, :]

            max_len = seq_lens.max()
            x = np.array(batch[:, :max_len].T)
            x[x == -1] = self.eos_id
            x = torch.from_numpy(x)
            y = torch.from_numpy(batch[:, 1:max_len + 1].T).contiguous()
            seq_lens = torch.from_numpy(seq_lens)
            if self.cuda:
                x = x.cuda()
                y = y.cuda()
                seq_lens = seq_lens.cuda()
            x = Variable(x, volatile=evaluation)
            y = Variable(y, volatile=evaluation)
            seq_lens = Variable(seq_lens, volatile=evaluation)
            yield x, y, seq_lens


class Wrapper(object):
    def __init__(self, dictionary, path, cuda, rng=None):
        self.class0 = SubsetBySentence(
                dictionary, path + '.0', cuda, rng=rng)
        self.class1 = SubsetBySentence(
                dictionary, path + '.1', cuda, rng=rng)

    def get_num_batches(self, bsz):
        return min(self.class0.get_num_batches(bsz),
                self.class1.get_num_batches(bsz))

    def iter_epoch(self, bsz, evaluation=False):
        return itertools.imap(zip, 
                self.class0.iter_epoch(bsz=bsz, evaluation=evaluation),
                self.class1.iter_epoch(bsz=bsz, evaluation=evaluation))


class Corpus(object):
    def __init__(self, path, cuda, rng=None):
        self.dictionary = Dictionary()
        self.train = Wrapper(self.dictionary, path + 'train', cuda, rng=rng) 
        self.valid = Wrapper(self.dictionary, path +'dev', cuda, rng=rng) 
        self.test = Wrapper(self.dictionary, path + 'test', cuda, rng=rng) 
