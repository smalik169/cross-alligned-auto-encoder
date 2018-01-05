import argparse
import codecs
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

from nltk import word_tokenize
import numpy as np

#parser = argparse.ArgumentParser(description='Language style transfer Beam search')
#parser.add_argument('--seed', type=int, default=1111,
#                    help='random seed')
#parser.add_argument('--cuda', action='store_true',
#                    help='use CUDA')
#parser.add_argument('--data', type=str, default='/pio/data/data/mikolov_simple_examples/data/ptb.',
#                    help='location of the data corpus')
#parser.add_argument('--beam-size', type=int, default=10,
#                    help='beam size for beam search')
#parser.add_argument('--n-best', type=int, default=10,
#                    help='N best paths to be returned')
#parser.add_argument('--seed-text', type=str, default='',
#                    help='text passed to the initializer')
#parser.add_argument('--model', type=str,  default='./',
#                    help='path to model dir containing model.pt and model.info')


#args = parser.parse_args()
#if args.seed_text_generator:
#    args.seed_text = eval(args.seed_text_generator)
#
## Set the random seed manually for reproducibility.
#torch.manual_seed(args.seed)
#if torch.cuda.is_available():
#    if not args.cuda:
#        print("WARNING: You have a CUDA device, so you should probably "
#              "run with --cuda")
#    else:
#        torch.cuda.manual_seed(args.seed)


def _set_word_prob(probs, idx, probable=True):
    if probable:
        probs.fill_(0.0)
        probs[:,idx] = 1.0
    else:
        probs[:,idx] = 0.0
        probs.div_(probs.sum(1, keepdim=True)) # Re-normalize
    return probs


# Based on OpenNMT-py (MIT license)
# https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
class Beam(object):
    def __init__(self, size, dictionary, rnn_input, n_best=1, constraints=[],
                 cuda=False):
        self.size = size
        self.dictionary = dictionary
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []

        # The backpointers at each time-step.
        # Indexes of beams in {1,..,size}
        self.prevKs = []

        # Has EOS topped the beam yet.
        self._eos = self.dictionary.bos_id
        self.eosTop = False

        # The outputs at each time-step.
        # Indexes of words in {1,...,vocab_size}
        self.nextYs = [rnn_input.data.squeeze()]

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best
        self.constraints = constraints

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        return self.nextYs[-1]

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk): #, attnOut):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        pos = len(self.prevKs)
        for constraint in self.constraints:
            wordLk = constraint.apply(wordLk, pos, self.dictionary)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.allScores.append(self.scores)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))
        # self.attn.append(attnOut.index_select(0, prevK))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            # self.allScores.append(self.scores)
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.n_best

    def sortFinished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def getHyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp = []
        for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
            hyp.append(self.nextYs[j+1][k])
            k = self.prevKs[j][k]
        return hyp[::-1]


class TransferStyle(object):
    """Samples from the model."""
    def __init__(self, dictionary, model, beam_size=10, n_best=10, cuda=False):
        self.dictionary = dictionary
        self.model = model
        self.beam_size = beam_size
        self.n_best = n_best
        self.tt = torch.cuda if cuda else torch

    def __call__(self, data, in_style, out_style, seq_lens=None):
        in_style = in_style if isinstance(in_style, list) else [in_style]
        in_style = self.model.style_encoder(
                Variable(self.tt.LongTensor(in_style)))

        out_style = out_style if isinstance(out_style, list) else [out_style]
        out_style = self.model.style_encoder(
                Variable(self.tt.LongTensor(out_style)))

        if not isinstance(data, Variable):
            data = data if isinstance(data, list) else [data]
            data, seq_lens = self._preprocess(data)

        assert(seq_lens is not None)

        return self._transfer(data, in_style, out_style, seq_lens)

    def _preprocess(self, sentences):
        data = []
        unk_id = self.dictionary.unk_id
        for sent in sentences:
            tokens = word_tokenize(sent)
            data.append(np.array([
                self.dictionary.word2idx[token]
                if token in self.dictionary.word2idx
                else unk_id
                for token in tokens]))

        seq_lens = np.array([sent.shape[0]+2 for sent in data])
        seq_lens_idx = np.argsort(-seq_lens)
        seq_lens = seq_lens[seq_lens_idx]

        padded_data = np.zeros(
            (len(data), seq_lens[0]), dtype='int64')
        padded_data = padded_data + self.dictionary.eos_id
        for i, sent in enumerate(data):
            padded_data[i, 1:sent.shape[0]+1] = sent
        padded_data[:, 0] = self.dictionary.bos_id
        padded_data = padded_data[seq_lens_idx]

        batch = self.tt.from_numpy(padded_data.T).contiguous()
        batch = Variable(batch)
        seq_lens = Variable(self.tt.from_numpy(seq_lens))
        return batch, seq_lens


    def _transfer(self, data, in_style, out_style, seq_lens, max_sent_len=80):
        encoder_init = self.model.encoder.split_initial_hidden(
                self.model.encoder_init_projection(in_style))

#        emb = self.model.encoder.embeddings(data)
#        encoder_output, _ = self.model.encoder.rnn(emb, encoder_init)
#        content = encoder_output[-1]
        content = self.model.encoder(encoder_init, data, seq_lens)

        generator_init = self.model.generator_init_projection(
                torch.cat([out_style, content], 1))
        generator_init = self.model.generator.split_initial_hidden(
                generator_init)
        hidden = generator_init

        batch_size = data.size(1)
        bos = data.data.new([[self.dictionary.bos_id]*self.beam_size])
        beams = [Beam(self.beam_size, self.dictionary,
                      Variable(bos.clone(), volatile=True),
                      n_best=self.n_best,
                      cuda=data.is_cuda)
                 for _ in range(batch_size)]

        for i in range(max_sent_len):

            if all((b.done() for b in beams)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            input = Variable(torch.stack([b.getCurrentState() for b in beams])
                             .t().contiguous().view(1, -1), volatile=True)

            # Run one step.
            output, _, hidden = self.model.generator(hidden, input)
            output = output.squeeze(0)
            # output: beam x rnn_size

            # Compute a vector of batch*beam word scores.
            out = self.model.generator.softmax(
                    output.div(self.model.generator.gamma)).data
            out = out.view(self.beam_size, batch_size, -1)

            # Advance each beam
            for idx, b in enumerate(beams):
                b.advance(out[:, idx])

                # LSTM passes hiddens as tuples, others do not
                def iter_hidden(hidden):
                    hidden = (hidden,) if not type(hidden) is tuple else hidden
                    for h in hidden:
                        yield h

                for h in iter_hidden(hidden):
                    # Each h is of size (num_layers, beam_size * batch_size, nhid)
                    a, br, d = h.size()
                    positions = b.getCurrentOrigin()
                    sentStates = h.view(a, self.beam_size, br // self.beam_size, d)[:, :, idx]
                    sentStates.data.copy_(
                        sentStates.data.index_select(1, positions))

        ret = []
        for b in beams:
            scores, ks = b.sortFinished(minimum=self.n_best)
            ret.append([])
            for i, (times, k) in enumerate(ks[:self.n_best]):
                hyp = b.getHyp(times, k)
                ret[-1].append([self.dictionary.idx2word[i] for i in hyp])
        return ret


def main():
    dictionary = data.Corpus(
        args.data, cuda=args.cuda, yield_sentences=True, rng=None).dictionary

    if False:
        # Old way of loading a model
        with open(args.model, 'rb') as f:
            mdl = torch.load(f)
        print(mdl)
    else:
        mdl = model.load(args.model)
    mdl.softmax = nn.Softmax()
    mdl = mdl.cuda() if args.cuda else mdl.cpu()
    mdl.eval()
    sampler = Sampler(dictionary, mdl)

    seed_texts = []
    if args.seed_text != '':
        seed_texts += [args.seed_text]
    if args.seed_file:
        with codecs.open(args.seed_file, 'r', 'utf-8') as f:
            seed_texts += [line.strip() for line in f]
    if seed_texts == []:
        seed_texts += ['']

    for seed_text in seed_texts:
        if args.print_seed_text:
            print(seed_text)
        if not args.seed_without_eos:
            seed_text = '<eos> ' + seed_text + ' <eos>'

        constraints = eval(args.constraint_list)
        tokenizer_fn = lambda s: dictionary.words_to_ids(
            data.tokenize(s, add_bos=False, add_eos=False), cuda=args.cuda)
        for c in constraints:
            if type(c) is SeedTextDictConstraint:
                c.set_seed_text(seed_text, tokenizer_fn, dictionary)

        if args.num_words:
            # Generate N words
            out_file = sys.stdout
            out_string = sampler.string(seed_text, args.prefix_text, 
                                        args.num_words, constraints=constraints)
            for i, word in enumerate(out_string):
                out_file.write(word + ('\n' if i % 20 == 19 else ' '))
            if i % 20 != 19:
                print('')
        else:
            # Beam search on sentences
            for batch in sampler.sentences(seed_text, args.prefix_text,
                                           constraints=constraints):
                for sent_tokens in batch:
                    #print( "len = %d, %s" % (len(sent_tokens), ' '.join(sent_tokens)) )
                    print(' '.join(sent_tokens)) 


if __name__ == '__main__':
    main()
