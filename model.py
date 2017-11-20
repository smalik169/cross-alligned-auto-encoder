import torch
import torch.nn as nn
from torch.autograd import Variable


def get_rnn(rnn_type, ninputs, nhid, nlayers, dropout=0, bidirectional=False):
    if rnn_type in ['LSTM', 'GRU']:
        return getattr(nn, rnn_type)(ninputs, nhid, nlayers, dropout=dropout,
                                     bidirectional=bidirectional)
    else:
        try:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
        except KeyError:
            raise ValueError("An invalid option for `rnn_type` was supplied, "
                             "options are ['LSTM', 'GRU', 'RNN_TANH' or "
                             "'RNN_RELU']")
        return nn.RNN(ninputs, nhid, nlayers, nonlinearity=nonlinearity,
                      dropout=dropout, bidirectional=bidirectional)


class Encoder(nn.Module):
    """Encodes text x of style y into content vector z"""
    def __init__(self, nhid, embeddings, rnn_type='LSTM',
                 nlayers=1, dropout=0.2):
        super(Encoder, self).__init__()

        self.embeddings = embeddings
        self.drop = nn.Dropout(dropout)
        self.rnn = get_rnn(rnn_type, self.embeddings.embedding_dim, 
                           nhid, nlayers, dropout)
        #self.projection = nn.Linear(nhid, dim)
        
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.initial_state_dim = nhid * nlayers
        if rnn_type == 'LSTM':
            self.initial_state_dim *= 2

    def forward(self, style, data, seq_lens):
        """
        Here style, encoded style of a given text, 
        serves as initial hidden state of encoder.
        """
        emb = self.drop(self.embeddings(data))
        output, _ = self.rnn(emb, style)
        
        _, batch_size, hidden_dim = output.size()
        arange = seq_lens.data.new(*seq_lens.size()).copy_(
                torch.arange(0, batch_size))
        last_states = output[seq_lens.data - 1, arange]
       
        last_states = self.drop(last_states)
        #output = self.projection(last_states)
        return last_states

    def split_initial_hidden(self, initial_hidden):
        initial_hidden = initial_hidden.view(
        initial_hidden.size(0), -1, self.nhid)
        initial_hidden = initial_hidden.transpose(0, 1).contiguous()
        if self.rnn_type == 'LSTM':
            initial_hidden = initial_hidden.chunk(2)
        return initial_hidden


class Generator(nn.Module):
    """Generates text x based on content z and style y"""
    def __init__(self, nhid, embeddings, eos_id, gamma=0.001, 
                 rnn_type='LSTM', nlayers=1, dropout=0.2):
        super(Generator, self).__init__()
        
        self.embeddings = embeddings
        self.drop = nn.Dropout(dropout)
        self.rnn = get_rnn(rnn_type, self.embeddings.embedding_dim, 
                           nhid, nlayers, dropout)
        self.projection = nn.Linear(nhid, self.embeddings.num_embeddings)
        self.softmax = nn.Softmax()
        self.gamma = gamma
        self.rnn_type = rnn_type
        self.eos = eos_id
        self.nhid = nhid

        self.initial_state_dim = nhid * nlayers
        if rnn_type == 'LSTM':
            self.initial_state_dim *= 2

    def forward(self, init_hidden, data, teacher_forcing):
        hidden = init_hidden
        
        if teacher_forcing:
            emb = self.drop(self.embeddings(data))
            outputs, hidden = self.rnn(emb, hidden)
            outputs = self.drop(outputs)
            decoded = self.projection(
                outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2)))
            decoded = decoded.view(
                outputs.size(0), outputs.size(1), decoded.size(1))
            
            return decoded, outputs, hidden
        
        else: # sampling
            outputs = []
            max_len, batch_size = data.size()
            eos = Variable(data.data.new([self.eos]))
            emb = self.embeddings(eos).expand(
                    [1, batch_size, self.embeddings.embedding_dim])

            for step in range(max_len):
                output, hidden = self.rnn(emb, hidden)
                outputs.append(output)

                decoded = self.projection(
                        output.view([-1, output.size(2)]))

                emb = torch.matmul(
                        self.softmax(decoded/self.gamma), self.embeddings.weight)
                emb = emb.view([1, batch_size, self.embeddings.embedding_dim])

            return torch.cat(outputs, 0), hidden

    def split_initial_hidden(self, initial_hidden):
        initial_hidden = initial_hidden.view(
        initial_hidden.size(0), -1, self.nhid)
        initial_hidden = initial_hidden.transpose(0, 1).contiguous()
        if self.rnn_type == 'LSTM':
            initial_hidden = initial_hidden.chunk(2)
        return initial_hidden


class Discriminator(nn.Module):
    def __init__(self, dim, filter_sizes, n_filters, 
            activation=nn.LeakyReLU(), dropout=0.2):
        super(Discriminator, self).__init__()
        
        self.activation = activation
        self.drop = nn.Dropout(dropout)
        self.projection = nn.Linear(len(filter_sizes)*n_filters, 1)
        self.conv_num = len(filter_sizes)

        for i, f_size in enumerate(filter_sizes):
            setattr(self, "convolution%d" % i, 
                    nn.Conv2d(1, n_filters, [f_size, dim]))
        
    def forward(self, data):
        data = data.transpose(0, 1).unsqueeze(1)
        outputs = []
        for i in range(self.conv_num):
            conv = getattr(self, "convolution%d" % i)
            hid = self.activation(conv(data))
            # maxpool over time:
            # Note that torch.max over a dimension 
            # returns tuple of tensors (max, argmax)
            pooled = torch.max(hid, 2)[0].contiguous()
            outputs.append(pooled.view([-1, conv.out_channels]))

        logits = self.projection(self.drop(torch.cat(outputs, dim=1)))
        return logits.squeeze(-1)


class Model(nn.Module):
    def __init__(self, ntokens, style_dim,
                 encoder_kwargs, generator_kwargs, 
                 discriminator_kwargs, lmb=1.0, tie_embeddings=True):
        super(Model, self).__init__()
        
        generator_embeddings = nn.Embedding(ntokens, generator_kwargs["nhid"])

        if tie_embeddings:
            if encoder_kwargs["nhid"] != generator_kwargs["nhid"]:
                raise ValueError("When using tied flag hidden states' dimenstions of encoder and generator must be the same")
            encoder_embeddings = generator_embeddings 
        else:
            encoder_embeddings = nn.Embedding(ntokens, encoder_kwargs["nhid"])

        self.lmb = lmb
        self.style_encoder = nn.Embedding(2, style_dim)
        self.generator = Generator(embeddings=generator_embeddings, **generator_kwargs)
        self.encoder = Encoder(embeddings=encoder_embeddings, **encoder_kwargs)
       
        self.encoder_init_projection = nn.Linear(
                style_dim, self.encoder.initial_state_dim)
        self.generator_init_projection = nn.Linear(
                style_dim+encoder_kwargs["nhid"], 
                self.generator.initial_state_dim)
        
        self.discriminator = [
                Discriminator(dim=generator_kwargs['nhid'], 
                    **discriminator_kwargs)
                for i in range(2)
                ]
        
        self.rec_criterion = nn.CrossEntropyLoss(
                ignore_index=-1)
        self.adv_criterion = nn.BCEWithLogitsLoss()

    def get_style_encoding(self, style_id, data):
        style = data.data.new([style_id])
        style = self.style_encoder(Variable(style))
        style = style.expand([data.size(1), 
            self.style_encoder.embedding_dim])

        return style

    def compute_losses(self, data, targets, seq_lens):
        assert (data[0].size(1) == data[1].size(1))
        
        batch_size = data[0].size(1)

        style = [self.get_style_encoding(0, data[0]),
                self.get_style_encoding(1, data[1])]
        
        labels = [ 
            Variable(data[0].data.new([0])).expand([batch_size]).float(),
            Variable(data[1].data.new([1])).expand([batch_size]).float()
            ]

        rec_loss = 0.0
        adv_loss = [0.0, 0.0]

        for (p, q) in [(0,1), (1,0)]:
            encoder_init = self.encoder_init_projection(style[p])
            encoder_init = self.encoder.split_initial_hidden(encoder_init)
            content = self.encoder(encoder_init, data[p], seq_lens[p])

            generator_init_true = self.generator_init_projection(
                    torch.cat([style[p], content], 1))
            generator_init_true = self.generator.split_initial_hidden(
                    generator_init_true)
            output, hiddens_true, _ = self.generator(
                    generator_init_true, data[p], teacher_forcing=True)
            
            generator_init_false = self.generator_init_projection(
                    torch.cat([style[q], content], 1))
            generator_init_false = self.generator.split_initial_hidden(
                    generator_init_false)
            hiddens_false, _ = self.generator(
                    generator_init_false, data[p], teacher_forcing=False)

            output_flat = output.view(-1, output.size(-1))
            targets_flat = targets[p].view(-1)
            
            
            rec_loss += self.rec_criterion(output_flat, targets_flat)

            adv_loss[p] += self.adv_criterion(
                    self.discriminator[p](hiddens_true),
                    labels[1])

            adv_loss[q] += self.adv_criterion(
                    self.discriminator[q](hiddens_false),
                    labels[0])

        return rec_loss, adv_loss[0], adv_loss[1]

            
    def eval_on(self, batch_iterator):
        # Turn on evaluation mode which disables dropout.
        self.eval()
        for d in self.discriminator:
            d.eval()
        total_rec = 0.0
        total_adv0 = 0.0
        total_adv1 = 0.0
        for batch_no, (data, targets, seq_lens) in enumerate(batch_iterator):
            rec, adv0, adv1 = self.compute_losses(data, targets, seq_lens)
            batch_size = data[0].size()[1]
            total_rec += batch_size * rec.data[0]
            total_adv0 += adv0.data[0]
            total_adv1 += adv1.data[0]


        total_rec /= (batch_no + 1.0)
        total_adv0 /= (batch_no + 1.0)
        total_adv1 /= (batch_no + 1.0)

        print ("batch %d: rec_loss: %f, ae_loss: %f, adv0: %f, adv1: %f" % 
                (total_rec, total_rec - self.lmb * (total_adv0 + total_adv1), 
                total_adv0, total_adv1))

        return {'total_ae_loss': 
                total_rec - self.lmb * (total_adv0 + total_adv1),
                'total_adv0' : total_adv0,
                'total_adv1' : total_adv1,
                }

    def train_on(self, batch_iterator, optimizer_step):
        # Turn on training mode which enables dropout.
        self.train()
        for d in self.discriminator:
            d.train()
        for batch_no, (data, targets, seq_lens) in enumerate(batch_iterator):
            rec, adv0, adv1 = self.compute_losses(data, targets, seq_lens)
            optimizer_step(rec, adv0, adv1)

            if batch_no % 500 == 0:
                print ("batch %d: rec_loss: %f, ae_loss: %f, adv0: %f, adv1: %f" % 
                        (batch_no, rec.data[0], 
                            (rec - self.lmb * (adv0 + adv1)).data[0], 
                            adv0.data[0], adv1.data[0]))


