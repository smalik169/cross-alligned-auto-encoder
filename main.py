#!/usr/bin/env python

import argparse
import time
import torch
import torch.optim as optim
import pprint

import data
import model
#from logger import Logger


parser = argparse.ArgumentParser(
    description='Style Transfer from Non-Parallel Text by Cross-Alignment')
parser.add_argument('--data', type=str,
                    default='/pio/scratch/2/i264266/cross-alligned-auto-encoder/data/sentiment.',
                    help='location of the data corpus')
parser.add_argument('--encoder-kwargs', type=str, default='',
                    help='kwargs for the encoder')
parser.add_argument('--generator-kwargs', type=str, default='',
                    help='k=v list of kwargs for the generator')
parser.add_argument('--discriminator-kwargs', type=str, default='',
                    help='kwargs for the discriminators')
parser.add_argument('--style-dim', type=int, default=200,
                    help='style embedding size')
parser.add_argument('--encoder-emb-dim', type=int, default=100,
                    help='style embedding size')
parser.add_argument('--generator-emb-dim', type=int, default=100,
                    help='style embedding size')
parser.add_argument('--tie-embeddings', type=bool, default=True,
                    help='use same word embeddings in encoder and generator')
parser.add_argument('--lmb', type=float, default=1.0,
                    help='regulates hom much of discriminator error is added to the ae_loss')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--lr-decay', type=float, default=2.0,
                    help='learning rate decay')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--optimizer', default='sgd',
                    choices=['sgd', 'adam', 'adagrad', 'adadelta'],
                    help='optimization method')
parser.add_argument('--optimizer-kwargs', type=str, default='',
                    help='kwargs for the optimizer (e.g., momentum=0.9)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--logdir', type=str, default=None,
                    help='path to save the final model')
parser.add_argument('--log_weights', action='store_true',
                    help="log weights' histograms")
parser.add_argument('--log_grads', action='store_true',
                    help="log gradients' histograms")
parser.add_argument('--load-model', action='store_true',
                    help='loads pretrained model')
parser.add_argument('--global-dropout', type=float, default=None,
                    help='set given dropout throughout whole model')


args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably "
              "run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################
corpus = data.Corpus(args.data, cuda=args.cuda, rng=args.seed)

eval_batch_size = 20
###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
encoder_kwargs = {'nhid': 500}
encoder_kwargs.update(eval("dict(%s)" % (args.encoder_kwargs,)))

generator_kwargs = {'nhid': args.style_dim+encoder_kwargs['nhid']}
generator_kwargs.update(eval("dict(%s)" % (args.generator_kwargs,)))
generator_kwargs['eos_id'] = corpus.train.class0.eos_id

discriminator_kwargs = {'filter_sizes': [3,4,5], 'n_filters': 128}
discriminator_kwargs.update(eval("dict(%s)" % (args.discriminator_kwargs,)))

model_kwargs = {'ntokens': ntokens, 'style_dim': args.style_dim,
                'encoder_emb_dim': args.encoder_emb_dim,
                'generator_emb_dim': args.generator_emb_dim,
                'tie_embeddings': args.tie_embeddings, 'lmb': args.lmb}

if args.global_dropout is not None:
    model_kwargs['dropout'] = args.global_dropout
    encoder_kwargs['dropout'] = args.global_dropout
    generator_kwargs['dropout'] = args.global_dropout
    discriminator_kwargs['dropout'] = args.global_dropout

model_kwargs['generator_kwargs'] = generator_kwargs
model_kwargs['encoder_kwargs'] = encoder_kwargs
model_kwargs['discriminator_kwargs'] = discriminator_kwargs

print("Instantiating model with args:\n%s" % (
    pprint.pformat(model_kwargs, indent=1)))

model = model.Model(**model_kwargs)

print("Model summary:\n%s" % (model,))
print("Model params:\n%s" % ("\n".join(
    ["%s: %s" % (p[0], p[1].size()) for p in model.named_parameters()])))

if args.cuda:
    model.cuda()

optimizer_proto = {'sgd': optim.SGD, 'adam': optim.Adam,
                   'adagrad': optim.Adagrad, 'adadelta': optim.Adadelta}
optimizer_kwargs = eval("dict(%s)" % args.optimizer_kwargs)
optimizer_kwargs['lr'] = args.lr

ae_optimizer = optimizer_proto[args.optimizer](
        (param for name, param in model.named_parameters()
            if name.split('.')[0] != 'discriminator'),
        **optimizer_kwargs)

#optimizer_kwargs['lr'] = args.lr
discriminator_optimizer = optimizer_proto[args.optimizer](
        model.discriminator.parameters(),
        **optimizer_kwargs)


model_path = "./model.pt"

def save_model():
    with open(model_path, 'wb') as f:
        torch.save(model.state_dict(), f)


def load_model():
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))


###############################################################################
# Training code
###############################################################################
# Loop over epochs.
best_val_loss = None


class OptimizerStep(object):
    def __init__(self, model, clip, ae_optimizer, d_optimizer,
		    ae_update_freq=2, debug=False, printout_freq=100):
        self.model = model
        self.ae_optimizer = ae_optimizer
        self.d_optimizer = d_optimizer
        self.clip = clip
        self.printout_freq = printout_freq
        self.ae_update_freq = ae_update_freq
	self.debug = debug
        self.step = 0
	self.epoch = 0

        self.curr_ae_max = 0.0
        self.curr_ae_min = float('inf')
        self.global_ae_max = 0.0
        self.global_ae_min = float('inf')

	self.curr_d_max = 0.0
        self.curr_d_min = float('inf')
        self.global_d_max = 0.0
        self.global_d_min = float('inf')

    def __call__(self, rec_loss, adv_loss0, adv_loss1, batch_no):
    	ae_total_norm = None
        if self.step % self.ae_update_freq == 0:
    	    self.model.zero_grad()
    	    ae_loss = rec_loss
    	    if max(adv_loss0.data[0], adv_loss1.data[0]) < 0.9:
    	        ae_loss = rec_loss - model.lmb * (adv_loss0 + adv_loss1)

    	    ae_loss.backward(retain_graph=True)
    	    # `clip_grad_norm` helps prevent the exploding gradient problem in
    	    # RNNs / LSTMs.
    	    ae_total_norm = torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
    	    self.ae_optimizer.step()

    	self.model.zero_grad()
    	(adv_loss0 + adv_loss1).backward()
        d_total_norm = torch.nn.utils.clip_grad_norm(
                self.model.discriminator.parameters(), self.clip)
    	self.d_optimizer.step()
        
        if self.debug:
            if ae_total_norm is not None:
                self.curr_ae_max = max(self.curr_ae_max, ae_total_norm)
                self.curr_ae_min = min(self.curr_ae_min, ae_total_norm)
            
            self.curr_d_max = max(self.curr_d_max, d_total_norm)
            self.curr_d_min = min(self.curr_d_min, d_total_norm)

            if self.step % self.printout_freq == 0:
                self.global_ae_max = max(self.global_ae_max, self.curr_ae_max)
                self.global_ae_min = min(self.global_ae_min, self.curr_ae_min)

                self.global_d_max = max(self.global_d_max, self.curr_d_max)
                self.global_d_min = min(self.global_d_min, self.curr_d_min)
                
                print ("generator: grad_norm = %s, curr_min = %f, curr_max = %f, global_min = %f, global_max = %f" 
                        % (str(ae_total_norm), self.curr_ae_min, self.curr_ae_max, self.global_ae_min, self.global_ae_max))
                print ("discriminator: grad_norm = %f, curr_min = %f, curr_max = %f, global_min = %f, global_max = %f\n" 
                        % (d_total_norm, self.curr_d_min, self.curr_d_max, self.global_d_min, self.global_d_max))
            
                self.curr_ae_max = 0.0
                self.curr_ae_min = float('inf')
                self.curr_ae_max = 0.0
                self.curr_ae_min = float('inf')
        
        self.step += 1

optimizer_step = OptimizerStep(model=model, clip=args.clip, debug=True,
        ae_optimizer=ae_optimizer, d_optimizer=discriminator_optimizer)


if args.load_model: # resume training
    load_model()

gamma_decay = 0.5
gamma_init = 1.0
gamma_min = model.generator.gamma
model.generator.gamma = gamma_init

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        model.train_on(
            corpus.train.iter_epoch(
                args.batch_size, evaluation=False),
            optimizer_step=optimizer_step)

        val_loss  = model.eval_on(
            corpus.valid.iter_epoch(
                eval_batch_size, evaluation=True))
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss['total_ae_loss'] < best_val_loss:
            save_model()
            #logger.save_model_state_dict(model.state_dict())
            #logger.save_model(model)
            best_val_loss = val_loss['total_ae_loss']
        else:
            # Anneal the learning rate if no improvement has been seen on
            # the validation dataset.
            if args.lr_decay:
                continue
                assert len(optimizer.param_groups) == 1
                optimizer.param_groups[0]['lr'] /= args.lr_decay
#                logger.lr = optimizer.param_groups[0]['lr']
        model.generator.gamma = max(
                gamma_min,
                gamma_decay*model.generator.gamma)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
#model = logger.load_model()
#model.load_state_dict(logger.load_model_state_dict())

load_model()

# Run on all data
train_loss = model.eval_on(
    corpus.train.iter_epoch(eval_batch_size, evaluation=True))
valid_loss = model.eval_on(
    corpus.valid.iter_epoch(eval_batch_size, evaluation=True))
test_loss = model.eval_on(
    corpus.test.iter_epoch(eval_batch_size, evaluation=True))

results = dict(train=train_loss, valid=valid_loss,
               test=test_loss)

#logger.final_log(results)
