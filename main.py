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
parser.add_argument('--logdir', type=str,  default=None,
                    help='path to save the final model')
parser.add_argument('--log_weights', action='store_true',
                    help="log weights' histograms")
parser.add_argument('--log_grads', action='store_true',
                    help="log gradients' histograms")
parser.add_argument('--load-model', action='store_true',
                    help='loads pretrained model')


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
encoder_kwargs = {'nhid': 200}
encoder_kwargs.update(eval("dict(%s)" % (args.encoder_kwargs,)))

generator_kwargs = {'nhid': 200}
generator_kwargs.update(eval("dict(%s)" % (args.generator_kwargs,)))
generator_kwargs['eos_id'] = corpus.train.class0.eos_id

discriminator_kwargs = {'filter_sizes': [3,4,5], 'n_filters': 128}
discriminator_kwargs.update(eval("dict(%s)" % (args.discriminator_kwargs,)))

model_kwargs = {'ntokens': ntokens, 'style_dim': 200}
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
    model.discriminator[0].cuda()
    model.discriminator[1].cuda()


optimizer_proto = {'sgd': optim.SGD, 'adam': optim.Adam,
                   'adagrad': optim.Adagrad, 'adadelta': optim.Adadelta}
optimizer_kwargs = eval("dict(%s)" % args.optimizer_kwargs)
optimizer_kwargs['lr'] = args.lr

ae_optimizer = optimizer_proto[args.optimizer](
        (param for name, param in model.named_parameters() 
            if name.split('.')[0] != 'discriminator'), 
        **optimizer_kwargs) 
# note that above won't update discriminators, as they are 'hidden' in a list

discriminator_optimizer = optimizer_proto[args.optimizer](
        model.discriminator.parameters(),
        **optimizer_kwargs)


def save_model():
    with open("./model.pt", 'wb') as f:
        torch.save(model.state_dict(), f)


def load_model():
    with open("./model.pt", 'rb') as f:
        model.load_state_dict(torch.load(f))


###############################################################################
# Training code
###############################################################################
# Loop over epochs.
best_val_loss = None


def optimizer_step(rec_loss, adv_loss0, adv_loss1):
    ae_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    if adv_loss0.data[0] < 1.0 and adv_loss1.data[0] < 1.0: 
        ae_loss = rec_loss - model.lmb * (adv_loss0 + adv_loss1)
    else:
        ae_loss = rec_loss
    
    ae_loss.backward(retain_graph=True)
    # `clip_grad_norm` helps prevent the exploding gradient problem in
    # RNNs / LSTMs.
    #torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    ae_optimizer.step()
    
    discriminator_optimizer.zero_grad()
    (adv_loss0 + adv_loss1).backward()
    #torch.nn.utils.clip_grad_norm(
    #        (param for D in model.discriminator for param in D.parameters()), 
    #        args.clip)
    discriminator_optimizer.step()


if args.load_model: # resume training
    load_model()

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        model.train_on(
            corpus.train.iter_epoch(
                args.batch_size, evaluation=False),
            optimizer_step)

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
            continue
            # Anneal the learning rate if no improvement has been seen on
            # the validation dataset.
            if args.lr_decay:
                assert len(optimizer.param_groups) == 1
                optimizer.param_groups[0]['lr'] /= args.lr_decay
#                logger.lr = optimizer.param_groups[0]['lr']
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
