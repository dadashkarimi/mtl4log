"""Run tests on toy data for IRW models."""
import argparse
import cgi
import collections
import itertools
import json
import math
import numpy
import os
import random
import re
import sys
import theano
import pickle

# Local imports
import atislexicon
from augmentation import Augmenter
import domains
from encoderdecoder import EncoderDecoderModel
from attention import AttentionModel
from attention_multi import AttentionModelMulti
from example import Example
import spec as specutil
from vocabulary import Vocabulary


MODELS = collections.OrderedDict([
    ('encoderdecoder', EncoderDecoderModel),
    ('attention', AttentionModel),
    ('attention_multi', AttentionModelMulti)
])

VOCAB_TYPES = collections.OrderedDict([
    ('raw', lambda s, e, **kwargs: Vocabulary.from_sentences(
        s, e, **kwargs)), 
    ('glove', lambda s, e, **kwargs: Vocabulary.from_sentences(
        s, e, use_glove=True, **kwargs))
])

# Global options
OPTIONS = None

# Global statistics
STATS = {}

def _parse_args():
  global OPTIONS
  parser = argparse.ArgumentParser(
      description='A neural semantic parser.',
      formatter_class=argparse.RawTextHelpFormatter
  )
  parser.add_argument('--domain-embedding-dim', '-e', type=int, default=0,
                      help='Dimenstion of domain embedding vecore. If size is zero domain embeddings are not used')
  parser.add_argument('--hidden-size', '-d', type=int,
                      help='Dimension of hidden units')
  parser.add_argument('--input-embedding-dim', '-i', type=int,
                      help='Dimension of input vectors.')
  parser.add_argument('--output-embedding-dim', '-o', type=int,
                      help='Dimension of output word vectors.')
  parser.add_argument('--shared-encoder', '-f', type=bool, default=False,
                      help='Whether to train in a shared encoder fashion.')
  parser.add_argument('--copy', '-p', default='none',
                      help='Way to copy words (options: [none, attention, attention-logistic]).')
  parser.add_argument('--unk-cutoff', '-u', type=int, default=0,
                      help='Treat input words with <= this many occurrences as UNK.')
  parser.add_argument('--num-epochs', '-t', default=[],
                      type=lambda s: [int(x) for x in s.split(',')], 
                      help=('Number of epochs to train (default is no training).'
                            'If comma-separated list, will run for some epochs, halve learning rate, etc.'))
  parser.add_argument('--learning-rate', '-r', type=float, default=0.1,
                      help='Initial learning rate (default = 0.1).')
  parser.add_argument('--step-rule', '-s', default='simple',
                      help='Use a special SGD step size rule (types=[simple, adagrad, rmsprop,nesterov])')
  parser.add_argument('--lambda-reg', '-l', type=float, default=0.0,
                      help='L2 regularization constant (per example).')
  parser.add_argument('--rnn-type', '-c',
                      help='type of continuous RNN model (options: [%s])' % (
                          ', '.join(specutil.RNN_TYPES)))
  parser.add_argument('--model', '-m',
                      help='type of overall model (options: [%s])' % (
                          ', '.join(MODELS)))
  parser.add_argument('--input-vocab-type',
                      help='type of input vocabulary (options: [%s])' % (
                          ', '.join(VOCAB_TYPES)), default='raw')
  parser.add_argument('--output-vocab-type',
                      help='type of output vocabulary (options: [%s])' % (
                          ', '.join(VOCAB_TYPES)), default='raw')
  parser.add_argument('--reverse-input', action='store_true',
                      help='Reverse the input sentence (intended for encoder-decoder).')
  parser.add_argument('--float32', action='store_true',
                      help='Use 32-bit floats (default is 64-bit/double precision).')
  parser.add_argument('--beam-size', '-k', type=int, default=0,
                      help='Use beam search with given beam size (default is greedy).')
  parser.add_argument('--domain', default=None,
                      help='Domain for augmentation and evaluation (options: [geoquery,atis,overnight-${domain}])')
  parser.add_argument('--use-lexicon', action='store_true',
                      help='Use a lexicon for copying (should also supply --domain)')
  parser.add_argument('--augment', '-a',
                      help=('Options for augmentation.  Format: '
                            '"nesting+entity+concat2".'))
  parser.add_argument('--aug-frac', type=float, default=0.0,
                      help='How many recombinant examples to add, relative to '
                      'training set size.')
  parser.add_argument('--distract-prob', type=float, default=0.0,
                      help='Probability to introduce distractors during training.')
  parser.add_argument('--distract-num', type=int, default=0,
                      help='Number of distracting examples to use.')
  parser.add_argument('--concat-prob', type=float, default=0.0,
                      help='Probability to concatenate examples during training.')
  parser.add_argument('--concat-num', type=int, default=1,
                      help='Number of examples to concatenate together.')
  parser.add_argument('--train-data', help='Path to training data.')
  parser.add_argument('--dev-data', help='Path to dev data.')
  parser.add_argument('--dev-frac', type=float, default=0.0,
                      help='Take this fraction of train data as dev data.')
  parser.add_argument('--dev-seed', type=int, default=0,
                      help='RNG seed for the train/dev splits (default = 0)')
  parser.add_argument('--model-seed', type=int, default=0,
                      help="RNG seed for the model's initialization and SGD ordering (default = 0)")
  parser.add_argument('--save-file', help='Path to save parameters.')
  parser.add_argument('--load-file', help='Path to load parameters, will ignore other passed arguments.')
  parser.add_argument('--stats-file', help='Path to save statistics (JSON format).')
  parser.add_argument('--shell', action='store_true', 
                      help='Start an interactive shell.')
  parser.add_argument('--server', action='store_true', 
                      help='Start an interactive web console (requires bottle).')
  parser.add_argument('--hostname', default='127.0.0.1', help='server hostname')
  parser.add_argument('--port', default=9001, type=int, help='server port')
  parser.add_argument('--theano-fast-compile', action='store_true',
                      help='Run Theano in fast compile mode.')
  parser.add_argument('--theano-profile', action='store_true',
                      help='Turn on profiling in Theano.')
  parser.add_argument('--num-samples', '-n',
                      help='number of samples (options: [%s])' % (
                          ', '.join(MODELS)))                  
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  OPTIONS = parser.parse_args()
  
  # Some basic error checking
  if OPTIONS.rnn_type not in specutil.RNN_TYPES:
    print >> sys.stderr, 'Error: rnn type must be in %s' % (
        ', '.join(specutil.RNN_TYPES))
    sys.exit(1)
  if OPTIONS.model not in MODELS:
    print >> sys.stderr, 'Error: model must be in %s' % (
        ', '.join(MODELS))
    sys.exit(1)
  if OPTIONS.input_vocab_type not in VOCAB_TYPES:
    print >> sys.stderr, 'Error: input_vocab_type must be in %s' % (
        ', '.join(VOCAB_TYPES))
    sys.exit(1)
  if OPTIONS.output_vocab_type not in VOCAB_TYPES:
    print >> sys.stderr, 'Error: output_vocab_type must be in %s' % (
        ', '.join(VOCAB_TYPES))
    sys.exit(1)


def configure_theano():
  if OPTIONS.theano_fast_compile:
    theano.config.mode='FAST_COMPILE'
  else:
    theano.config.mode='FAST_RUN'
    #theano.config.mode = 'FAST_COMPILE'
    #theano.config.exception_verbosity = 'high'
    theano.config.linker='cvm'
  if OPTIONS.theano_profile:
    theano.config.profile = True

def load_dataset(filename, domain):
  dataset = []
  from collections import defaultdict
  samples = set() #defaultdict(list)
  with open(filename) as f:
    i=0
    for line in f:
      parts = line.rstrip('\n').split('\t')
      x = parts[0]
      y = parts [1]

      if len(parts) == 2:
        z = 'overnight-'+domain.subdomain
      else:
        z = parts[2]
      if domain:
        y = domain.preprocess_lf(y)
      if z in samples:
        i=0
        continue
      elif i > int(OPTIONS.num_samples) and re.match(".+_train.tsv", filename):
        samples.add(z)
      i+=1
      dataset.append((x, y, z))
  return dataset

def get_input_vocabulary(dataset):
  sentences = [x[0] for x in dataset]
  constructor = VOCAB_TYPES[OPTIONS.input_vocab_type]
  if OPTIONS.float32:
    return constructor(sentences, OPTIONS.input_embedding_dim,
                       unk_cutoff=OPTIONS.unk_cutoff,
                       float_type=numpy.float32)
  else:
    return constructor(sentences, OPTIONS.input_embedding_dim,
                       unk_cutoff=OPTIONS.unk_cutoff)

def get_output_vocabulary(dataset):
  sentences = [x[1] for x in dataset]
  constructor = VOCAB_TYPES[OPTIONS.output_vocab_type]
  if OPTIONS.float32:
    return constructor(sentences, OPTIONS.output_embedding_dim,
                       float_type=numpy.float32)
  else:
    return constructor(sentences, OPTIONS.output_embedding_dim)


def get_domain_vocabulary(dataset):
  sentences = [x[2] for x in dataset]
  constructor = VOCAB_TYPES[OPTIONS.input_vocab_type]
  if OPTIONS.domain_embedding_dim == -1:   # fixed one-hot vector to represent domain
    domain_set = set()
    [domain_set.add(x) for x in sentences]
    domain_embedding_dim = len(domain_set)
    fixed_embed = True
  else:   # learned domain embedding
    domain_embedding_dim = OPTIONS.domain_embedding_dim
    fixed_embed = False
  if OPTIONS.float32:
    return constructor(sentences, domain_embedding_dim,
                       special_tokens=False, fixed_embed=fixed_embed,
                       float_type=numpy.float32)
  else:
    return constructor(sentences, domain_embedding_dim,
                       special_tokens=False, fixed_embed=fixed_embed,
                       )

def get_input_vocabularies(dataset):
  vocabs = {}
  domain_sents = {}
  for x in dataset:
    sent = x[0]
    domain = x[2]
    if domain not in domain_sents:
      domain_sents[domain] = []
    domain_sents[domain].append(sent)
  constructor = VOCAB_TYPES[OPTIONS.input_vocab_type]
  if OPTIONS.float32:
    for domain in domain_sents:
      vocabs[domain] = constructor(domain_sents[domain], OPTIONS.input_embedding_dim,
                       unk_cutoff=OPTIONS.unk_cutoff,
                       float_type=numpy.float32)
  else:
    for domain in domain_sents:
      vocabs[domain] = constructor(domain_sents[domain], OPTIONS.input_embedding_dim,
                       unk_cutoff=OPTIONS.unk_cutoff)
  return vocabs

def get_output_vocabularies(dataset):
  vocabs = {}
  domain_sents = {}
  for x in dataset:
    sent = x[1]
    domain = x[2]
    if domain not in domain_sents:
      domain_sents[domain] = []
    domain_sents[domain].append(sent)
  constructor = VOCAB_TYPES[OPTIONS.output_vocab_type]
  if OPTIONS.float32:
    for domain in domain_sents:
      vocabs[domain] = constructor(domain_sents[domain], OPTIONS.output_embedding_dim,
                       float_type=numpy.float32)
  else:
    for domain in domain_sents:
      vocabs[domain] = constructor(domain_sents[domain], OPTIONS.output_embedding_dim)
  return vocabs

def update_model(model, dataset):
  """Update model for new dataset if fixed word vectors were used.
  
  Note: glove_fixed has been removed for now.
  """
  need_new_model = False
  if OPTIONS.input_vocab_type == 'glove_fixed':
    in_vocabulary = get_input_vocabulary(dataset)
    need_new_model = True
  else:
    in_vocabulary = model.in_vocabulary

  if OPTIONS.output_vocab_type == 'glove_fixed':
    out_vocabulary = get_output_vocabulary(dataset)
    need_new_model = True
  else:
    out_vocabulary = model.out_vocabulary

  if need_new_model:
    spec = model.spec
    spec.set_in_vocabulary(in_vocabulary)
    spec.set_out_vocabulary(out_vocabulary)
    model = get_model(spec)  # Create a new model!
  return model

def preprocess_data(model, raw):
  lexicon = model.lexicon

  data = []
  for raw_ex in raw:
    x_str, y_str, sub_domain = raw_ex
    in_vocabulary = model.specs[sub_domain].in_vocabulary
    out_vocabulary = model.specs[sub_domain].out_vocabulary
    domain_vocaulary = model.specs[sub_domain].domain_vocabulary
    in_vocabulary_shared = model.specs['_shared_'].in_vocabulary
    out_vocabulary_shared = model.specs['_shared_'].out_vocabulary
    ex = Example(x_str, y_str, in_vocabulary_shared, out_vocabulary_shared, domain_vocaulary, lexicon,
                 reverse_input=OPTIONS.reverse_input, sub_domain=sub_domain)
    ex_domain = Example(x_str, y_str, in_vocabulary, out_vocabulary, domain_vocaulary,
                  lexicon, reverse_input=OPTIONS.reverse_input, sub_domain=sub_domain)
    data.append((ex, ex_domain))
  return data

def get_spec(in_vocabulary, out_vocabulary, domain_vocabulary, domain, lexicon):
  kwargs = {'rnn_type': OPTIONS.rnn_type, 'step_rule': OPTIONS.step_rule}
  if OPTIONS.copy.startswith('attention'):
    if OPTIONS.model == 'attention' or OPTIONS.model == 'attention_multi':
      kwargs['attention_copying'] = OPTIONS.copy
    else:
      print >> sys.stderr, "Can't use use attention-based copying without attention model"
      sys.exit(1)
  constructor = MODELS[OPTIONS.model].get_spec_class()
  return constructor(in_vocabulary, out_vocabulary, domain_vocabulary, domain, lexicon,
                     OPTIONS.hidden_size, **kwargs)

def get_model(specs):
  constructor = MODELS[OPTIONS.model]
  if OPTIONS.float32:
    model = constructor(specs, distract_num=OPTIONS.distract_num, float_type=numpy.float32)
  else:
    model = constructor(specs, distract_num=OPTIONS.distract_num)
  return model

def print_accuracy_metrics(name, is_correct_list, tokens_correct_list,
                           x_len_list, y_len_list, denotation_correct_list):
  # Overall metrics
  num_examples = len(is_correct_list)
  num_correct = sum(is_correct_list)
  num_tokens_correct = sum(tokens_correct_list)
  num_tokens = sum(y_len_list)
  seq_accuracy = float(num_correct) / num_examples
  token_accuracy = float(num_tokens_correct) / num_tokens

  STATS[name] = {}

  # Print sequence-level accuracy
  STATS[name]['sentence'] = {
      'correct': num_correct,
      'total': num_examples,
      'accuracy': seq_accuracy,
  }
  print 'Sequence-level accuracy: %d/%d = %g' % (num_correct, num_examples, seq_accuracy)

  # Print token-level accuracy
  STATS[name]['token'] = {
      'correct': num_tokens_correct,
      'total': num_tokens,
      'accuracy': token_accuracy,
  }
  print 'Token-level accuracy: %d/%d = %g' % (num_tokens_correct, num_tokens, token_accuracy)

  # Print denotation-level accuracy
  if denotation_correct_list:
    denotation_correct = sum(denotation_correct_list)
    denotation_accuracy = float(denotation_correct)/num_examples
    STATS[name]['denotation'] = {
        'correct': denotation_correct,
        'total': num_examples,
        'accuracy': denotation_accuracy
    }
    print 'Denotation-level accuracy: %d/%d = %g' % (denotation_correct, num_examples, denotation_accuracy)

def decode(model, ex):
  if OPTIONS.beam_size == 0:
    return model.decode_greedy(ex, max_len=100)
  else:
    return model.decode_beam(ex, beam_size=OPTIONS.beam_size)

def insert_new_sub_domain(sub_domains, new_sub_domain):
  results = {}
  results['is_correct'] = []
  results['tokens_correct'] = []
  results['x_len'] = []
  results['y_len'] = []
  results['denotation_correct'] = []
  sub_domains[new_sub_domain] = results


def evaluate_multiple_domains(name, model, dataset, domain=None):
  sub_domains = {}
  for ex in dataset:
    ex_shared = ex[0]
    sd = ex_shared.sub_domain
    if sd not in sub_domains:
      sub_domains[sd] = []
    sub_domains[sd].append(ex)

  for sub_domain_str in sub_domains:
    print sub_domain_str
    sub_domain = domains.new(sub_domain_str)
    evaluate(name, model, sub_domains[sub_domain_str], sub_domain)


def evaluate(name, model, dataset, domain=None):
  """Evaluate the model. """
  is_correct_list = []
  tokens_correct_list = []
  x_len_list = []
  y_len_list = []

  if domain:
    all_derivs = [decode(model, ex) for ex in dataset]
    true_answers = [ex[1].y_str for ex in dataset]
    derivs, denotation_correct_list = domain.compare_answers(true_answers, all_derivs)
  else:
    derivs = [decode(model, ex)[0] for ex in dataset]
    denotation_correct_list = None

  for i, ex in enumerate(dataset):
    print 'Example %d' % i
    print '  x      = "%s"' % ex[1].x_str
    print '  y      = "%s"' % ex[1].y_str
    prob = derivs[i].p
    y_pred_toks = derivs[i].y_toks
    y_pred_str = ' '.join(y_pred_toks)

    # Compute accuracy metrics
    is_correct = (y_pred_str == ex[1].y_str)
    tokens_correct = sum(a == b for a, b in zip(y_pred_toks, ex[1].y_toks))
    is_correct_list.append(is_correct)
    tokens_correct_list.append(tokens_correct)
    x_len_list.append(len(ex[1].x_toks))
    y_len_list.append(len(ex[1].y_toks))
    print '  y_pred = "%s"' % y_pred_str
    print '  sequence correct = %s' % is_correct
    print '  token accuracy = %d/%d = %g' % (
        tokens_correct, len(ex[1].y_toks), float(tokens_correct) / len(ex[1].y_toks))
    if denotation_correct_list:
      denotation_correct = denotation_correct_list[i]
      print '  denotation correct = %s' % denotation_correct
  print_accuracy_metrics(name + '_' + domain.subdomain, is_correct_list, tokens_correct_list,
                         x_len_list, y_len_list, denotation_correct_list)


def run_shell(model):
  print '==== Neural Network Semantic Parsing REPL ===='
  print ''
  print 'Enter an utterance:'
  while True:
    s = raw_input('> ').strip()
    example = Example(s, '', model.in_vocabulary, model.out_vocabulary,
                      model.lexicon, reverse_input=OPTIONS.reverse_input)
    print ''
    print 'Result:'
    preds = decode(model, example)
    for prob, y_toks in preds[:10]:
      y_str = ' '.join(y_toks)
      print '  [p=%f] %s' % (prob, y_str)
    print ''

def make_heatmap(x_str, y_str, attention_list, copy_list):
  """Make an HTML heatmap of attention."""
  def css_color(r, g, b):
    """r, g, b are in 0-1, make """
    r2 = int(r * 255)
    g2 = int(g * 255)
    b2 = int(b * 255)
    return 'rgb(%d,%d,%d)' % (r2, g2, b2)

  x_toks = [cgi.escape(w) for w in x_str.split(' ')] + ['EOS']
  if y_str == '':
    y_toks = ['EOS']
  else:
    y_toks = [cgi.escape(w) for w in y_str.split(' ')] + ['EOS']
  lines = ['<table>', '<tr>', '<td/>']
  for w in y_toks:
    lines.append('<td>%s</td>' % w)
  lines.append('</tr>')
  for i, w in enumerate(x_toks):
    lines.append('<tr>')
    lines.append('<td>%s</td>' % w)
    for j in range(len(y_toks)):
      do_copy = copy_list[j]
      if do_copy:
        color = css_color(1 - attention_list[j][i], 1 - attention_list[j][i], 1)
      else:
        color = css_color(1, 1 - attention_list[j][i], 1 - attention_list[j][i])
      lines.append('<td/ style="background-color: %s">' % color)
    lines.append('</tr>')
  lines.append('</table>')
  return '\n'.join(lines)

def run_server(model, hostname='127.0.0.1', port=9001):
  import bottle
  print '==== Neural Network Semantic Parsing Server ===='

  app = bottle.Bottle()
  
  @app.route('/debug')
  def debug():
    content = make_heatmap(
        'what states border texas',
        'answer ( A , ( state ( A ) , next_to ( A , B ) , const ( B , stateid ( texas ) ) ) )',
        [[0.0, 0.25, 0.5, 0.75, 1.0]] * 29)
    return bottle.template('main', prompt='Enter a new query', content=content)

  @app.route('/post_query')
  def post_query():
    query = bottle.request.params.get('query')
    print 'Received query: "%s"' % query
    example = Example(query, '', model.in_vocabulary, model.out_vocabulary,
                      model.lexicon, reverse_input=OPTIONS.reverse_input)
    preds = decode(model, example)
    lines = ['<b>Query: "%s"</b>' % query, '<ul>']
    for i, deriv in enumerate(preds[:10]):
      y_str = ' '.join(deriv.y_toks)
      lines.append('<li> %d. [p=%f] %s' % (i, deriv.p, y_str))
      lines.append(make_heatmap(query, y_str, deriv.attention_list, deriv.copy_list))
    lines.append('</ul>')

    content = '\n'.join(lines)
    return bottle.template('main', prompt='Enter a new query', content=content)

  @app.route('/')
  def index():
    return bottle.template('main', prompt='Enter a query', content='')

  bottle.run(app, host=hostname, port=port)

def load_raw_all(domain=None):
  # Load train, and dev too if dev-frac was provided
  random.seed(OPTIONS.dev_seed)
  if OPTIONS.train_data:
    train_raw = load_dataset(OPTIONS.train_data, domain=domain)
    if OPTIONS.dev_frac > 0.0:
      num_dev = int(round(len(train_raw) * OPTIONS.dev_frac))
      random.shuffle(train_raw)
      dev_raw = train_raw[:num_dev]
      train_raw = train_raw[num_dev:]
      print >> sys.stderr, 'Split dataset into %d train, %d dev examples' % (
          len(train_raw), len(dev_raw))
    else:
      dev_raw = None
  else:
    train_raw = None
    dev_raw = None

  # Load dev data from separate file
  if OPTIONS.dev_data:
    if dev_raw:
      # Overwrite dev frac from before, if it existed
      print >> sys.stderr, 'WARNING: Replacing dev-frac dev data with dev-data'
    dev_raw = load_dataset(OPTIONS.dev_data, domain=domain)

  return train_raw, dev_raw

def get_augmenter(train_raw, domain):
  if OPTIONS.augment:
    aug_types = OPTIONS.augment.split('+')
    augmenter = Augmenter(domain, train_raw, aug_types)
    return augmenter
  else:
    return None


def get_lexicon():
  if OPTIONS.use_lexicon:
    if OPTIONS.domain == 'atis':
      return atislexicon.get_lexicon()
    raise Exception('No lexicon for domain %s' % OPTIONS.domain)
  return None

def init_spec(train_raw):
  if OPTIONS.load_file:
    print >> sys.stderr, 'Loading saved params from %s' % OPTIONS.load_file
    with open(OPTIONS.load_file) as f:
      specs = pickle.load(f)
  elif OPTIONS.train_data:
    print >> sys.stderr, 'Initializing parameters...'
    numpy.random.seed(1)
    domain_vocabulary = get_domain_vocabulary(train_raw)
    numpy.random.seed(0)
    in_vocabulary = get_input_vocabulary(train_raw)
    out_vocabulary = get_output_vocabulary(train_raw)
    in_vocabularies = get_input_vocabularies(train_raw)
    out_vocabularies = get_output_vocabularies(train_raw)

    lexicon = get_lexicon()
    specs = {}
    spec_shared = get_spec(in_vocabulary, out_vocabulary, domain_vocabulary, '_shared_', lexicon)
    specs['_shared_'] = spec_shared
    for domain in in_vocabularies:
      specs[domain] = get_spec(in_vocabularies[domain], out_vocabularies[domain], domain_vocabulary, domain, lexicon)
  else:
    raise Exception('Must either provide parameters to load or training data.')
  return specs

def evaluate_train(model, train_data, domain=None):
  print >> sys.stderr, 'Evaluating on training data...'
  print 'Training data:'
  #evaluate('train', model, train_data, domain=domain)
  evaluate_multiple_domains('train', model, train_data, domain=domain)

def evaluate_dev(model, dev_raw, domain=None):
  print >> sys.stderr, 'Evaluating on dev data...'
  #dev_model = update_model(model, dev_raw)
  dev_model = model
  dev_data = preprocess_data(dev_model, dev_raw)
  print 'Dev data:'
  evaluate_multiple_domains('dev', dev_model, dev_data, domain=domain)

def write_stats():
  if OPTIONS.stats_file:
    out = open(OPTIONS.stats_file, 'w')
    print >>out, json.dumps(STATS)
    out.close()

def run():
  configure_theano()
  domain = None
  if OPTIONS.domain:
    domain = domains.new(OPTIONS.domain)
  train_raw, dev_raw = load_raw_all(domain=domain)
  random.seed(OPTIONS.model_seed)
  numpy.random.seed(OPTIONS.model_seed)
  specs = init_spec(train_raw)
  model = get_model(specs)

  if train_raw:
    train_data = preprocess_data(model, train_raw)
    random.seed(OPTIONS.model_seed)
    dev_data = None
    if dev_raw:
      dev_data = preprocess_data(model, dev_raw)
    augmenter = get_augmenter(train_raw, domain)
    if not OPTIONS.load_file:
      model.train(train_data, T=OPTIONS.num_epochs, eta=OPTIONS.learning_rate,
                  dev_data=dev_data, l2_reg=OPTIONS.lambda_reg,
                  distract_prob=OPTIONS.distract_prob,
                  distract_num=OPTIONS.distract_num,
                  concat_prob=OPTIONS.concat_prob, concat_num=OPTIONS.concat_num,
                  augmenter=augmenter, aug_frac=OPTIONS.aug_frac)


  if OPTIONS.save_file:
    print >> sys.stderr, 'Saving parameters...'
    with open(OPTIONS.save_file, 'w') as f:
      pickle.dump(specs, f)

  #if train_raw:
  #  evaluate_train(model, train_data, domain=domain)
  if dev_raw:
    evaluate_dev(model, dev_raw, domain=domain)

  write_stats()

  if OPTIONS.shell:
    run_shell(model)
  elif OPTIONS.server:
    run_server(model, hostname=OPTIONS.hostname, port=OPTIONS.port)

def main():
  _parse_args()
  print OPTIONS
  print >> sys.stderr, OPTIONS
  run()

if __name__ == '__main__':
  main()
