"""Specifies a particular instance of a soft attention model.

We use the global attention model with input feeding
used by Luong et al. (2015).
See http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf
"""
import numpy
import theano
from theano import tensor as T

from outputlayer import OutputLayer
from spec import Spec

class Attention2HistorySpec(Spec):
  """Abstract class for a specification of an encoder-decoder model.
  
  Concrete subclasses must implement the following method:
  - self.create_rnn_layer(vocab, hidden_size): Create an RNN layer.
  """
  def __init__(self, in_vocabulary, out_vocabulary, domain_size, lexicon, hidden_size,
               rnn_type='lstm', step_rule='simple', **kwargs):
    """Initialize.

    Args:
      in_vocabulary: list of words in the vocabulary of the input
      out_vocabulary: list of words in the vocabulary of the output
      embedding_dim: dimension of word vectors
      hidden_size: dimension of hidden layer
    """
    self.in_vocabulary = in_vocabulary
    self.out_vocabulary = out_vocabulary
    self.lexicon = lexicon
    self.domain_size = domain_size
    self.hidden_size = hidden_size
    self.rnn_type = rnn_type
    self.step_rule = step_rule
    self.create_vars()
    self.cur_y_in_domain = 0
    self._process_init_kwargs(**kwargs)


  def _process_init_kwargs(self, pair_stat,em_model,test_domain, attention_copying=False):
    self.attention_copying = attention_copying
    self.pair_stat = pair_stat
    self.em_model = em_model
    self.test_domain = test_domain

  def set_test_domain(self,test_domain):
      self.test_domain = test_domain

  def create_vars(self):
    if self.rnn_type == 'lstm' or self.rnn_type =='atnh':
      annotation_size = 4 * self.hidden_size
      dec_full_size = 2 * self.hidden_size
    else:
      annotation_size = 2 * self.hidden_size
      dec_full_size = self.hidden_size

    self.fwd_encoder = self.create_rnn_layer(
        self.hidden_size, self.in_vocabulary.emb_size,
        self.in_vocabulary.size(), True)
    self.bwd_encoder = self.create_rnn_layer(
        self.hidden_size, self.in_vocabulary.emb_size,
        self.in_vocabulary.size(), True)
    self.decoder = self.create_rnn_layer(
        self.hidden_size, self.out_vocabulary.emb_size + annotation_size,
        self.out_vocabulary.size(), False)
    self.writer = self.create_output_layer(self.out_vocabulary,self.domain_size,
                                           self.hidden_size + annotation_size)
    self.w_local_history = theano.shared(
        name='w_local_history',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.hidden_size, annotation_size)).astype(theano.config.floatX))

    self.w_local_attention = theano.shared(
        name='w_local_attention',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.hidden_size, annotation_size)).astype(theano.config.floatX))
    self.w_enc_to_dec = theano.shared(
        name='w_enc_to_dec',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (dec_full_size, annotation_size)).astype(theano.config.floatX))
    self.w_attention = theano.shared(
        name='w_attention',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.hidden_size, self.in_vocabulary.size())).astype(theano.config.floatX))
    self.w_domain_attention_c_t = theano.shared(
        name='w_domain_attention_c_t',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (annotation_size, self.domain_size)).astype(theano.config.floatX))
    self.w_domain_attention_h_t = theano.shared(
        name='w_domain_attention_h_t',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.hidden_size, self.domain_size)).astype(theano.config.floatX))

    self.w_history = theano.shared(
        name='w_history',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.in_vocabulary.size(),annotation_size)).astype(theano.config.floatX))
    self.u_zt = theano.shared(
        name='u_zt',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.in_vocabulary.size(),self.hidden_size)).astype(theano.config.floatX))
    self.w_zt = theano.shared(
        name='w_zt',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.in_vocabulary.size(),annotation_size)).astype(theano.config.floatX))
    self.w_co = theano.shared(
        name='w_co',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.in_vocabulary.size())).astype(theano.config.floatX))
    self.r_t = theano.shared(
        name='r_t',
        value=0.1 * numpy.random.uniform(0.0, 1.0, (self.out_vocabulary.size())).astype(theano.config.floatX))
    self.d_t = theano.shared(
        name='d_t',
        value=0.1 * numpy.random.uniform(-0.1, 1.0, (self.hidden_size,self.domain_size)).astype(theano.config.floatX))
    
    self.alpha = theano.shared(
        name='alpha',
        value=0.1 * numpy.random.uniform(0.0, 1.0, (1)).astype(theano.config.floatX))



  def set_pair_stat(self,pair_stat):
      self.pair_stat = pair_stat

  def set_em_model(self,em_model):
      self.em_model = em_model
  
  def get_local_params(self):
    return (self.fwd_encoder.params + self.bwd_encoder.params + 
            self.decoder.params + self.writer.params + [self.w_enc_to_dec] + [self.w_history]+[self.w_local_attention]+[self.w_co]+[self.w_zt]+[self.u_zt]+[self.w_local_history]+[self.r_t]+[self.alpha]+[self.w_domain_attention_h_t]+[self.w_domain_attention_c_t])

  def create_output_layer(self, vocab, domain_size, hidden_size):
    return OutputLayer(vocab, domain_size, hidden_size)

  def get_init_fwd_state(self):
    return self.fwd_encoder.get_init_state()

  def get_init_bwd_state(self):
    return self.bwd_encoder.get_init_state()

  def f_enc_fwd(self, x_t, h_prev):
    """Returns the next hidden state for forward encoder."""
    input_t = self.in_vocabulary.get_theano_embedding(x_t)
    return self.fwd_encoder.step(input_t, h_prev) #hiB = LSTM(phi(xi),hi-1B)

  def f_enc_bwd(self, x_t, h_prev):
    """Returns the next hidden state for backward encoder."""
    input_t = self.in_vocabulary.get_theano_embedding(x_t)
    return self.bwd_encoder.step(input_t, h_prev) #hiF = LSTM(phi(xi),hi-1F)

  def get_dec_init_state(self, enc_last_state):
    return T.tanh(T.dot(self.w_enc_to_dec, enc_last_state))#s1 = tanh(Ws[hmF,h1B])

  def f_dec(self, y_t, c_prev, h_prev):
    """Returns the next hidden state for decoder."""
    y_emb_t = self.out_vocabulary.get_theano_embedding(y_t)
    input_t = T.concatenate([y_emb_t, c_prev]) # [phi(yj);cj]
    return self.decoder.step(input_t, h_prev)

  def get_attention_scores_inner(self, h_for_write, annotations):
    S1 = T.dot(self.w_local_attention, self.w_history.T).T# eji = sjT * Wa * bi
    return S1

  def get_local_attention_scores(self, h_for_write, annotations):
    return T.dot(T.dot(self.w_local_attention, annotations.T).T, h_for_write) # eji = sjT * Wa * bi

  def get_domain_scores_h_t(self, h_for_write):
    return T.dot(self.w_domain_attention_h_t.T,  h_for_write) # eji = sjT * Wa * bi

  def get_domain_scores_c_t(self, c_t):
    return T.dot(self.w_domain_attention_c_t.T, c_t) # eji = sjT * Wa * bi
  
  def get_attention_scores(self, h_for_write, annotations):
    loc_scores = T.dot(T.dot(self.w_local_attention, annotations.T).T, h_for_write)
    loc_alpha = self.get_alpha(loc_scores)
    loc_c_t = self.get_local_context(loc_alpha,annotations)
    z_t = T.nnet.sigmoid(T.dot(loc_c_t,self.w_zt.T)+T.dot(h_for_write,self.u_zt.T))
    return loc_scores


  def get_alpha(self, scores):
    alpha = T.nnet.softmax(scores)[0] # exp(eji)/sumi(exp(eji))
    return alpha

  def get_local_context(self, alpha,annotations):
    c_t = T.dot(alpha, annotations)
    return c_t

  def get_context(self, alpha,annotations):
    c_t = T.dot(alpha, annotations)
    return c_t

  def f_write(self, h_t, c_t, cur_domain, scores):
    """Gives the softmax output distribution."""
    input_t = T.concatenate([h_t, c_t])
    if not self.attention_copying:
        scores = None
    in_domain_scores = self.writer.write(input_t,cur_domain, scores)
    return in_domain_scores
