"""An output layer."""
import numpy
import theano
from theano.ifelse import ifelse
import theano.tensor as T

class OutputLayer(object):
  """Class that sepcifies parameters of an output layer.
  
  Conventions used by this class (shared with spec.py):
    nh: dimension of hidden layer
    nw: number of words in the vocabulary
    de: dimension of word embeddings
  """ 
  def __init__(self, vocab, domain_size, hidden_size):
    self.vocab = vocab
    self.de = vocab.emb_size
    self.nh = hidden_size
    self.nw = vocab.size()
    self.dm = domain_size
    self.ws = []
    self.create_vars()
    #self.domain = 0

  def set_domain(self,domain):
      self.domain = domain
  def create_vars(self):
      #for i in range(self.dm):
    self.z_t = theano.shared(
            name='z_t', 
            value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nw,)).astype(theano.config.floatX))

    self.w_out = theano.shared(
            name='w_out', 
            value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nw, self.nh)).astype(theano.config.floatX))
    self.w_domain_attention_y_t = theano.shared(
        name='w_domain_attention_y_t',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.dm*self.nh,self.nh)).astype(theano.config.floatX))


    #self.ws.append(w_out)
    self.params = [self.w_out]+[self.w_domain_attention_y_t]+[self.z_t]
        #self.params.append(self.w_out)
        # Each row is one word

  def write(self, h_t, cur_domain, attn_scores=None):
    """Get a distribution over words to write.
    
    Entries in [0, nw) are probablity of emitting i-th output word,
    and entries in [nw, nw + len(attn_scores))
    are probability of copying the (i - nw)-th word.

    Args:
      h_t: theano vector representing hidden state
      attn_scores: unnormalized scores from the attention module, if doing 
          attention-based copying.
    """
    W = T.dot(h_t,self.w_domain_attention_y_t.T)
    Wr = T.tanh(T.reshape(W,(self.dm,self.nh)))
    #Wr = T.reshape(W,(self.dm,self.nh))
    H = T.dot(cur_domain,Wr)
    scores = T.dot(H, self.w_out.T)
    '''wh =  T.dot(cur_domain,self.w_domain_attention_y_t)
    wh_reshape = T.reshape(wh,(self.nw,self.nh))
    scores_2 = T.dot(h_t,wh_reshape.T)
    scores = scores_1 + T.nnet.sigmoid(self.z_t)*scores_2'''
    
    if attn_scores:
      return T.nnet.softmax(T.concatenate([scores, attn_scores]))[0] #p(aj=write[w]|x,y) = exp(Uw[sj,cj]) 
    else:
        return T.nnet.softmax(scores)[0] # p(aj=copy[i]) = exp(eji)
