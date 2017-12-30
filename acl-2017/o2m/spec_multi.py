"""Specifies a particular instance of a model."""
import numpy
import dill as pickle

from gru import GRULayer
from lstm import LSTMLayer
from vanillarnn import VanillaRNNLayer
from spec import Spec

RNN_TYPES=['vanillarnn', 'gru', 'lstm']

class SpecMulti(Spec):
  """Abstract class for a specification of a sequence-to-sequence RNN model.

  Concrete sublcasses must implement the following methods:
  - self.create_vars(): called by __init__, should initialize parameters.
  - self.get_local_params(): Get all local parameters (excludes vocabulary).
  """


  def f_read_embedding_domain(self, i, domain):
    return self.in_vocabularies[domain].get_theano_embedding(i)

  def f_write_embedding_domain(self, i, domain):
    return self.out_vocabularies[domain].get_theano_embedding(i)

  def get_params(self):
    """Get all parameters (things we optimize with respect to)."""
    params = (self.get_local_params()
              + self.in_vocabulary.get_theano_params())  # shared encoder word embeddings
    for domain in self.in_vocabularies:  # domain specific word embeddings
      params = params + self.in_vocabularies[domain].get_theano_params()
    for domain in self.out_vocabularies:  # domain specific word embeddings
      params = params + self.out_vocabularies[domain].get_theano_params()
    return params

  def get_all_shared(self):
    """Get all shared theano varaibles.

    There are shared variables that we do not necessarily optimize respect to,
    but may be held fixed (e.g. GloVe vectors, if desired).
    We don't backpropagate through these, but we do need to feed them to scan.
    """
    params = (self.get_local_params()
              + self.in_vocabulary.get_theano_params())  # shared encoder word embeddings
    for domain in self.in_vocabularies:  # domain specific word embeddings
      params = params + self.in_vocabularies[domain].get_theano_params()
    for domain in self.out_vocabularies:  # domain specific word embeddings
      params = params + self.out_vocabularies[domain].get_theano_params()
    return params
