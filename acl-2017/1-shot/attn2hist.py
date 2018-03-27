"""A soft attention model

We use the globar attention model with input feeding
used by Luong et al. (2015).
See http://stanford.edu/~lmthang/data/papers/emnlp15_attn.pdf
"""
import itertools
import numpy
import theano
from theano import tensor as T
from theano.ifelse import ifelse
import sys

from atnhspec import Attention2HistorySpec
from derivation import Derivation
from neural2 import NeuralModel2, CLIP_THRESH, NESTEROV_MU
from vocabulary import Vocabulary

class Attention2HistoryModel(NeuralModel2):
  """An encoder-decoder RNN model."""
  def setup(self):
    self.setup_encoder()
    self.setup_decoder_step()
    self.setup_decoder_write()
    self.setup_backprop()

  @classmethod
  def get_spec_class(cls):
    return Attention2HistorySpec

  def _symb_encoder(self, x):
    """The encoder (symbolically), for decomposition."""
    def fwd_rec(x_t, h_prev, *params):
      return self.spec.f_enc_fwd(x_t, h_prev)
    def bwd_rec(x_t, h_prev, *params):
      return self.spec.f_enc_bwd(x_t, h_prev)

    fwd_states, _ = theano.scan(fwd_rec, sequences=[x],
                                outputs_info=[self.spec.get_init_fwd_state()],
                                non_sequences=self.spec.get_all_shared())
    bwd_states, _ = theano.scan(bwd_rec, sequences=[x],
                                outputs_info=[self.spec.get_init_bwd_state()],
                                non_sequences=self.spec.get_all_shared(),
                                go_backwards=True)
    enc_last_state = T.concatenate([fwd_states[-1], bwd_states[-1]])
    dec_init_state = self.spec.get_dec_init_state(enc_last_state)

    bwd_states = bwd_states[::-1]  # Reverse backward states.
    annotations = T.concatenate([fwd_states, bwd_states], axis=1)
    self._get_fwd_states = theano.function(inputs = [x], outputs=[fwd_states[-1]])
    return (dec_init_state, annotations)

  def setup_encoder(self):
    """Run the encoder.  Used at test time."""
    x = T.lvector('x_for_enc')
    z= T.lvector('z_for_enc')
    dec_init_state, annotations = self._symb_encoder(x)

    h_for_write = self.spec.decoder.get_h_for_write(dec_init_state)
    
    loc_scores = self.spec.get_local_attention_scores(h_for_write, annotations)
    loc_alpha = self.spec.get_alpha(loc_scores)
    loc_c_t = self.spec.get_local_context(loc_alpha,annotations)
    loc_write_dist = self.spec.f_write(h_for_write, loc_c_t, z,loc_scores)

    scores = self.spec.get_attention_scores(h_for_write, annotations)
    alpha = self.spec.get_alpha(scores)
    c_t = loc_c_t#self.spec.get_context(alpha,annotations)
    write_dist = self.spec.f_write(h_for_write, c_t, z, scores)
    


    #self.get_x = theano.function(
    #    inputs=[annotations], outputs=[x],on_unused_input='warn')
    self.h_for = theano.function(
        inputs=[x], outputs=[h_for_write])
    self.get_scores = theano.function(inputs=[x], outputs=[scores])
    self._encode = theano.function(
        inputs=[x], outputs=[dec_init_state, annotations])



  def setup_decoder_step(self):
    """Advance the decoder by one step.  Used at test time."""
    y_t = T.lscalar('y_t_for_dec')
    c_prev = T.vector('c_prev_for_dec')
    h_prev = T.vector('h_prev_for_dec')
    h_t = self.spec.f_dec(y_t, c_prev, h_prev)
    self._decoder_step = theano.function(inputs=[y_t, c_prev, h_prev], outputs=h_t)

  def setup_decoder_write(self):
    """Get the write distribution of the decoder.  Used at test time."""
    annotations = T.matrix('annotations_for_write')
    h_prev = T.vector('h_prev_for_write')
    z = numpy.zeros(self.domain_size)
    z[self.spec.test_domain] = 1
    h_for_write = self.spec.decoder.get_h_for_write(h_prev)
    
    
    loc_scores = self.spec.get_local_attention_scores(h_for_write, annotations)
    loc_alpha = self.spec.get_alpha(loc_scores)
    loc_c_t = self.spec.get_local_context(loc_alpha,annotations)
    loc_write_dist = self.spec.f_write(h_for_write, loc_c_t,z, loc_scores)


    scores = self.spec.get_attention_scores(h_for_write, annotations)
    alpha = self.spec.get_alpha(scores)
    c_t = loc_c_t# self.spec.get_context(alpha,annotations)
    write_dist = self.spec.f_write(h_for_write, c_t, z, scores)
    
    self._decoder_write = theano.function(inputs=[annotations, h_prev], outputs=[write_dist, c_t, alpha],on_unused_input='warn')
    
    self._loc_decoder_write = theano.function(inputs=[annotations, h_prev], outputs=[loc_write_dist, loc_c_t, loc_alpha],on_unused_input='warn')#,on_unused_input='warn')
    

  def setup_backprop(self):
    eta = T.scalar('eta_for_backprop')
    x = T.lvector('x_for_backprop')
    y = T.lvector('y_for_backprop')
    y_in_x_inds = T.lmatrix('y_in_x_inds_for_backprop')
    y_in_src_inds = T.lmatrix('y_in_src_inds_for_backprop')
    y_in_domain = T.lmatrix('y_in_domain_for_backprop')
    l2_reg = T.scalar('l2_reg_for_backprop')

    # Normal operation
    dec_init_state, annotations = self._symb_encoder(x)
    nll, p_y_seq, objective, updates  = self._setup_backprop_with(
        dec_init_state, annotations, y,  y_in_x_inds, y_in_src_inds,y_in_domain, eta, l2_reg)
    self._get_nll = theano.function(
        inputs=[x, y, y_in_x_inds,y_in_src_inds,y_in_domain], outputs=nll, on_unused_input='warn')
    self._backprop = theano.function(
        inputs=[x, y, eta, y_in_x_inds,y_in_src_inds,y_in_domain, l2_reg],
        outputs=[p_y_seq, objective],
        updates=updates,on_unused_input='warn')
    # Add distractors
    self._get_nll_distract = []
    self._backprop_distract = []
    if self.distract_num > 0:
      x_distracts = [T.lvector('x_distract_%d_for_backprop' % i) 
                     for i in range(self.distract_num)]
      all_annotations = [annotations]
      for i in range(self.distract_num):
        _, annotations_distract = self._symb_encoder(x_distracts[i])
        all_annotations.append(annotations_distract)
      annotations_with_distract = T.concatenate(all_annotations, axis=0)
      nll_d, p_y_seq_d, objective_d, updates_d = self._setup_backprop_with(
          dec_init_state, annotations_with_distract, y, y_in_x_inds,y_in_src_inds,y_in_domain, eta, l2_reg)
      self._get_nll_distract = theano.function(
          inputs=[x, y, y_in_x_inds,y_in_src_inds,y_in_domain] + x_distracts, outputs=nll_d,
          on_unused_input='warn')
      self._backprop_distract = theano.function(
          inputs=[x, y, eta, y_in_x_inds,y_in_src_inds,y_in_domain, l2_reg] + x_distracts,
          outputs=[p_y_seq_d, objective_d],
          updates=updates_d)

  def _setup_backprop_with(self, dec_init_state, annotations, y, y_in_x_inds,y_in_src_inds,y_in_domain, eta, l2_reg):
      
    def history_decoder_recurrence(y_t, cur_y_in_x_inds, cur_y_in_src_inds, cur_y_in_domain, h_prev, annotations, *params):
       
        
      orig_dom = numpy.nonzero(cur_y_in_domain)[0][0]
      
      h_for_write = self.spec.decoder.get_h_for_write(h_prev)
      loc_scores = self.spec.get_local_attention_scores(h_for_write, annotations)
      loc_alpha = self.spec.get_alpha(loc_scores)
      loc_c_t = self.spec.get_local_context(loc_alpha,annotations)
      c_t = loc_c_t
      
      domain_scores_h_t = T.nnet.sigmoid(self.spec.get_domain_scores_h_t(h_for_write))
      domain_scores_c_t = T.nnet.softmax(self.spec.get_domain_scores_c_t(c_t))
      
      pred_dom_h_t = numpy.argmax(domain_scores_h_t)
      pred_dom_c_t = numpy.argmax(domain_scores_c_t)
      z = numpy.zeros(self.spec.domain_size)
      z[pred_dom_h_t] =1

      #cur_dom_score = T.dot(domain_scores_c_t*domain_scores_h_t,cur_y_in_domain)#numpy.nonzero(cur_y_in_domain)[0][0]
      f1 = T.dot(domain_scores_c_t*domain_scores_h_t,cur_y_in_domain)#numpy.nonzero(cur_y_in_domain)[0][0]
      f2 = T.dot(domain_scores_h_t,cur_y_in_domain)#numpy.nonzero(cur_y_in_domain)[0][0]
      f3 = T.dot(domain_scores_c_t,cur_y_in_domain)#numpy.nonzero(cur_y_in_domain)[0][0]
      f4 = 1-T.nnet.relu(orig_dom-pred_dom_h_t)
      f5 = 1-T.nnet.relu(orig_dom-pred_dom_c_t)
      f6 = f4*f1
      f7= domain_scores_h_t[pred_dom_h_t]
      f8= domain_scores_h_t[pred_dom_c_t]
      f9= domain_scores_h_t[orig_dom]
      f10= domain_scores_h_t[-1]

      write_dist = self.spec.f_write(h_for_write, c_t,cur_y_in_domain, loc_scores)
      write_dist_2 = self.spec.f_write(h_for_write, c_t,z, loc_scores)

      alpha = T.nnet.sigmoid(domain_scores_h_t[pred_dom_h_t])
      base_p_y_t = write_dist[y_t]
      if self.spec.attention_copying:
        loc_copying_p_y_t = T.dot(write_dist[-cur_y_in_x_inds.shape[0]:],cur_y_in_x_inds)
        p_y_t =  base_p_y_t + loc_copying_p_y_t #+ copying_p_y_t
      else:
        p_y_t = base_p_y_t 
      p_y_t = p_y_t #+ alpha*write_dist_2[y_t]
      h_t = self.spec.f_dec(y_t, c_t, h_prev)
      return (h_t, p_y_t)
    
    hist_dec_results, _ = theano.scan(fn=history_decoder_recurrence, sequences=[y, y_in_x_inds,y_in_src_inds,y_in_domain],outputs_info=[dec_init_state, None],
            non_sequences=[annotations] + self.spec.get_all_shared())

    #p_y_seq = loc_dec_results[1] #Liang's method
    h_final = hist_dec_results[0] # our method
    domain_scores1 = T.nnet.softmax(self.spec.get_domain_scores_h_t(h_final))
    domain_scores2 = T.nnet.softmax(self.spec.get_domain_scores_h_t(dec_init_state))

    orig_dom = numpy.nonzero(y_in_domain[0])[0][0]
    pred_dom1 = numpy.argmax(domain_scores1)
    pred_dom2 = numpy.argmax(domain_scores2)
    
    orig_dom = numpy.nonzero(y_in_domain[0])[0][0]
    #domain_weight = T.dot(domain_scores,y_in_domain[0])
    p_y_seq = hist_dec_results[1] # our method
    log_p_y = T.sum(T.log(p_y_seq))
    abs_diff1 = numpy.absolute(pred_dom1 - orig_dom)
    abs_diff2 = numpy.absolute(pred_dom2 - orig_dom)
    #idxs_neg=(abs_diff<0).nonzero()
    #idxs_pos=(abs_diff>0).nonzero()
    #abs_diff=theano.tensor.set_subtensor(abs_diff[idxs_neg], -1)
    #abs_diff=theano.tensor.set_subtensor(abs_diff[idxs_pos], 1)
    #nll = - log_p_y #+ T.tanh(100*abs_diff1) + T.tanh(100*abs_diff2)
    g1 = 1/2*T.pow(abs_diff1,2) + 1/2*T.pow(abs_diff2,2)
    g2 = 1/2*T.pow(abs_diff2,2)
    g3 = 1/2*T.pow(abs_diff1,2)
    nll = - log_p_y
    #nll = -log_p_y +numpy.sign(numpy.absolute(pred_dom-orig_dom))
    #nll = -log_p_y +numpy.sign(pred_dom-orig_dom)
    
    # Add L2 regularization
    regularization = l2_reg / 2 * sum(T.sum(p**2) for p in self.params)
    objective = nll + regularization
    gradients = T.grad(objective, self.params)

    # Do the updates here
    updates = []
    if self.spec.step_rule in ('adagrad', 'rmsprop'):
      print('adagrad')
      # Adagrad updates
      for p, g, c in zip(self.params, gradients, self.grad_cache):
        grad_norm = g.norm(2)
        clipped_grad = ifelse(grad_norm >= CLIP_THRESH, 
                              g * CLIP_THRESH / grad_norm, g)
        if self.spec.step_rule == 'adagrad':
          new_c = c + clipped_grad ** 2
        else:  # rmsprop
          decay_rate = 0.9  # Use fixed decay rate of 0.9
          new_c = decay_rate * c + (1.0 - decay_rate) * clipped_grad ** 2
        new_p = p - eta * clipped_grad / T.sqrt(new_c + 1e-4)
        has_non_finite = T.any(T.isnan(new_p) + T.isinf(new_p))
        updates.append((p, ifelse(has_non_finite, p, new_p)))
        updates.append((c, ifelse(has_non_finite, c, new_c)))
    elif self.spec.step_rule == 'nesterov':
      print('nestrov')
      # Nesterov momentum
      for p, g, v in zip(self.params, gradients, self.grad_cache):
        grad_norm = g.norm(2)
        clipped_grad = ifelse(grad_norm >= CLIP_THRESH, 
                              g * CLIP_THRESH / grad_norm, g)
        new_v = NESTEROV_MU * v - eta * clipped_grad
        new_p = p - NESTEROV_MU * v + (1 + NESTEROV_MU) * new_v
        has_non_finite = (T.any(T.isnan(new_p) + T.isinf(new_p)) +
                          T.any(T.isnan(new_v) + T.isinf(new_v)))
        updates.append((p, ifelse(has_non_finite, p, new_p)))
        updates.append((v, ifelse(has_non_finite, v, new_v)))
    else:
      # Simple SGD updates
      for p, g in zip(self.params, gradients):
        grad_norm = g.norm(2)
        clipped_grad = ifelse(grad_norm >= CLIP_THRESH, 
                              g * CLIP_THRESH / grad_norm, g)
        new_p = p - eta * clipped_grad
        has_non_finite = T.any(T.isnan(new_p) + T.isinf(new_p))
        updates.append((p, ifelse(has_non_finite, p, new_p)))
    return nll, p_y_seq, objective, updates

  def decode_by_em(self,ex,max_len=100):
    h_t, annotations = self._encode(ex.x_inds)
    y_tok_seq = []
    p_y_seq = []  # Should be handy for error analysis
    p = 1
    print(ex.x_str)
    for i in range(max_len): # step 1
      write_dist, c_t, alpha = self._decoder_write(annotations, h_t)
      #write_test = write_dist[(self.spec.domain_size-1)*self.out_vocabulary.size():]
      domain_scores_h_t = T.nnet.softmax(self.spec.get_domain_scores_h_t(h_t))
      domain_scores_c_t = T.nnet.softmax(self.spec.get_domain_scores_c_t(c_t))

      #d_t = numpy.argmax(domain_scores_h_t)
      y_t = numpy.argmax(write_dist) #FIXME
      #p_y_t = domain_scores[d_t]*write_dist[y_t] #FIXME
      p_y_t = write_dist[y_t] #FIXME
      p_y_seq.append(p_y_t)
      p *= p_y_t
      if y_t == Vocabulary.END_OF_SENTENCE_INDEX:
        break
      if y_t < self.out_vocabulary.size():
      #if y_t < self.out_vocabulary.size():
        #y_t = (y_t%self.out_vocabulary.size())
        y_tok = self.out_vocabulary.get_word(y_t)
      else:
        new_ind = y_t - self.out_vocabulary.size()
        #new_ind = y_t - self.out_vocabulary.size()
        augmented_copy_toks = ex.copy_toks + [Vocabulary.END_OF_SENTENCE]
        y_tok = augmented_copy_toks[new_ind]
        y_t = self.out_vocabulary.get_index(y_tok)
      '''elif y_t>=self.out_vocabulary.size() and y_t<(self.out_vocabulary.size()+self.in_vocabulary.size()):
        new_ind = y_t -self.out_vocabulary.size()
        y_tok = self.in_vocabulary.get_word(new_ind)
        print('History >')
        print(y_tok,write_dist[y_t])
        y_t = self.out_vocabulary.get_index(y_tok)
      else:
        new_ind = y_t - (self.out_vocabulary.size()+self.in_vocabulary.size())
        augmented_copy_toks = ex.copy_toks + [Vocabulary.END_OF_SENTENCE]
        y_tok = augmented_copy_toks[new_ind]
        print('Local >')
        print(y_tok,write_dist[y_t])
        y_t = self.out_vocabulary.get_index(y_tok)'''
      y_tok_seq.append(y_tok)
      h_t = self._decoder_step(y_t, c_t, h_t)
    return [Derivation(ex, p, y_tok_seq)]


  def decode_greedy(self, ex, max_len=100):
    h_t, annotations = self._encode(ex.x_inds)
    y_tok_seq = []
    p_y_seq = []  # Should be handy for error analysis
    p = 1
    for i in range(100):#max_len):
      write_dist, c_t, alpha = self._loc_decoder_write(annotations, h_t)
      for j in range(len(write_dist)):
          y_t = j
          x_t = 0
          if y_t < self.out_vocabulary.size():
            y_tok = self.out_vocabulary.get_word(y_t)
          if y_t >= self.out_vocabulary.size():
            new_ind = y_t - self.out_vocabulary.size()
            augmented_copy_toks = ex.copy_toks + [Vocabulary.END_OF_SENTENCE]
            y_tok = augmented_copy_toks[new_ind]
            x_t = self.in_vocabulary.get_index(y_tok)
            if x_t in self.spec.pair_stat:
                for y_,p_xy in self.spec.pair_stat[x_t]:
                    write_dist[y_] = 1.0*write_dist[y_]+0.0*p_xy
                    #write_dist[y_] = 0.9*write_dist[y_]+0.1*p_xy
                    #write_dist[y_] = 0.8*write_dist[y_]+0.2*p_xy
                    #write_dist[y_] = 0.7*write_dist[y_]+0.3*p_xy
                    #write_dist[y_] = 0.6*write_dist[y_]+0.4*p_xy
                    #write_dist[y_] = 0.5*write_dist[y_]+0.5*p_xy
       
      y_t = numpy.argmax(write_dist)
      
      p_y_t = write_dist[y_t]
      p_y_seq.append(p_y_t)
      p *= p_y_t
      if y_t == Vocabulary.END_OF_SENTENCE_INDEX:
        break
      if y_t < self.out_vocabulary.size():
        y_tok = self.out_vocabulary.get_word(y_t)
      else:
        new_ind = y_t - self.out_vocabulary.size()
        augmented_copy_toks = ex.copy_toks + [Vocabulary.END_OF_SENTENCE]
        y_tok = augmented_copy_toks[new_ind]
        y_t = self.out_vocabulary.get_index(y_tok)

      y_tok_seq.append(y_tok)
      h_t = self._decoder_step(y_t, c_t, h_t)
    return [Derivation(ex, p, y_tok_seq)]

  def decode_beam(self, ex, beam_size=1, max_len=100):
    h_t, annotations = self._encode(ex.x_inds)
    beam = [[Derivation(ex, 1, [], hidden_state=h_t, 
                        attention_list=[], copy_list=[])]]
    finished = []
    for i in range(1, max_len):
      #print >> sys.stderr, 'decode_beam: length = %d' % i
      if len(beam[i-1]) == 0: break
      # See if beam_size-th finished deriv is best than everything on beam now.
      if len(finished) >= beam_size:
        finished_p = finished[beam_size-1].p
        cur_best_p = beam[i-1][0].p
        if cur_best_p < finished_p:
          break
      new_beam = []
      for deriv in beam[i-1]:
        cur_p = deriv.p
        h_t = deriv.hidden_state
        y_tok_seq = deriv.y_toks
        attention_list = deriv.attention_list
        copy_list = deriv.copy_list
        write_dist, c_t, alpha = self._loc_decoder_write(annotations, h_t)
        #for p_y_t,y_t in enumerate(write_dist):
        #    y_tok = self.out_vocabulary.get_word(y_t)
        #    print(y_tok)
        #    input('im here ..')
        sorted_dist = sorted([(p_y_t, y_t) for y_t, p_y_t in enumerate(write_dist)],
                             reverse=True)
        for j in range(beam_size):
          p_y_t, y_t = sorted_dist[j]
          new_p = cur_p * p_y_t
          if y_t == Vocabulary.END_OF_SENTENCE_INDEX:
            finished.append(Derivation(ex, new_p, y_tok_seq,
                                       attention_list=attention_list + [alpha],
                                       copy_list=copy_list + [0]))
            continue
          if y_t < self.out_vocabulary.size():
            y_tok = self.out_vocabulary.get_word(y_t)
            do_copy = 0
          else:
            new_ind = y_t - self.out_vocabulary.size()
            augmented_copy_toks = ex.copy_toks + [Vocabulary.END_OF_SENTENCE]
            y_tok = augmented_copy_toks[new_ind]
            y_t = self.out_vocabulary.get_index(y_tok)
            do_copy = 1
          new_h_t = self._decoder_step(y_t, c_t, h_t)
          new_entry = Derivation(ex, new_p, y_tok_seq + [y_tok],
                                 hidden_state=new_h_t,
                                 attention_list=attention_list + [alpha],
                                 copy_list=copy_list + [do_copy])
          new_beam.append(new_entry)
      new_beam.sort(key=lambda x: x.p, reverse=True)
      beam.append(new_beam[:beam_size])
      finished.sort(key=lambda x: x.p, reverse=True)
    return sorted(finished, key=lambda x: x.p, reverse=True)
