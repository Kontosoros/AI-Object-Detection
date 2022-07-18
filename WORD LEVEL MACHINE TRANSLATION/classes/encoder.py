# This Pythob class provides fundamental computational funtionality for the 
# implementation of the encoder. In particular, the encoder will be composed of
# an embedding layer followed by a GRU layer. The input to the encoder is a 
# sequence of integers, which is converted to a sequence of embedding vectors
# of size embedding_dim. This sequence of vectors is subsequently sent to an
# RNN, which converts the input at each of the timesteps_num timesteps to a 
# vector of size encode_dim. Mind that only the output at the last time step
# is actually returned since the return_sequences parameter is set to False.


import tensorflow as tf

# class Encoder(tf.keras.Model):
    
#     def __init__(self,vocab_size,timesteps_num,embedding_dim,encoder_dim,**kwards):
#         super(Encoder,self).__init__(**kwards)
#         self.encoder_dim = encoder_dim
#         self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim,
#                                                    input_length=timesteps_num)
#         self.rnn = tf.keras.layers.GRU(encoder_dim,return_sequences=False,
#                                        return_state=True)
                    
    
#     def call(self,x,state):
#         x = self.embedding(x)
#         x, state = self.rnn(x,initial_state=state)
#         return x, state
    
#     def init_state(self,batch_size):
#         return tf.zeros((batch_size,self.encoder_dim))


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    ##-------- LSTM layer in Encoder ------- ##
    self.lstm_layer = tf.keras.layers.LSTM(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')



  def call(self, x, hidden):
    x = self.embedding(x)
    output, h, c = self.lstm_layer(x, initial_state = hidden)
    return output, h, c

  def initialize_hidden_state(self):
    return [tf.zeros((self.batch_sz, self.enc_units)), tf.zeros((self.batch_sz, self.enc_units))]
