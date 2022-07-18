# This Pythob class provides fundamental computational funtionality for the
# implementation of the encoder. In particular, the decoder has almost the same
# structure as the encoder, except that there exists an additional dense layer
# that converts the vector of size decoder_dim which is the output from the RNN
# layer, into a vector that represents the probability distribution across the
# the target vocabulary. Mind that the decoder returns outputs along all its
# timesteps since the corresponding return_sequences parameter is set to True.

from re import X
import tensorflow as tf
import tensorflow_addons as tfa


# class Decoder(tf.keras.Model):
#     def __init__(self, vocab_size, timesteps_num, embedding_dim, decoder_dim, **kwards):
#         super(Decoder, self).__init__(**kwards)
#         self.decoder_dim = decoder_dim
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=timesteps_num)
#         self.rnn = tf.keras.layers.GRU(decoder_dim, return_sequences=True, return_state=True)
#         self.dense = tf.keras.layers.Dense(vocab_size)

#     def call(self, x, state):
#         x = self.embedding(x)
#         x, state = self.rnn(x, state)
#         x = self.dense(x)
#         return x, state

# vocab_size, embedding_dim, DECODER_DIM, batch_sz, attention_type='luong'
'''class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, DECODER_DIM, batch_sz, attention_type='luong'):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = DECODER_DIM
        self.attention_type = attention_type

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(self.dec_units, None, self.batch_sz * [max_length_input], self.attention_type)

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(batch_sz)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.fc)

    def build_rnn_cell(self, batch_sz):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnn_cell, self.attention_mechanism, attention_layer_size=self.dec_units)
        return rnn_cell

    def build_attention_mechanism(self, dec_units, memory, memory_sequence_length, attention_type="luong"):
        # ------------- #
        # typ: Which sort of attention (Bahdanau, Luong)
        # dec_units: final dimension of attention outputs
        # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
        # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

        if attention_type == "bahdanau":
            return tfa.seq2seq.BahdanauAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)
        else:
            return tfa.seq2seq.LuongAttention(units=dec_units, memory=memory, memory_sequence_length=memory_sequence_length)

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_sz, dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz * [max_length_input - 1])
        return outputs
'''