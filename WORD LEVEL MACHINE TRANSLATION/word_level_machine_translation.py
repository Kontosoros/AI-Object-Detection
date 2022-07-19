# This is the main script file facilitating all the available functionality for
# the machine translation task at the word level.
from tensorflow.keras.models import Model

# Import all required Python frameworks.
from classes.data_preparation import DataPreparation
# from classes.encoder import Encoder

# from classes.decoder import Decoder
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os
import tensorflow_addons as tfa
import time


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




class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, DECODER_DIM, batch_sz, attention_type="luong"):
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
        outputs, _, _ = self.decoder(x, initial_state=initial_state, sequence_length=self.batch_sz * [max_length_output - 1])
        return outputs


# -----------------------------------------------------------------------------
# This function defines the loss function to be minimized during training.
# Given the fact that input and target sequences for both langauges have been
# padded in order to reflect the maximum sequence length for the respective
# language, it is imperative to avoid considering equality of pad words between
# the true labels and the estimated predictions. To this end, the utilized loos
# function masks the estimated predictions with the true labels, so that padded
# positions on the label are also removed from the predictions. Thus, the loss
# function is actually evaluated exclusively on the non-zero elements of both
# labels and predictions.
optimizer = tf.keras.optimizers.Adam()
def loss_function(real, pred):
    #print("REAAAL",real,"PREEEED",pred)
    # real shape = (BATCH_SIZE, max_length_output)
    # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.logical_not(tf.math.equal(real, 0))  # output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask * loss
    loss = tf.reduce_mean(loss)
    return loss


DATAPATH = "datasets"

DATAFILE = "fra.txt"

SENTENCE_PAIRS = 15000

BATCH_SIZE = 64

CHECKPOINT_DIRECTORY = "checkpoints"
data_preparation = DataPreparation(DATAPATH, DATAFILE, SENTENCE_PAIRS, BATCH_SIZE)
EMBEDDING_DIM = 256

ENCODER_DIM, DECODER_DIM = 1024, 1024
train_dataset = data_preparation.train_dataset
example_input_batch, example_target_batch = next(iter(train_dataset))
example_input_batch.shape, example_target_batch.shape
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]
vocab_inp_size = data_preparation.english_vocabulary_size + 1
vocab_tar_size = data_preparation.french_vocabulary_size + 1

encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, ENCODER_DIM, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, DECODER_DIM, BATCH_SIZE, attention_type="luong")

checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_h, enc_c = encoder(inp, enc_hidden)
        dec_input = targ[:, :-1]  # Ignore <end> token
        real = targ[:, 1:]  # ignore <start> token
        # Set the AttentionMechanism object with encoder_outputs
        decoder.attention_mechanism.setup_memory(enc_output)

        # Create AttentionWrapperState as initial_state for decoder
        decoder_initial_state = decoder.build_initial_state(BATCH_SIZE, [enc_h, enc_c], tf.float32)
        pred = decoder(dec_input, decoder_initial_state)
        logits = pred.rnn_output
        loss = loss_function(real, logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


EPOCHS = 10


for epoch in range(EPOCHS):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(train_dataset.take(SENTENCE_PAIRS)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
        if batch % 100 == 0:
            print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch, batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss / SENTENCE_PAIRS))
    print("Time taken for 1 epoch {} sec\n".format(time.time() - start))


# Visualize training history for BLEU Score.
# plt.plot(eval_scores)
# plt.title("Model Accuracy in terms of the BLEU Score.")
# plt.ylabel("BLEU Score")
# plt.xlabel("epoch")
# plt.legend(["BLEU Score"], loc="lower right")
# plt.grid()
# plt.show()
# # Visualize training history for Loss.
# # plt.plot(losses)
# plt.title("Model Accuracy in terms of the Loss.")
# plt.ylabel("Loss")
# plt.xlabel("epoch")
# plt.legend(["LOSS"], loc="upper right")
# plt.grid()
# plt.show()
