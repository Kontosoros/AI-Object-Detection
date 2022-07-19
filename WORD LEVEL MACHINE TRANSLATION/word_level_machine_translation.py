# Import all required Python frameworks.
from classes.data_preparation import DataPreparation
from classes.encoder import Encoder
from classes.decoder import Decoder
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import os
import tensorflow_addons as tfa
import time

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
SENTENCE_PAIRS = 500
BATCH_SIZE = 64
CHECKPOINT_DIRECTORY = "checkpoints"
EMBEDDING_DIM = 256
ENCODER_DIM, DECODER_DIM = 1024, 1024

data_preparation = DataPreparation(DATAPATH, DATAFILE, SENTENCE_PAIRS, BATCH_SIZE)
train_dataset = data_preparation.train_dataset
example_input_batch, example_target_batch = next(iter(train_dataset))
example_input_batch.shape, example_target_batch.shape
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]
vocab_inp_size = data_preparation.english_vocabulary_size + 1
vocab_tar_size = data_preparation.french_vocabulary_size + 1

encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, ENCODER_DIM, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, DECODER_DIM, BATCH_SIZE,max_length_input,max_length_output, attention_type="luong")

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

steps_per_epoch = SENTENCE_PAIRS // BATCH_SIZE
print("STEPS PER EPOCH :" ,steps_per_epoch)
for epoch in range(EPOCHS):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(train_dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
        # if batch % 100 == 0:
        #     print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch, batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    # if (epoch + 1) % 2 == 0:
    #     checkpoint.save(file_prefix=checkpoint_prefix)

    print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss / steps_per_epoch))
    #print("Time taken for 1 epoch {} sec\n".format(time.time() - start))


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
