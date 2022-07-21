# Import all required Python frameworks.
from xml.sax.xmlreader import InputSource
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
from tqdm import tqdm


EPOCHS = 20
SENTENCE_PAIRS = 1000
BATCH_SIZE = 64
steps_per_epoch = SENTENCE_PAIRS // BATCH_SIZE
optimizer = tf.keras.optimizers.Adam()


def loss_function(real, pred):
    # print("REAAAL",real,"PREEEED",pred)
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
decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, DECODER_DIM, BATCH_SIZE, max_length_input, max_length_output, attention_type="luong")

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
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss / steps_per_epoch))
    print("Time taken for 1 epoch {} sec\n".format(time.time() - start))



def evaluate_sentence(sentence):
    sentence = data_preparation.preprocess_sentence(sentence)
    inputs = [data_preparation.english_tokenizer_word_index[i] for i in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_input, padding="post")
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]
    result = ""
    enc_start_state = [tf.zeros((inference_batch_size, ENCODER_DIM)), tf.zeros((inference_batch_size, ENCODER_DIM))]
    enc_out, enc_h, enc_c = encoder(inputs, enc_start_state)
    dec_h = enc_h
    dec_c = enc_c
    start_tokens = tf.fill([inference_batch_size], data_preparation.french_tokenizer_word_index["BOS"])
    end_token = data_preparation.french_tokenizer_word_index["EOS"]
    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()
    # Instantiate BasicDecoder object
    decoder_instance = tfa.seq2seq.BasicDecoder(cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc)
    # Setup Memory in decoder stack
    decoder.attention_mechanism.setup_memory(enc_out)
    # set decoder_initial_state
    decoder_initial_state = decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)
    ### Since the BasicDecoder wraps around Decoder's rnn cell only, you have to ensure that the inputs to BasicDecoder
    ### decoding step is output of embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this.
    ### You only need to get the weights of embedding layer, which can be done by decoder.embedding.variables[0] and pass this callabble to BasicDecoder's call() function
    decoder_embedding_matrix = decoder.embedding.variables[0]
    outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
    return outputs.sample_id.numpy()


def translate(sentence):
    result = evaluate_sentence(sentence)
    print("OUTPUT",result)
    result = data_preparation.french_tokenizer.sequences_to_texts(result)
    print("Input: %s" % (sentence))
    print("Predicted translation: {}".format(result))


checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate("i Run !")

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
