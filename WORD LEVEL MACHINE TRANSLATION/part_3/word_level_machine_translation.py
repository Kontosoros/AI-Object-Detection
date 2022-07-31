# Import all required Python frameworks.
from classes.data_preparation import DataPreparation
from classes.encoder import Encoder
from classes.decoder import Decoder
import tensorflow as tf
import os
import tensorflow_addons as tfa
from tqdm import tqdm
import os
import numpy
import os
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.config.list_physical_devices("GPU")
if tf.config.list_physical_devices("GPU"):
    print("The model will use the GPU !!!")
else:
    print("The model will use the CPU !!!")

EPOCHS = 1
SENTENCE_PAIRS = 4000
BATCH_SIZE = 64
DATAPATH = "datasets"
DATAFILE = "fra.txt"


EMBEDDING_DIM = 256
ENCODER_DIM, DECODER_DIM = 1024, 1024
STEPS_PER_EPOCH = SENTENCE_PAIRS // BATCH_SIZE
optimizer = tf.keras.optimizers.Adam()
train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
test_accuracy = train_accuracy
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

checkpoint_path = "./training_checkpoints/"
ckpt = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")


def loss_function(real, pred):
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.logical_not(tf.math.equal(real, 0))  # output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask * loss
    loss = tf.reduce_mean(loss)
    return loss


def accuracy_function(real, pred, flag=True):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type="int32") if flag == True else pred)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_h, enc_c = encoder(inp, enc_hidden)
        dec_input = targ[:, :-1]  # Ignore EOS token
        real = targ[:, 1:]  # ignore BOS token
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
    train_accuracy(accuracy_function(real, logits, flag=True))
    return loss

print(f"==================== TRAINING PROCESS (EPOCHS= {EPOCHS} , NUMBER OF SENTENCES FOR TRAINING = {int(SENTENCE_PAIRS * 0.8)}) ====================")
for epoch in range(EPOCHS):
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0
    train_accuracy.reset_states()
    for (batch, (inp, targ)) in enumerate(train_dataset.take(STEPS_PER_EPOCH)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
    print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss / STEPS_PER_EPOCH), f"Accuracy {train_accuracy.result():.4f}")


def test_or_infer(en_input,infer_flag=False):
    if infer_flag == True:
        sentence = data_preparation.preprocess_sentence(en_input)
        inputs = [data_preparation.english_tokenizer_word_index[i] for i in sentence.split(" ")]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_input, padding="post")
        en_input = tf.convert_to_tensor(inputs)
    inference_batch_size = en_input.shape[0]
    enc_start_state = [tf.zeros((inference_batch_size, ENCODER_DIM)), tf.zeros((inference_batch_size, ENCODER_DIM))]
    enc_out, enc_h, enc_c = encoder(en_input, enc_start_state)
    start_tokens = tf.fill([inference_batch_size], data_preparation.french_tokenizer_word_index["BOS"])
    end_token = data_preparation.french_tokenizer_word_index["EOS"]
    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()
    decoder_instance = tfa.seq2seq.BasicDecoder(cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc)
    decoder.attention_mechanism.setup_memory(enc_out)
    decoder_initial_state = decoder.build_initial_state(inference_batch_size, [enc_h, enc_c], tf.float32)
    decoder_embedding_matrix = decoder.embedding.variables[0]
    outputs, _, _ = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
    return outputs.sample_id.numpy()[0] if infer_flag == False else outputs.sample_id.numpy()

# print(f"==================== TEST PROCESS ====================")
# predicted_list = []
# input_english = data_preparation.input_tensor_val
# target_fr = numpy.array(data_preparation.target_tensor_val)
# for en_input in tqdm(input_english, desc="TESTING PROCESS !!!"):
#     en_input = numpy.array([en_input])
#     r = test_or_infer(en_input,infer_flag=False)
#     predicted_list.append(r)
# padding_predicted_list = tf.keras.preprocessing.sequence.pad_sequences(predicted_list, padding="post", maxlen=max_length_output)
# target_fr = tf.convert_to_tensor(target_fr)
# score = test_accuracy(accuracy_function(target_fr, padding_predicted_list, flag=False))
# tf.print(f"TEST ACCURACY : {score:.4f}")
# for i , j in zip(list(target_fr[:10]),padding_predicted_list[:10]):
#     tf.print("TARGET :",list(i) ,"PREDICT :",list(j) )


def translate(sentence):
    result = test_or_infer(sentence, infer_flag = True)
    print("OUTPUT", result)
    result = data_preparation.french_tokenizer.sequences_to_texts(result)
    print("Input: %s" % (sentence))
    print("Infer translation: {}".format(result))

translate("Go .")