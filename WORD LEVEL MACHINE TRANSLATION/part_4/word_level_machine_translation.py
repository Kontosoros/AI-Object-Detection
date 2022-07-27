# Import all required Python frameworks.
from torch import batch_norm_backward_reduce
from classes.data_preparation import DataPreparation
from classes.transformer import Transformer, create_padding_mask, create_look_ahead_mask
import tensorflow as tf
import time
from classes.optimizer import CustomSchedule
import numpy
from tqdm import tqdm
EPOCHS = 2
SENTENCE_PAIRS = 1000
BATCH_SIZE = 64
steps_per_epoch = SENTENCE_PAIRS // BATCH_SIZE
DATAPATH = "datasets"
DATAFILE = "fra.txt"
NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred,flag = True):
    #tf.print("real:",real[0].shape,"pred :",tf.argmax(pred, axis=2, output_type="int32")[0].shape)
    accuracies = tf.equal(real, tf.argmax(pred, axis=2, output_type="int32") if flag == True else pred )
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


data_preparation = DataPreparation(DATAPATH, DATAFILE, SENTENCE_PAIRS, BATCH_SIZE)
train_dataset = data_preparation.train_dataset

example_input_batch, example_target_batch = next(iter(train_dataset))
example_input_batch.shape, example_target_batch.shape
max_length_input = example_input_batch.shape[1]
max_length_output = example_target_batch.shape[1]-1
print("max_length_output", max_length_output)
vocab_inp_size = data_preparation.english_vocabulary_size + 1
vocab_tar_size = data_preparation.french_vocabulary_size + 1


learning_rate = CustomSchedule(d_model=D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
transformer = Transformer(
    num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF, input_vocab_size=vocab_inp_size, target_vocab_size=vocab_tar_size, rate=DROPOUT_RATE
)

checkpoint_path = "./checkpoints/"
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Latest checkpoint restored!!")


@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    # tf.print("tar_inp :" , tar_inp[0])
    # tf.print("tar_real :" , tar_real[0])
    # tf.print("len tar_real :" , len(tar_real[0]))
    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp], training=True)
        loss = loss_function(tar_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions,flag =True))


for epoch in range(EPOCHS):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    for (batch, (inp, tar)) in enumerate(train_dataset.take(steps_per_epoch)):
        train_step(inp, tar)
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        # print(f"Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}")

    print(f"Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}")
    # print(f"Time taken for 1 epoch: {time.time() - start:.2f} secs\n")


def test(en_input):
    encoder_input = en_input
    decoder_input = [3]
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, decoder_input)
    for i in range(1,max_length_output):
        output = tf.transpose(output_array.stack())
        predictions, _ = transformer([encoder_input, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        # tf.print("predicted_id:" ,predicted_id)
        output_array = output_array.write(i + 1, predicted_id[0])
    output = tf.transpose(output_array.stack())
    return output.numpy().tolist()[0]


predicted_list = []
input_english = data_preparation.input_tensor_val
target_fr = numpy.array(data_preparation.target_tensor_val)
for en_input in tqdm(input_english, desc ="TESTING PROCESS !!!"):
    en_input = numpy.array([en_input])
    r = test(en_input)
    predicted_list.append(r)
predicted_list = tf.convert_to_tensor(predicted_list)
target_fr = tf.convert_to_tensor(target_fr)
# print("predicted_list :",predicted_list)
# print("predicted_list",predicted_list.shape)
# # print("target_fr :",target_fr)
# print("target_fr",target_fr.shape)

score = accuracy_function(target_fr, predicted_list,flag =False)
print("score",score)

def evaluate(inp_sentence):
    sentence = data_preparation.preprocess_sentence(inp_sentence)
    inputs = [data_preparation.english_tokenizer_word_index[i] for i in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_input, padding="post")
    encoder_input = inputs
    decoder_input = [3]
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, decoder_input)
    for i in range(1,max_length_output):
        output = tf.transpose(output_array.stack())
        predictions, _ = transformer([encoder_input, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        # tf.print("predicted_id:" ,predicted_id)
        output_array = output_array.write(i + 1, predicted_id[0])
        if predicted_id[0][0] == 1:
            break
    output = tf.transpose(output_array.stack())
    #print(output.numpy().tolist()[0])
    return output.numpy().tolist()[0]


def translate(sentence):
    result = evaluate(sentence)
    lista = []
    for i in result:
        if i in list(data_preparation.french_idx2word.keys()) and i < vocab_tar_size:
            lista.append(data_preparation.french_idx2word[i])
    print("Input: {}".format(sentence))
    print("Predicted translation: {}".format(lista))


english_sent = ["run !"]
for sent in english_sent:
    translate(sent)
