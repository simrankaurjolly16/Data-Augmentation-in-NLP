import inquirer
import tensorflow as tf
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description='Model prediction')
parser.add_argument('-path','--root_path',required=True)
args = parser.parse_args()
root_path = args.root_path


questions = [
  inquirer.List("options", message="What size do you need?", choices=['1.English To Punjabi', '2.Punjabi To English'],),]
answers = inquirer.prompt(questions)

def read_dataset(filepath):
    with open(filepath, 'rb') as fp:
        return pickle.load(fp)

def get_file_path(options,root_path):
    # root_path = "/home/whirldata/Downloads/Colab_NLP/Saved_Path/"
    if options["options"].__contains__("2"):
        external_path = root_path+"Punjabi_to_English/"
        return(external_path+"data.p",external_path+"checkpoints/")
    else:
        external_path = root_path+"English_to_Punjabi/"
        return(external_path+"data.p",external_path+"checkpoints/")

pickle_path, checkpoint_path = get_file_path(answers,root_path)

# read dataset
dataset_location = pickle_path
X, Y, l1_word2idx, l1_idx2word, l1_vocab, l2_word2idx, l2_idx2word, l2_vocab = read_dataset(dataset_location)
input_sentence = input("Enter the Sentence : ")
l1_sentences = []
l1_sentences.append(input_sentence)
x = [[l1_word2idx.get(word.strip(',." ;:)(|][?!<>'), 0) for word in sentence.split()] for sentence in l1_sentences]


input_seq_len = 20
output_seq_len = 22
l1_vocab_size = len(l1_vocab) + 2 # + <pad>, <ukn>
l2_vocab_size = len(l2_vocab) + 4 # + <pad>, <ukn>, <eos>, <go>


def data_padding(x, length = 20):
    for i in range(len(x)):
        x[i] = x[i] + (length - len(x[i])) * [l1_word2idx['<pad>']]
    return x

l1_test_data = data_padding(x)

def decode_sentence_string(sentences, idx2word):
    result = []
    for sentence in sentences:
        temp = """"""
        for i in range(len(sentence)):
            if sentence[i] not in [1, 2, 3] and sentence[i] in idx2word :
                temp += idx2word[sentence[i]] +" "
        result.append(temp)
    return result


# simple softmax function
def softmax(x):
    n = np.max(x)
    e_x = np.exp(x - n)
    return e_x / e_x.sum()


def decode_output(output_seq):
    words = []
    for i in range(output_seq_len):
        smax = softmax(output_seq[i])
        idx = np.argmax(smax)
        words.append(l2_idx2word[idx])
    return words


# For whole test data

with tf.Graph().as_default():
    
    # placeholders
    encoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'encoder{}'.format(i)) for i in range(input_seq_len)]
    decoder_inputs = [tf.placeholder(dtype = tf.int32, shape = [None], name = 'decoder{}'.format(i)) for i in range(output_seq_len)]

    # output projection
    size = 512
    w_t = tf.get_variable('proj_w', [l2_vocab_size, size], tf.float32)
    b = tf.get_variable('proj_b', [l2_vocab_size], tf.float32)
    w = tf.transpose(w_t)
    output_projection = (w, b)
    
    
    # change the model so that output at time t can be fed as input at time t+1
    outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                                                encoder_inputs,
                                                decoder_inputs,
                                                tf.nn.rnn_cell.BasicLSTMCell(size),
                                                num_encoder_symbols = l1_vocab_size,
                                                num_decoder_symbols = l2_vocab_size,
                                                embedding_size = 80,
                                                feed_previous = True, # <-----this is changed----->
                                                output_projection = output_projection,
                                                dtype = tf.float32)
    
    # ops for projecting outputs
    outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(output_seq_len)]
    
    l1_sentences_encoded = l1_test_data
    
    l1_sentences = decode_sentence_string(l1_sentences_encoded, l1_idx2word)
        
    # restore all variables - use the last checkpoint saved
    saver = tf.train.Saver()
    
    path = tf.train.latest_checkpoint(checkpoint_path)
    
    with tf.Session() as sess:
            # restore
            saver.restore(sess, path)

            # feed data into placeholders
            feed = {}
            for i in range(input_seq_len):
                feed[encoder_inputs[i].name] = np.array([l1_sentences_encoded[j][i] for j in range(len(l1_sentences_encoded))])
            
            feed[decoder_inputs[0].name] = np.array([l2_word2idx['<go>']] * len(l1_sentences_encoded))

            # translate
            output_sequences = sess.run(outputs_proj, feed_dict = feed)
            
            words = decode_output(output_sequences)
            
            ouput_seq = [output_sequences[j] for j in range(output_seq_len)]
            
            print("Actual ==============================================>>>>>>",l1_sentences)
            predicted_words = []
            for i in range(len(words)):
                    if words[i] not in ['<eos>', '<pad>', '<go>']:
                        predicted_words.append(words[i])
            print("Prediction ==========================================>>>>>"," ".join(predicted_words))
            
