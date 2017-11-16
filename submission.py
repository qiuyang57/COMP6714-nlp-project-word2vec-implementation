## Submission.py for COMP6714-Project2
###################################################################################################################
import codecs
import math
import random
import zipfile
import numpy as np
import tensorflow as tf
import spacy
import re
import collections
import gensim
import pickle

vocabulary_size = 9000  # This variable is used to define the maximum vocabulary size. 15000 best
data_index = 0


def build_dataset(words, n_words):
    """Process raw inputs into a dataset.
       words: a list of words, i.e., the input data
       n_words: Vocab_size to limit the size of the vocabulary. Other words will be mapped to 'UNK'
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # i.e., one of the 'UNK' words
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def adjective_embeddings(data_file, embeddings_file_name, num_steps=100001, embedding_dim=200):
    with open(data_file, 'rb') as f:
        data_list = pickle.load(f)

    data, count, dictionary, reverse_dictionary = build_dataset(data_list, vocabulary_size)
    # Specification of Training data:
    batch_size = 128  # Size of mini-batch for skip-gram model.
    skip_window = 2  # How many words to consider left and right of the target word.
    num_samples = 4  # How many times to reuse an input to generate a label.
    num_sampled = 500  # Sample size for negative examples. 64
    logs_path = './log/'

    # Specification of test Sample:
    sample_size = 20  # Random sample of words to evaluate similarity.
    sample_window = 100  # Only pick samples in the head of the distribution.
    sample_examples = np.random.choice(sample_window, sample_size, replace=False)  # Randomly pick a sample of size 16

    ## Constructing the graph...
    graph = tf.Graph()

    with graph.as_default():

        with tf.device('/cpu:0'):
            # Placeholders to read input data.
            with tf.name_scope('Inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            # Look up embeddings for inputs.
            with tf.name_scope('Embeddings'):
                sample_dataset = tf.constant(sample_examples, dtype=tf.int32)
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dim], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

                # Construct the variables for the NCE loss
                weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_dim],
                                                          stddev=1.0 / math.sqrt(embedding_dim)))
                biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            with tf.name_scope('Loss'):
                loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=weights, biases=biases,
                                                                 labels=train_labels, inputs=embed,
                                                                 num_sampled=num_sampled, num_classes=vocabulary_size))

            with tf.name_scope('Adam'):
                optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)

            # with tf.name_scope('Gradient_Descent'):
            #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

            # Normalize the embeddings to avoid overfitting.
            with tf.name_scope('Normalization'):
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                normalized_embeddings = embeddings / norm

            sample_embeddings = tf.nn.embedding_lookup(normalized_embeddings, sample_dataset)
            similarity = tf.matmul(sample_embeddings, normalized_embeddings, transpose_b=True)

            # Add variable initializer.
            init = tf.global_variables_initializer()

            # Create a summary to monitor cost tensor
            tf.summary.scalar("cost", loss)
            # Merge all summary variables.
            merged_summary_op = tf.summary.merge_all()

    # the variable is abused in this implementation.
    # Outside the sample generation loop, it is the position of the sliding window: from data_index to data_index + span
    # Inside the sample generation loop, it is the next word to be added to a size-limited buffer.

    def generate_batch(batch_size, num_samples, skip_window):
        global data_index

        assert batch_size % num_samples == 0
        assert num_samples <= 2 * skip_window

        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # span is the width of the sliding window
        buffer = collections.deque(maxlen=span)
        if data_index + span > len(data):
            data_index = 0
        buffer.extend(data[data_index:data_index + span])  # initial buffer content = first sliding window

        data_index += span
        for i in range(batch_size // num_samples):
            context_words = [w for w in range(span) if w != skip_window]

            random.shuffle(context_words)
            words_to_use = collections.deque(context_words)  # now we obtain a random list of context words
            for j in range(num_samples):  # generate the training pairs
                batch[i * num_samples + j] = buffer[skip_window]
                context_word = words_to_use.pop()
                labels[i * num_samples + j, 0] = buffer[context_word]  # buffer[context_word] is a random context word

            # slide the window to the next position
            if data_index == len(data):
                buffer = data[:span]
                data_index = span
            else:
                buffer.append(data[
                                  data_index])  # note that due to the size limit, the left most word is automatically removed from the buffer.
                data_index += 1

        # end-of-for
        data_index = (data_index + len(data) - span) % len(data)  # move data_index back by `span`
        return batch, labels

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        session.run(init)
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        print('Initializing the model')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_samples, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op using session.run()
            _, loss_val, summary = session.run([optimizer, loss, merged_summary_op], feed_dict=feed_dict)

            summary_writer.add_summary(summary, step)
            average_loss += loss_val

            if step % 5000 == 0:
                if step > 0:
                    average_loss /= 5000

                    # The average loss is an estimate of the loss over the last 5000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

            # Evaluate similarity after every 10000 iterations.
            if step % 10000 == 0:
                sim = similarity.eval()  #
                for i in range(sample_size):
                    sample_word = reverse_dictionary[sample_examples[i]]
                    top_k = 10  # Look for top-10 neighbours for words in sample set.
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % sample_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
                print()

        final_embeddings = normalized_embeddings.eval()
        out = ''
        num = 0
        for i in range(1, len(reverse_dictionary)):
            word = reverse_dictionary[i]
            if '|ADJ' == word[-4:]:
                out += word + ' '
                num += 1
                for j, vector in enumerate(final_embeddings[i]):
                    if j != len(final_embeddings[i]) - 1:
                        out += '{} '.format(vector)
                    else:
                        out += '{}\n'.format(vector)
        out = '{} {}\n'.format(num - 1, int(embedding_dim)) + out
        with codecs.open(embeddings_file_name, 'w', encoding='utf8') as f:
            f.write(out)


LABELS = {
    'ENT': 'ENT',
    'PERSON': 'ENT',
    'NORP': 'ENT',
    'FAC': 'ENT',
    'ORG': 'ENT',
    'GPE': 'ENT',
    'LOC': 'ENT',
    'LAW': 'ENT',
    'PRODUCT': 'ENT',
    'EVENT': 'ENT',
    'WORK_OF_ART': 'ENT',
    'LANGUAGE': 'ENT',
    'DATE': 'DATE',
    'TIME': 'TIME',
    'PERCENT': 'PERCENT',
    'MONEY': 'MONEY',
    'QUANTITY': 'QUANTITY',
    'ORDINAL': 'ORDINAL',
    'CARDINAL': 'CARDINAL'
}

pre_format_re = re.compile(r'^[\`\*\~]')
post_format_re = re.compile(r'[\`\*\~]$')
url_re = re.compile(r'\[([^]]+)\]\(%%URL\)')
link_re = re.compile(r'\[([^]]+)\]\(https?://[^\)]+\)')


def strip_meta(text):
    text = link_re.sub(r'\1', text)
    text = text.replace('&gt;', '>').replace('&lt;', '<')
    text = pre_format_re.sub('', text)
    text = post_format_re.sub('', text)
    return text


def transform_doc(doc):
    for ent in doc.ents:
        ent.merge(tag=ent.root.tag_, lemma=ent.text, ent_type=LABELS[ent.label_])
    strings = []
    for sent in doc.sents:
        if sent.text.strip():
            strings.append(' '.join(represent_word(w) for w in sent if not w.is_space and not w.pos_ == 'PUNCT'))
    if strings:
        return '\n'.join(strings) + '\n'
    else:
        return ''


def represent_word(word):
    if word.like_url:
        return '%%URL|X'
    tag = LABELS.get(word.ent_type_, word.pos_)
    if tag == 'ADJ':
        text = re.sub(r'\s', '_', word.text.lower())
    else:
        text = re.sub(r'\s', '_', word.lemma_.lower())

    if not tag:
        tag = '?'
    if tag in LABELS or tag == 'NUM' or tag == 'SYM':
        return tag
    else:
        return text + '|' + tag


def process_data(input_data):
    data_name = 'data_file'
    nlp = spacy.load('en')
    data = []
    with zipfile.ZipFile(input_data) as f:
        i = 0
        for name in f.namelist():
            if i % 100 == 0: print(i)
            article = tf.compat.as_str(f.read(name))
            if article != '':
                new_article = transform_doc(nlp(strip_meta(article)))
                data.extend(new_article.split())
            i += 1
    with open(data_name, 'wb') as f:
        pickle.dump(data, f)
    return data_name


def Compute_topk(model_file, input_adjective, top_k):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    output = []
    for a, b in model.most_similar(positive=[input_adjective + '|ADJ'], topn=top_k):
        output.append(a[:-4])
    return output
