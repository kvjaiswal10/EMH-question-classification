import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Lambda, Dense, Concatenate, Bidirectional
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# Load your dataset
data = pd.read_csv('data\dataset - MANUAL FINAL.csv') 

paragraphs = data['Document'].tolist()
questions = data['Question'].tolist()
labels = data['Difficulty'].str.lower().tolist()

# Hyperparameters
vocab_size = 50000
max_length = 40
embedding_dim = 100
hiddendim = 128

# Tokenizers
tokenizer_paragraph = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer_question = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

# Fit tokenizers on text
tokenizer_paragraph.fit_on_texts(paragraphs)
tokenizer_question.fit_on_texts(questions)

# Convert texts to sequences
sequences_paragraph = tokenizer_paragraph.texts_to_sequences(paragraphs)
sequences_question = tokenizer_question.texts_to_sequences(questions)

# Pad sequences
padded_paragraph = pad_sequences(sequences_paragraph, maxlen=max_length, padding='post', truncating='post')
padded_question = pad_sequences(sequences_question, maxlen=max_length, padding='post', truncating='post')

label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = tf.keras.utils.to_categorical(integer_encoded_labels)


(train_paragraph, val_paragraph, 
 train_question, val_question, 
 train_labels, val_labels) = train_test_split(
    padded_paragraph, padded_question, categorical_labels, test_size=0.2, random_state=42)


# Load GloVe embeddings
embedding_index = {}
with open('glove.6B.100d.txt', encoding='utf-8') as f:  # Ensure you have downloaded GloVe embeddings
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
word_index = tokenizer_paragraph.word_index
for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector



# Parameters
maxsentence_length = max_length
embedding_dim = 100
max_num_words = 50000
hiddendim = 128   

def manhattandistance(l1, l2):
    return K.exp(-K.sum(K.abs(l1 - l2), axis=1, keepdims=True))

def siamese_manhattan_network(embed_mat_weights):

    ques1 = Input(shape=(maxsentence_length,))
    ques2 = Input(shape=(maxsentence_length,))

    embedding_layer = Embedding(input_dim=max_num_words, output_dim=embedding_dim, weights=[embed_mat_weights],
                                trainable=False, input_length=maxsentence_length)

    ques1_embed = embedding_layer(ques1)
    ques2_embed = embedding_layer(ques2)

    lstm = LSTM(hiddendim, return_sequences=False)

    ques1_lstm_out = lstm(ques1_embed)
    ques2_lstm_out = lstm(ques2_embed)

    manhattan_dis = Lambda(lambda x: manhattandistance(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([ques1_lstm_out, ques2_lstm_out])

    # Fully connected layers for classification
    dense_1 = Dense(128, activation='relu')(manhattan_dis)
    output = Dense(categorical_labels.shape[1], activation='softmax')(dense_1)

    model = Model(inputs=[ques1, ques2], outputs=output)

    optimizer = Adam(clipnorm=1.25)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = siamese_manhattan_network(embed_mat_weights=embedding_matrix)
model.summary()

epochs = 10
batchsize = 128

start = time.perf_counter()

history = model.fit([train_paragraph, train_question], train_labels,
                    validation_data=([val_paragraph, val_question], val_labels),
                    batch_size=batchsize, epochs=epochs, verbose=1)

elapsed = time.perf_counter() - start
print(f'Elapsed {elapsed:.3f} seconds.')

loss, accuracy = model.evaluate([val_paragraph, val_question], val_labels, verbose=1)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# Make predictions on the validation data
predictions = model.predict([val_paragraph, val_question])

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)
decoded_predictions = label_encoder.inverse_transform(predicted_classes)
print(decoded_predictions)


