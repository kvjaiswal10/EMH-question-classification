import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Conv1D,LSTM,Bidirectional,Embedding,GlobalMaxPooling1D,Dropout,Flatten,MaxPool1D,MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Concatenate, Conv1D, MaxPooling1D, Flatten


data = pd.read_csv('data\dataset - MANUAL FINAL.csv')  

paragraphs = data['Paragraph'].tolist()
questions = data['Question'].tolist()
answers = data['Answer'].tolist()
labels = data['Difficulty'].str.lower().tolist()
print(labels)

vocab_size=10000
embedding_dim=128
max_length=400


# Tokenizers
tokenizer_paragraph = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer_question = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer_answer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")

# Fit tokenizers on text
tokenizer_paragraph.fit_on_texts(paragraphs)
tokenizer_question.fit_on_texts(questions)
tokenizer_answer.fit_on_texts(answers)

# Convert texts to sequences
sequences_paragraph = tokenizer_paragraph.texts_to_sequences(paragraphs)
sequences_question = tokenizer_question.texts_to_sequences(questions)
sequences_answer = tokenizer_answer.texts_to_sequences(answers)

# Pad sequences
padded_paragraph = pad_sequences(sequences_paragraph, maxlen=max_length, padding='post', truncating='post')
padded_question = pad_sequences(sequences_question, maxlen=max_length, padding='post', truncating='post')
padded_answer = pad_sequences(sequences_answer, maxlen=max_length, padding='post', truncating='post')

label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = tf.keras.utils.to_categorical(integer_encoded_labels)

(train_paragraph, val_paragraph, 
 train_question, val_question, 
 train_answer, val_answer, 
 train_labels, val_labels) = train_test_split(
    padded_paragraph, padded_question, padded_answer, categorical_labels, test_size=0.2, random_state=42)


tf.keras.backend.clear_session()

# Define inputs
input_paragraph = Input(shape=(max_length,))
input_question = Input(shape=(max_length,))
input_answer = Input(shape=(max_length,))

# Embedding layers
embedding_dim = 128
embedding_paragraph = Embedding(vocab_size, embedding_dim, input_length=max_length)(input_paragraph)
embedding_question = Embedding(vocab_size, embedding_dim, input_length=max_length)(input_question)
embedding_answer = Embedding(vocab_size, embedding_dim, input_length=max_length)(input_answer)

# LSTM layers
lstm_paragraph = Bidirectional(LSTM(128, return_sequences=True))(embedding_paragraph)
lstm_question = Bidirectional(LSTM(128, return_sequences=True))(embedding_question)
lstm_answer = Bidirectional(LSTM(128, return_sequences=True))(embedding_answer)

# Conv1D and MaxPooling layers
conv_paragraph = Conv1D(64, 3, activation='relu')(lstm_paragraph)
conv_question = Conv1D(64, 3, activation='relu')(lstm_question)
conv_answer = Conv1D(64, 3, activation='relu')(lstm_answer)

pool_paragraph = MaxPooling1D(pool_size=2)(conv_paragraph)
pool_question = MaxPooling1D(pool_size=2)(conv_question)
pool_answer = MaxPooling1D(pool_size=2)(conv_answer)

# Flatten layers
flat_paragraph = Flatten()(pool_paragraph)
flat_question = Flatten()(pool_question)
flat_answer = Flatten()(pool_answer)

# Concatenate the outputs
merged = Concatenate()([flat_paragraph, flat_question, flat_answer])

# Fully connected layers
dense_1 = Dense(128, activation='relu')(merged)
output = Dense(categorical_labels.shape[1], activation='softmax')(dense_1)

# Define the model
model = Model(inputs=[input_paragraph, input_question, input_answer], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()


# Training the model
history = model.fit([train_paragraph, train_question, train_answer], train_labels,
                    epochs=100, batch_size=20,
                    validation_data=([val_paragraph, val_question, val_answer], val_labels),
                    callbacks=[EarlyStopping(monitor='val_accuracy', patience=20)],  # early stopping if validation accuracy doesnt improve after n epochs
                    verbose=1)


# Evaluate the model on the validation data
loss, accuracy = model.evaluate([val_paragraph, val_question, val_answer], val_labels, verbose=1)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')


# Make predictions on the validation data
predictions = model.predict([val_paragraph, val_question, val_answer])

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)
decoded_predictions = label_encoder.inverse_transform(predicted_classes)
print(decoded_predictions)


# Example new data
new_paragraph = ["""A database can be of any size and complexity. For example, the list of names and addresses referred to earlier may consist of only a few hundred records, each with a simple structure. On the other hand, the computerized catalog of a large library may contain half a million entries organized under different categoriesâ€”by pri mary authorâ€™s last name, by subject, by book titleâ€”with each category organized alphabetically. A database of even greater size and complexity would be maintained by a social media company such as Facebook, which has more than a billion users. The database has to maintain information on which users are related to one another as friends, the postings of each user, which users are allowed to see each posting, and a vast amount of other types of information needed for the correct operation of their Web site. For such Web sites, a large number of databases are needed to keep track of the constantly changing information required by the social media Web site.
"""]
new_question = ["What does DBMS stand for?"]
new_answer = [r"Database Management System"] # hard

# Convert texts to sequences
new_seq_paragraph = tokenizer_paragraph.texts_to_sequences(new_paragraph)
new_seq_question = tokenizer_question.texts_to_sequences(new_question)
new_seq_answer = tokenizer_answer.texts_to_sequences(new_answer)

# Pad sequences
new_padded_paragraph = pad_sequences(new_seq_paragraph, maxlen=max_length, padding='post', truncating='post')
new_padded_question = pad_sequences(new_seq_question, maxlen=max_length, padding='post', truncating='post')
new_padded_answer = pad_sequences(new_seq_answer, maxlen=max_length, padding='post', truncating='post')

# Make predictions
new_predictions = model.predict([new_padded_paragraph, new_padded_question, new_padded_answer])

# Convert new predictions to class labels
new_predicted_classes = np.argmax(new_predictions, axis=1)
new_decoded_predictions = label_encoder.inverse_transform(new_predicted_classes)
print(new_decoded_predictions)

