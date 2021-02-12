import pandas as pd
from clean_text import CleanText
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy import array
import numpy as np
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding, Conv1D, MaxPooling1D, concatenate
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

df = pd.read_pickle('./data.pkl')

clean = CleanText()

df['clean_text'] = df['ticket_text'].apply(lambda x: clean.prepare_text(x))

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(lines):
    return max([len(s.split()) for s in lines])

def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded

tokenizer = create_tokenizer(df['clean_text'])
length = max_length(df['clean_text'])
vocab_size = len(tokenizer.word_index) + 1

trainX = encode_text(tokenizer, df['clean_text'], length)
print(trainX.shape)

y_train = pd.get_dummies(df['issue'])

X_train, X_test, y_train, y_test = train_test_split(trainX, y_train, test_size=.3, stratify=y_train, random_state=1)

class_weights = compute_class_weight('balanced', np.unique(df['issue']), df['issue'])
cw = {
    0: class_weights[0],
    1: class_weights[1],
    2: class_weights[2],
    3: class_weights[3],
    4: class_weights[4],
    5: class_weights[5],
    6: class_weights[6],
    7: class_weights[7],
    8: class_weights[8],
    9: class_weights[9],
    10: class_weights[10],
}


model = Sequential()
model.add(Input(shape=(length,)))
model.add(Embedding(vocab_size, 100))
model.add(Conv1D(filters=32, kernel_size=4, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(11, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', AUC()])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, class_weight=cw)

df['ticket_text'].iloc[2001]
df['issue'].iloc[2001]
test_text = ['I am trying to add my Betterment bank account and when I try and enter my email login credentials and password, it tells me that they are invalid. I am 100% certain they are not. ']
tt = [clean.prepare_text(text) for text in test_text]

tt = encode_text(tokenizer, tt, length)
print(tt)
res = model.predict(tt)
print(res)
np.sum(res)
y_train.columns[np.argmax(res)]



rnn = Sequential()
rnn.add(Input(shape=(length,)))
rnn.add(Embedding(vocab_size, 100))
rnn.add(Bidirectional(GRU(64)))
rnn.add(Dense(32, activation='relu'))
rnn.add(Dense(11, activation='softmax'))

rnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', AUC()])
rnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, class_weight=cw)

df['ticket_text'].iloc[2937]
df['issue'].iloc[2937]
test_text = ['I would like a refund for my recent payment']
tt = [clean.prepare_text(text) for text in test_text]
print(tt)
tt = encode_text(tokenizer, tt, length)
print(tt)
res = rnn.predict(tt)
print(res)
np.sum(res)
y_train.columns[np.argmax(res)]
