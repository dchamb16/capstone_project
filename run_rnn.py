import pandas as pd
from sklearn.model_selection import train_test_split
from clean_text import CleanText
from encode_text import EncodeText
from rnn import Bidirectional_GRU
from sklearn.utils import compute_class_weight
import numpy as np

df = pd.read_pickle('./data_2.pkl')
ct = CleanText()
encoder = EncodeText()

df = df[df['issue'] != '']

df['clean_text'] = df['ticket_text'].apply(lambda x: ct.prepare_text(x))


w = compute_class_weight('balanced', np.unique(df['issue']), df['issue'])

weights = {
    0: w[0],
    1: w[1],
    2: w[2],
    3: w[3],
    4: w[4],
    5: w[5],
    6: w[6],
    7: w[7],
    8: w[8],
    9: w[9]
}

model = Bidirectional_GRU()

trainLines, trainLabels = df['clean_text'], df['issue']

labels = pd.get_dummies(trainLabels)

X_train, X_test, y_train, y_test = train_test_split(trainLines, labels, test_size=.2, random_state=42, stratify=labels)

length = encoder.max_length(X_train)
vocab_size = encoder.vocab_size(X_train)
X_train = encoder.encode_text(X_train)
X_test = encoder.encode_text(X_test, test_data=True)

# encoder.save_encoder('./encoder_files/encoder.pkl')
# encoder.save_encoder_variables('./encoder_files/encoder_variables')

model.define_model(length, vocab_size, labels.shape[1])
model.fit_model(X_train, y_train, X_test, y_test, epochs=20, batch_size = 256, class_weights=weights)

model.save_model('./model_files/rnn_classification_model.h5')