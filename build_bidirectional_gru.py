import pandas as pd
from numpy import array
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from clean_text import CleanText
from encode_text import EncodeText
from bidirectional_gru import ConvNet
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.utils import compute_class_weight
import numpy as np

df = pd.read_pickle('./data.pkl')
ct = CleanText()
encoder = EncodeText()

df = df[df['issue'] != '']

df = df[df.issue.str.contains('cant_add_bank|refund_e_|transactions_not_importing')]

df['clean_text'] = df['ticket_text'].apply(lambda x: ct.prepare_text(x))


w = compute_class_weight('balanced', np.unique(df['issue']), df['issue'])

weights = {
    0: w[0],
    1: w[1],
    2: w[2]
}

model = ConvNet()

trainLines, trainLabels = df['clean_text'], df['issue']

# pickle binarizer and load into main.py
lb = LabelEncoder()
transformed_labels = lb.fit_transform(trainLabels)

transformed_labels = to_categorical(transformed_labels)

label_binarizer_path = './label_binarizer.pkl'
with open(label_binarizer_path, 'wb') as handle:
    pickle.dump(lb, handle)

X_train, X_test, y_train, y_test = train_test_split(trainLines, transformed_labels, test_size=.2, random_state=42, stratify=transformed_labels)


length = encoder.max_length(X_train)
vocab_size = encoder.vocab_size(X_train)
X_train = encoder.encode_text(X_train)
X_test = encoder.encode_text(X_test, test_data=True)

encoder.save_encoder('./encoder.pkl')
encoder.save_encoder_variables('./encoder_variables')

model.define_model(length, vocab_size)

model.fit_model(X_train, array(y_train), X_test, array(y_test), epochs=10, batch_size = 64)#, class_weights=weights)

model.save_model('./classification_model.h5')