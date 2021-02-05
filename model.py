import pandas as pd
from numpy import array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from clean_text import CleanText
from encode_text import EncodeText
from conv_net import ConvNet

df = pd.read_pickle('./data.pkl')
ct = CleanText()
encoder = EncodeText()

df['clean_text'] = df['ticket_text'].apply(lambda x: ct.prepare_text(x))

model = ConvNet()

trainLines, trainLabels = df['clean_text'], df['issue']

lb = LabelBinarizer()
transformed_labels = lb.fit_transform(trainLabels)

X_train, X_test, y_train, y_test = train_test_split(trainLines, transformed_labels, test_size=.2, random_state=42, stratify=transformed_labels)


length = encoder.max_length(X_train)
vocab_size = encoder.vocab_size(X_train)
X_train = encoder.encode_text(X_train)
X_test = encoder.encode_text(X_test, test_data=True)

encoder.save_encoder('./encoder.pkl')
encoder.save_encoder_variables('./encoder_variables')

model.define_model(length, vocab_size)

model.fit_model(X_train, array(y_train), X_test, array(y_test), epochs=50, batch_size = 32)

model.save_model('./classification_model.h5')