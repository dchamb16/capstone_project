from encode_text import EncodeText
from clean_text import CleanText
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding, MaxPooling1D, Conv1D, concatenate
from tensorflow.keras import metrics
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from numpy import array
import pandas as pd

df = pd.read_pickle('./data.pkl')
ct = CleanText()
encoder = EncodeText()

df = df[df.issue.str.contains('cant_add_bank|refund_e_|transactions_not_importing')]

df['clean_text'] = df['ticket_text'].apply(lambda x: ct.prepare_text(x))

trainLines, trainLabels = df['clean_text'], df['issue']

lb = LabelEncoder()
transformed_labels = lb.fit_transform(trainLabels)
transformed_labels = to_categorical(transformed_labels)

X_train, X_test, y_train, y_test = train_test_split(trainLines, transformed_labels, test_size=.2, random_state=42, stratify=transformed_labels)


length = encoder.max_length(X_train)
vocab_size = encoder.vocab_size(X_train)
X_train = encoder.encode_text(X_train)
X_test = encoder.encode_text(X_test, test_data=True)

inputs1 = Input(shape=(length,))
embedding1 = Embedding(vocab_size, 100)(inputs1)
conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
drop1 = Dropout(0.5)(conv1)
pool1 = MaxPooling1D(pool_size=2)(drop1)
flat1 = Flatten()(pool1)

dense1 = Dense(64, activation='relu')(flat1)
outputs = Dense(3, activation='softmax')(dense1)

model = Model(inputs=inputs1, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', metrics.AUC()])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)

test_text = "I have been an every dollar user for the past 18 months and occasionally have issues loading my USAA accounts.  I followed your troubleshooting page and have deleted my bank account and attempted to reload it but hasn't loaded.  Any help would be appreciated."

test_text = ct.prepare_text(test_text)

test_text = encoder.encode_text(test_text, test_data=True)

model.predict(test_text)