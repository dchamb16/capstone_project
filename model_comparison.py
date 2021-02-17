from tensorflow.keras.models import load_model
import pandas as pd
from clean_text import CleanText
from encode_text import EncodeText
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

df = pd.read_pickle('./test_data.pkl')

X_test, y_test = df['clean_text'], df.loc[:, df.columns != 'clean_text']

encoder = EncodeText()

encoder.load_encoder('./encoder_files/encoder.pkl')
encoder.load_encoder_variables('./encoder_files/encoder_variables.json')

X_test = encoder.encode_text(X_test, test_data=True)

cnn = load_model('./model_files/cnn_classification_model.h5')
rnn = load_model('./model_files/rnn_classification_model.h5')
hybrid = load_model('./model_files/hybrid_attention_classification_model.h5')

clean = CleanText()

test_text = ['''I cant get my morgan stanley account to connect to EveryDollar. If I cant get it to connect, 
    Im going to need to get a refund. Its the only value I get from the app''']

tt = [clean.prepare_text(t) for t in test_text]
tt = encoder.encode_text(tt, test_data=True)

cnn_res = cnn.predict(tt)
y_test.columns[np.argmax(cnn_res)]

rnn_res = rnn.predict(tt)
y_test.columns[np.argmax(rnn_res)]

hybrid_res = hybrid.predict(tt)
y_test.columns[np.argmax(hybrid_res)]



cnn_res = cnn.predict(X_test)
cnn_res_t = (cnn_res > .5)

rnn_res = rnn.predict(X_test)
rnn_res_t = (rnn_res > .5)

hybrid_res = hybrid.predict(X_test)
hybrid_res_t = (hybrid_res > .5)

cnn_f1 = f1_score(y_test, cnn_res_t, average='weighted')
rnn_f1 = f1_score(y_test, rnn_res_t, average='weighted')
hybrid_f1 = f1_score(y_test, hybrid_res_t, average='weighted')

print(f'cnn f1 score = {round(cnn_f1, 2)}')
print(f'rnn f1 score = {round(rnn_f1, 2)}')
print(f'hybrid f1 score = {round(hybrid_f1, 2)}')

cnn_auc = roc_auc_score(y_test, cnn_res_t, multi_class='ovr', average='weighted')
rnn_auc = roc_auc_score(y_test, rnn_res_t, multi_class='ovr', average='weighted')
hybrid_auc = roc_auc_score(y_test, hybrid_res_t, multi_class='ovr', average='weighted')

print(f'cnn auc score = {round(cnn_auc, 2)}')
print(f'rnn auc score = {round(rnn_auc, 2)}')
print(f'hybrid auc score = {round(hybrid_auc, 2)}')
