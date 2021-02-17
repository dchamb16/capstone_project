# Import Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, GRU, Bidirectional, Conv1D, MaxPooling1D, SpatialDropout1D, Attention, Flatten
from tensorflow.keras import metrics
from numpy import array

class HybridModel:
    '''
    Class for convolutional neural network text classification model
    '''
    def __init__(self):
        pass

    def define_model(self, length, vocab_size, num_outcome_classes):
        '''
        Defines and compiles a convolutional neural network model

        Parameters
        ___________
        length: int
            Max length of the text strings
        vocab_size: int
            Vocabulary size of the text documents
        '''
        
        model = Sequential()
        model.add(Input(shape=(length,)))
        model.add(Embedding(vocab_size, 200))
        model.add(Bidirectional(GRU(128, return_sequences=True)))
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dropout(.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_outcome_classes, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy', metrics.AUC()])
        
        self.model = model

    def fit_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=128, length=None, vocab_size=None, class_weights=None):
        '''
        Fits the CNN model to provided data

        Parameters
        __________
        X_train: array
            Array of encoded text data to train the model on
        y_train: array
            Array of response variables, matching X_train
        X_test: array
            Array of encoded text data to validate the model on
        y_test: array
            Array of response variables, matching X_test, to validate the model
        epochs: int
            Number of iterations to train the model
        batch_size: int
            Number of training samples to work through before updating model parameters
        length: int
            Max length of the text strings
        vocab_size: int
            Vocabulary size of the text documents
        '''
        if not self.model:
            self.define_model(length, vocab_size)

        model = self.model

        if class_weights:
            history=model.fit(X_train, array(y_train), validation_data=(X_test, array(y_test)), epochs=epochs, batch_size=batch_size, class_weight=class_weights)
        else:
            history=model.fit(X_train, array(y_train), validation_data=(X_test, array(y_test)), epochs=epochs, batch_size=batch_size)


        self.model = model
        self.history = history

    def save_model(self, path):
        '''
        Saves CNN model to a h5 file

        Parameters
        __________
        path: str
            Path in which to write the h5 file to
        '''
        model = self.model
        
        if path[-3:] != '.h5':
            save_path = ''.join([path, '.h5'])
        else:
            save_path = path

        model.save(save_path)