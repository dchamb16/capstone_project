# Import Libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding, MaxPooling1D, Conv1D, concatenate
from numpy import array

class ConvNet:
    '''
    Class for convolutional neural network text classification model
    '''
    def __init__(self):
        pass

    def define_model(self, length, vocab_size):
        '''
        Defines and compiles a convolutional neural network model

        Parameters
        ___________
        length: int
            Max length of the text strings
        vocab_size: int
            Vocabulary size of the text documents
        '''
        inputs1 = Input(shape=(length,))
        embedding1 = Embedding(vocab_size, 100)(inputs1)
        conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)

        # conv2 = Conv1D(filters=32, kernel_size=2, activation='relu')(embedding1)
        # drop2 = Dropout(0.5)(conv2)
        # pool2 = MaxPooling1D(pool_size=2)(drop2)
        # flat2 = Flatten()(pool2)

        # conv3 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding1)
        # drop3 = Dropout(0.5)(conv3)
        # pool3 = MaxPooling1D(pool_size=2)(drop3)
        # flat3 = Flatten()(pool3)

        # concat = concatenate([flat1,flat2, flat3])

        dense1 = Dense(32, activation='relu')(flat1)
        drop4 = Dropout(0.5)(dense1)
        outputs = Dense(11, activation='softmax')(drop4)

        model = Model(inputs=inputs1, outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.model = model

    def fit_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=128, length=None, vocab_size=None):
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