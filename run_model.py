import pandas as pd
from sklearn.model_selection import train_test_split
from clean_text import CleanText
from encode_text import EncodeText
from sklearn.utils import compute_class_weight
import numpy as np

class RunModel:
    def __init__(self, model):
        """Loads the specified model for training

        Args:
            model (str): Can be rnn, cnn, or hybrid. Specifies the model to load.
        """
        if model == 'rnn':
            from rnn import Bidirectional_GRU
            self.model = Bidirectional_GRU()
        elif model == 'cnn':
            from conv_net import ConvNet
            self.model = ConvNet()
        elif model == 'hybrid':
            from hybrid_model_attention import HybridModel
            self.model = HybridModel()

    def _weights_helper(self, y):
        """Calculates the class weights and returns a dictionary to be passed into the model training

        Args:
            y (str array): Labels for training data

        Returns:
            dict: dictionary containing the class weights
        """
        w = compute_class_weight('balanced', np.unique(y), y)
        weights = {}

        for i in range(len(w)):
            weights[i] = w[i]

        return weights

    def _prepare_data(self, data_path, test_size, random_state):
        """Loads data and prepares for training

        Args:
            data_path (str): File path to the data
            test_size (float): Percent of the data to use for the test set
            random_state (int): Seed for randomly splitting data for train and test sets
        """
        ct = CleanText()

        df = pd.read_pickle(data_path)
        df = df[df['issue'] != '']

        df['clean_text'] = df['ticket_text'].apply(lambda x: ct.prepare_text(x))

        weights = self._weights_helper(df['issue'])

        trainLines, trainLabels = df['clean_text'], df['issue']
        labels = pd.get_dummies(trainLabels)

        X_train, X_test, y_train, y_test = train_test_split(trainLines, labels, test_size=test_size, random_state=random_state, stratify=labels)

        encoder = EncodeText()
        length = encoder.max_length(X_train)
        vocab_size = encoder.vocab_size(X_train)
        X_train = encoder.encode_text(X_train)
        X_test = encoder.encode_text(X_test, test_data=True)

        self.weights = weights
        self.labels = labels
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.length = length
        self.vocab_size = vocab_size

    def run_model(self, data_path, model_save_to_path, test_size=.2, random_state=42, epochs=20, batch_size=256):
        """Loads data, trains the model, and saves the model to a file.

        Args:
            data_path (str): File path to the data
            model_save_to_path (str): File path for the model to be saved at
            test_size (float, optional): Percent of the data to use for the test set. Defaults to .2.
            random_state (int, optional): Seed for randomly splitting data for train and test sets. Defaults to 42.
            epochs (int, optional): Number of iterations to train the model for. Defaults to 20.
            batch_size (int, optional): Number of training samples for each batch during training. Defaults to 256.
        """
        model = self.model
        
        self._prepare_data(data_path, test_size=test_size, random_state=random_state)
        model.define_model(self.length, self.vocab_size, self.labels.shape[1])
        model.fit_model(self.X_train, self.y_train, self.X_test, self.y_test, epochs=epochs, batch_size=batch_size, class_weights=self.weights)

        model.save_model(model_save_to_path)