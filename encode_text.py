# Import Libraries
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json

class EncodeText:
    '''
    Class to encode text into numeric values in preperation of modeling
    '''

    def __init__(self):
        self.name = 'EncodeText'
        self.tokenizer = None

    def _create_tokenizer(self, lines):
        '''
        Creates a tokenizer object and fits on lines

        Parameters:
        __________
        lines: str
            The text that is to be encoded

        Returns:
        ________
        Obj
            tokenizer object
        '''
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer
         
    def max_length(self, lines):
        '''
        Finds the max length of the text across the array/list

        Parameters:
        ___________
        lines: str
            Text array/list to find the max length over

        Returns:
        ________
        Int
            Max length of text string in array/list
        '''
        max_len = int(max([len(s.split()) for s in lines]))
        return max_len

    def vocab_size(self, lines):
        '''
        Finds the vocab size of a text array/list

        Parameters:
        ___________
        lines: str
            Text array/list to find the vocabulary over

        Returns:
        ________
        Int
            Vocab size of array/list
        '''
        tokenizer = self._create_tokenizer(lines)
        vocab_size = len(tokenizer.word_index) + 1
        self.vocab_size = vocab_size
        return vocab_size

    def encode_text(self, lines, test_data = False):
        '''
        Encodes text into numerical values

        Parameters:
        ___________
        lines: str
            Text array/list to encode
        test_data: bool
            If test/prod data, don't fit length (changes text padding)

        Returns:
        ________
        List
            list of encoded and padded sequences to be used in modeling
        '''
        # handle if loading a tokenizer
        if self.tokenizer is None:
            tokenizer = self._create_tokenizer(lines)
        else:
            tokenizer = self.tokenizer
        
        if test_data:
            max_len = self.max_len
        else:
            max_len = self.max_length(lines)
            self.max_len = max_len

        encoded = tokenizer.texts_to_sequences(lines)
        padded = pad_sequences(encoded, maxlen=max_len, padding='post')
        
        if not self.tokenizer:
            self.tokenizer = tokenizer
        
        return padded

    def save_encoder(self, path):
        '''
        Serializes encoder and writes to file

        Parameters:
        ___________
        path: str
            Path to write the serialized encoder to
        '''
        with open(path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle)

    def save_encoder_variables(self, path):
        '''
        Pickles variables to be used in the encoder

        Parameters:
        ___________
        path: str
            Path to write the pickled variables to
        '''

        if path[-5:] != '.json':
            save_path = ''.join([path, '.json'])
        else:
            save_path = path

        vars = {"vocab_size": self.vocab_size,
            "max_length": self.max_len}

        with open(save_path, 'w') as outfile:
            json.dump(vars, outfile)

    def load_encoder(self, path):
        '''
        Loads a serialized encoder from a file path

        Parameters:
        ___________
        path: str
            Path to load the serialized encoder from
        '''
        with open(path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        self.tokenizer = tokenizer

    def load_encoder_variables(self, path):
        '''
        Loads variables to be used in encoder from pickle file

        Parameters:
        ___________
        path: str
            Path to pickled encoder variables
        '''
        with open(path, 'rb') as openfile:
            variables = json.load(openfile)
        self.vocab_size = variables['vocab_size']
        self.max_len = variables['max_length']