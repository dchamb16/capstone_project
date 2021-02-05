# Import Libraries
import re
import pandas as pd
import string

class CleanText:
    '''
    Class to clean text before encoding and using for modeling
    '''
    def __init__(self):
        pass

    def remove_urls(self, string):
        '''
        Removes URLs from a string

        Parameters:
        __________
        string: str
            string to remove urls from

        Returns:
        ________
        Str
            input strings with URLs removed
        '''
        text = re.sub(r'http\S+', ' ', string, flags=re.MULTILINE)
        return text

    def remove_email_addresses(self, string):
        '''
        Removes email addressess from a string

        Parameters:
        ___________
        string: str
            string to remove email addresses from

        Returns:
        ________
        Str
            input string with email addresses removed
        '''
        text = re.sub(r'\S*@\S*\s?', ' ', string, flags=re.MULTILINE)
        return text

    def remove_byte_order_mark(self, string):
        '''
        Removes byte order mark (\ufeff) from a string 

        Parameters:
        ___________
        string: str
            string to remove byte order mark from
        
        Returns:
        ________
        Str
            input string with byte order mark removed
        '''
        text = re.sub(r'\ufeff', '',string)
        return text
    
    def remove_new_lines(self, string):
        '''
        Removes the newline characters (\n) from a string

        Parameters:
        ___________
        string: str
            string to remove new line breaks from

        Returns:
        ________
        Str
            input string with new line breaks removed
        '''
        text = re.sub(r'\n', ' ', string, flags=re.MULTILINE)
        return text
        
    def remove_carriage_returns(self, string):
        '''
        Removes carriage returns (\r) from a string

        Parameters:
        ___________
        string: str
            string to remove carriage returns from

        Returns:
        ________
        Str
            input string with carriage returns removed
        '''
        text = re.sub(r'\r', '', string, flags = re.MULTILINE)
        return text

    def remove_punctuation(self, s):
        '''
        Removes punctuation from a string

        Parameters:
        ___________
        string: str
            string to remove punctuation from

        Returns:
        ________
        Str
            input string with punctuation removed
        '''
        text = s.translate(str.maketrans('','', string.punctuation))
        return text

    def remove_phone_numbers(self, string):
        '''
        Removes phone numbers from a string

        Parameters:
        ___________
        string: str
            string to remove phone numbers from

        Returns:
        ________
        Str
            input string with phone numbers removed
        '''
        text = re.sub(r'\d{10}|\d{11}', '', string, re.MULTILINE)
        return text

    def make_lowercase(self, string):
        '''
        Makes text lower case

        Parameters:
        ___________
        string: str
            string to make lower case

        Returns:
        ________
        Str
            lower case input string
        '''
        text = string.lower()
        return text

    def remove_extra_space(self, string):
        '''
        Removes extra spaces from string

        Parameters:
        ___________
        string: str
            string to remove spaces from

        Returns:
        ________
        Str
            input string with extra spaces removed
        '''
        text = re.sub(r'\s+', ' ', string)
        return text

    def remove_followup_request(self, string):
        '''
        Removes the text "this is a followup to your previous request" from a string

        Parameters:
        ___________
        string: str
            string to remove text from
        
        Returns:
        ________
        Str
            input string with text removed
        '''
        text = re.sub(r'this is a followup to your previous request \d{6}', '', string)
        return text

    def prepare_text(self, string):
        '''
        Runs all the text cleaning methods

        Parameters:
        ___________
        string: str
            string to clean

        Returns:
        ________
        Str
            input string to clean
        '''
        text = self.remove_urls(string)
        text = self.remove_email_addresses(text)
        text = self.remove_byte_order_mark(text)
        text = self.remove_new_lines(text)
        text = self.remove_carriage_returns(text)
        text = self.remove_punctuation(text)
        text = self.remove_phone_numbers(text)
        text = self.make_lowercase(text)
        text = self.remove_followup_request(text)
        text = self.remove_extra_space(text)
        text = text.strip()

        return text
