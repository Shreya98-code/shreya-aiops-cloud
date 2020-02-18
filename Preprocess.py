"""
Assignment_3 Library for Tweet Preprocessing
"""
import re
from nltoolkit import TweetTokenizer
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

class Preprocess_tweet:

	import numpy as np
	"""
	Class to perform preprocessing and generate padded embeddings of tweets
	"""

	def __init__(self, max_length_tweet =50 ,max_length_dictionary = 10000):

		"""
        Performing class initialization
        :params max_length_tweet:
        :params max_length_dictionary:
        :params embeddings_dict:

        """

		self.max_length_tweet = max_length_tweet
		self.max_length_dictionary = max_length_dictionary
		self.embeddings_dict = {}

		#Loading the dictionary
		glove_files = ['glove_25d_1.txt','glove_25d_2.txt','glove_25d_3.txt']

		j = 0

		for doc in glove_files:
			if j >= max_length_dictionary:
				break
			with open(doc, 'r') as file:
				for line in file:
					values = line.split()
					word = values[0]
					vector = np.asarray(values[1:], "float32")
					self.embeddings_dict[word] = vector
					j += 1
					if j >= max_length_dictionary:
						break						


	#tweet = "Great to have a record 50 patrons"

	@staticmethod
	def remove_stop_word(tweet):
        

        stopwords = []
        with open("english") as files:
            for line in files:
                values = line.split()
                word = values[0]
                stopwords.append(word)
        pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
        tweet = pattern.sub('', tweet)
        return tweet

    def clean_text(self, tweet):
        """
        Clean text
        """

        # URL
        tweet = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", '', tweet)

        tweet = tweet.lower()

        # Numbers
        tweet = re.sub(r"[0-9]+", '', tweet)

        # Stopwords
        tweet = self.remove_sw(tweet)

        # Removing #
        tweet = re.sub(r"#", '', tweet)

        # Removing handle
        tweet = re.sub(r"@[a-zA-Z0-9]+", '', tweet)

        return tweet

    @staticmethod
    def tokenize_text(tweet):

        """
        Tokenize
        """
        tokenizer = TweetTokenizer()
        return tokenizer.tokenize(tweet)

    def replace_token_with_index(self, tweet):
        """
        Replace token
        """
        embd = []
        for i in tweet:
            index = self.embeddings_dict.get(i, 0)
            if isinstance(index, np.ndarray):
                embd.append(index)
        return embd

    def pad_sequence(self, token_ind):
        """
        Pad tokenized sequence
        """
        length = len(token_ind)
        if length < self.max_length_tweet:
            req_d = self.max_length_tweet - length
            token_ind.extend([np.zeros_like(token_ind[0])] * req_d)
        elif length > self.max_length_tweet:
            token_ind = token_ind[:self.max_length_tweet].copy()

        return token_ind


 #    cleaned_text = clean_text(tweet)


	# token = tokenize_text(cleaned_text)

	# token_ind = replace_token_with_index(token)

	# token_ind_pad = pad_sequence(token_ind)

