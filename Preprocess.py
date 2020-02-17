import re
import numpy as np
from nltoolkit import TweetTokenizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

class Preprocess:

	def __init__(self, max_length_tweet =50 ,max_length_dictionary = 10000):
		self.max_length_tweet = max_length_tweet
		self.max_length_dictionary = max_length_dictionary

	#Loading the dictionary
	glov_files = ['glove_25d_1.txt', 'glove_25d_2.txt', 'glove_25d_3.txt']

        c = 0

        self.embeddings_dict = {}

        for doc in glove_files:
            if c >= max_length_dictionary:
                break
            with open(doc, 'r') as f:
            	for line in f:
        			values = line.split()
        			word = values[0]
        			vector = np.asarray(values[1:], "float32")
        			self.embeddings_dict[word] = vector
        			c += 1
                    if c >= max_length_dictionary:
                    	break


	#tweet = "Great to have a record 50 patrons"

	def clean_text(x):
    	tweet = x
    	tweet = tweet.lower()
    	#Removing numbers
    	tweet = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", tweet)
    	#Removing symbols
    	tweet = re.sub(r'[^\w]', ' ', tweet)
    	#Removing links
    	tweet = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", '', tweet)

    	#Removing stopwords
    	stop_words = set(stopwords.words('english')) 

    	# import TweetTokenizer() method from nltk 
    	from nltk.tokenize import TweetTokenizer 

    	# Create a reference variable for Class TweetTokenizer 
    	tk = TweetTokenizer() 

    	# Use tokenize method 
    	token = tk.tokenize(tweet)

    	filtered_sentence = [w for w in token if not w in stop_words] 

    	filtered_sentence = [] 

    	for w in token: 
        	if w not in stop_words: 
            	filtered_sentence.append(w) 
    	s = " "
    	s = s.join(filtered_sentence)
    	return s

    def tokenize_text(filtered_sentence):
    	# import TweetTokenizer() method from nltk 
    	from nltk.tokenize import TweetTokenizer 

    	# Create a reference variable for Class TweetTokenizer 
    	tk = TweetTokenizer() 

    	# Use tokenize method 
    	token = tk.tokenize(filtered_sentence)
    	return token

    def replace_token_with_index(token):
    	list_ = []
    	for i in token:
        	tuy = embeddings_dict.get(i, 0)
        	if isinstance(tuy, np.ndarray):
            	list_.append(tuy)
    	return list_

    def pad_sequence(self, token_ind):
        l = len(token_ind)
        if l < max_length_tweet:
            req_d = max_length_tweet - l
            token_ind.extend([np.zeros_like(token_ind[0])]*req_d)
        elif l > max_length_tweet:
            token_ind = token_ind[:max_length_tweet].copy()
            
        return token_ind

    
 #    cleaned_text = clean_text(tweet)
	

	# token = tokenize_text(cleaned_text)
	
	# token_ind = replace_token_with_index(token)

	# token_ind_pad = pad_sequence(token_ind)







