import json
import os
import random
import re

import nltk
import torch
from torch import nn, optim
import torch.nn.functional as F

import numpy as np

def read_file(training_data_file_name):

  with open(training_data_file_name, 'r') as f:
      twits = json.load(f)

  print(twits['data'][:10])
  print(len(twits['data']))

  messages = [twit['message_body'] for twit in twits['data']]
  # Since the sentiment scores are discrete, scale the sentiments to 0 to 4 for use in the network
  sentiments = [twit['sentiment'] + 2 for twit in twits['data']]

  return messages, sentiments


def preprocess(message):
    """
    This function takes a string as input, then performs these operations: 
        - lowercase
        - remove URLs
        - remove ticker symbols 
        - removes punctuation
        - tokenize by splitting the string on whitespace 
        - removes any single character tokens
    
    Parameters
    ----------
        message : The text message to be preprocessed.
        
    Returns
    -------
        tokens: The preprocessed text into tokens.
    """ 
    
    # Lowercase the twit message
    text = message.lower()
    
    # Replace URLs with a space in the message
    text = re.sub("http(s)?://([\w\-]+\.)+[\w-]+(/[\w\- ./?%&=]*)?",' ', text)
    
    # Replace ticker symbols with a space. The ticker symbols are any stock symbol that starts with $.
    text = re.sub("\$[^ \t\n\r\f]+", ' ', text)
    
    # Replace StockTwits usernames with a space. The usernames are any word that starts with @.
    text = re.sub("@[^ \t\n\r\f]+", ' ', text)

    # Replace everything not a letter with a space
    text = re.sub("[^a-z]", ' ', text)
    
    
    # Tokenize by splitting the string on whitespace into a list of words
    tokens = text.split()

    # Lemmatize words using the WordNetLemmatizer. You can ignore any word that is not longer than one character.
    wnl = nltk.stem.WordNetLemmatizer()
    tokens = [wnl.lemmatize(w, pos='v') for w in tokens if len(w) > 1]
    
    return tokens


def bow(tokenized):

  out_list = tokenized
  words = [element for in_list in out_list for element in in_list]

  print(words[:13])
  print(len(words))

  """
  Create a vocabulary by using Bag of words
  """
  from collections import Counter
  word_counts = Counter(words)
  sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
  int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
  vocab_to_int = {word:ii for ii, word in int_to_vocab.items()}

  print("len(sorted_vocab):",len(sorted_vocab))
  print("sorted_vocab - top:", sorted_vocab[:3])
  print("sorted_vocab - least:", sorted_vocab[-15:])

  # Dictionart that contains the Frequency of words appearing in messages.
  # The key is the token and the value is the frequency of that word in the corpus.
  total_count = len(words)
  freqs = {word: count/total_count for word, count in word_counts.items()}

  print("freqs[the]:",freqs["the"] )

  return sorted_vocab, freqs


def cutoff(sorted_vocab, tokenized, freqs, low_cutoff = 0.000002, high_cutoff = 20):
  # low_cutoff: Float that is the frequency cutoff. Drop words with a frequency that is lower or equal to this number.
  # high_cutoff: Float that is the frequency cutoff. Drop words with a frequency that is higher or equal to this number.

  # Integer that is the cut off for most common words. Drop words that are the `high_cutoff` most common words.
  
  print("high_cutoff:",high_cutoff)
  print("low_cutoff:",low_cutoff)

  # The k most common words in the corpus. Use `high_cutoff` as the k.
  #K_most_common = [word for word in sorted_vocab[:high_cutoff]]
  K_most_common = sorted_vocab[:high_cutoff]

  print("K_most_common:",K_most_common)


  ## Updating Vocabulary by Removing Filtered Words

  filtered_words = [word for word in freqs if (freqs[word] > low_cutoff and word not in K_most_common)]

  print("len(filtered_words):",len(filtered_words))

  # A dictionary for the `filtered_words`. The key is the word and value is an id that represents the word. 
  vocab =  {word:ii for ii, word in enumerate(filtered_words)}
  # Reverse of the `vocab` dictionary. The key is word id and value is the word. 
  id2vocab = {ii:word for word, ii in vocab.items()}
  # tokenized with the words not in `filtered_words` removed.

  print("len(tokenized):", len(tokenized))

  filtered = [[token for token in tokens if token in vocab] for tokens in tokenized]
  print("len(filtered):", len(filtered))
  print("tokenized[:1]", tokenized[:1])
  print("filtered[:1]",filtered[:1])
  
  return filtered, vocab


def balance_classes(sentiments, filtered):
  balanced = {'messages': [], 'sentiments':[]}

  n_neutral = sum(1 for each in sentiments if each == 2)
  N_examples = len(sentiments)
  keep_prob = (N_examples - n_neutral)/4/n_neutral

  for idx, sentiment in enumerate(sentiments):
      message = filtered[idx]
      if len(message) == 0:
          # skip this message because it has length zero
          continue
      elif sentiment != 2 or random.random() < keep_prob:
          balanced['messages'].append(message)
          balanced['sentiments'].append(sentiment)

  n_neutral = sum(1 for each in balanced['sentiments'] if each == 2)
  N_examples = len(balanced['sentiments'])
  print(n_neutral/N_examples)


  ## convert our tokens into integer ids which we can pass to the network.
  messages = balanced['messages']
  #token_ids = [[vocab[word] for word in message] for message in balanced['messages']]
  sentiments = balanced['sentiments']
  return sentiments, messages


def convert_to_token_ids(messages, vocab):
  token_ids = [[vocab[word] for word in message] for message in messages]
  return token_ids
