import nltk
nltk.download('wordnet')

from sentiment.proc import read_file, preprocess, bow, cutoff, balance_classes, convert_to_token_ids
from sentiment.model import split_data, create_model, train_model

training_data_file_name = 'output.json'
#training_data_file_name = 'twits_dumped.json'

messages, sentiments = read_file(training_data_file_name)

print(messages[:3])

tokenized = list(map(preprocess, messages))

print(tokenized[:3])
print(len(tokenized))

sorted_vocab, freqs = bow(tokenized)

filtered, vocab = cutoff(sorted_vocab, tokenized, freqs)

sentiments, messages = balance_classes(sentiments, filtered)

token_ids = convert_to_token_ids(messages, vocab)

print(type(token_ids))
print(len(token_ids)*0.5)
split_idx = int(len(token_ids)*0.5)

split_frac = 0.98 # for small data
#split_frac = 0.8 # for big data
train_features, train_labels, tf, tl, vf, vl = split_data(token_ids, sentiments, vocab, split_frac = split_frac)

model = create_model(train_features, train_labels, vocab)

acc, loss = train_model(model, train_features, train_labels, print_every = 1) # for small data
#acc, loss = train_model(model, train_features, train_labels) # for big data

import cdsw
model_filename = "model.torch"
vocab_filename = "vocab.pickle"
cdsw.track_file(model_filename)
cdsw.track_file(vocab_filename)

cdsw.track_metric("Accuracy",acc)
cdsw.track_metric("Loss",loss)
