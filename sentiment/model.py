import random

import torch
from torch import nn, optim

import numpy as np

def load_data(messages, labels, sequence_length=30, batch_size=32, shuffle=False):
    """ 
    Load data.
    """
    if shuffle:
        indices = list(range(len(messages)))
        random.shuffle(indices)
        messages = [messages[idx] for idx in indices]
        labels = [labels[idx] for idx in indices]

    total_sequences = len(messages)

    for ii in range(0, total_sequences, batch_size):
        batch_messages = messages[ii: ii+batch_size]
        
        # First initialize a tensor of all zeros
        batch = torch.zeros((sequence_length, len(batch_messages)), dtype=torch.int64)
        for batch_num, tokens in enumerate(batch_messages):
            token_tensor = torch.tensor(tokens)
            # Left pad!
            start_idx = max(sequence_length - len(token_tensor), 0)
            batch[start_idx:, batch_num] = token_tensor[:sequence_length]
        
        label_tensor = torch.tensor(labels[ii: ii+len(batch_messages)])
        
        yield batch, label_tensor


def train_model(model, train_features, train_labels, epochs = 4, batch_size =  512, learning_rate = 0.001, print_every = 100):
  """
  Train a model with dropout. Clip gradients.
  Print the training loss, validation loss, and validation accuracy for every 100 steps.
  """ 

  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  model.train()

  val_losses = []
  accuracy = []

  for epoch in range(epochs):
      print('Starting epoch {}'.format(epoch + 1))

      steps = 0
      for text_batch, labels in load_data(
              train_features, train_labels, batch_size=batch_size, sequence_length=20, shuffle=True):
          steps += 1
          hidden = model.init_hidden(labels.shape[0]) #pass

          # Set Device
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          text_batch, labels = text_batch.to(device), labels.to(device)
          for each in hidden:
              each.to(device)

          # Train Model
          hidden = tuple([each.data for each in hidden])
          model.zero_grad()
          output, hidden = model(text_batch, hidden)
          loss = criterion(output.squeeze(), labels)
          loss.backward()
          clip = 5
          nn.utils.clip_grad_norm_(model.parameters(), clip)
          optimizer.step()

          # Accumulate loss
          val_losses.append(loss.item())

          correct_count = 0.0
          if steps % print_every == 0:
              model.eval()

              # Calculate accuracy
              ps = torch.exp(output)
              top_p, top_class = ps.topk(1, dim=1)

              correct_count += torch.sum(top_class.squeeze()== labels)

              label_count = len(labels)
              #print(label_count)
              #print(correct_count.cpu().numpy())

              correct_count_num = correct_count.cpu().numpy()
              #print(correct_count_num/label_count)

              accuracy.append(correct_count_num/label_count)

              # Print metrics
              print("Epoch: {}/{}...".format(epoch+1, epochs),
                   "Step: {}...".format(steps),
                   "Collect Count: {}".format(correct_count),
                   "Total Count: {}".format(len(labels)),
                   "Loss: {:.6f}...".format(loss.item()),
                   "Loss Avg: {:.6f}".format(np.mean(val_losses)),
                   #"Accuracy: {:.2f}".format((100*correct_count_num/len(labels))),
                   "Accuracy: {:.2f}".format(correct_count_num/label_count),
                   "Accuracy Avg: {:.2f}".format(np.mean(accuracy))
                   )

              model.train()
  print("Last Loss Avg: {:.6f}".format(np.mean(val_losses)))
  print("Last Accuracy Avg: {:.2f}".format(np.mean(accuracy)))

  #torch.save(model.state_dict(), "./model.pth")
  torch.save(model, "./model.torch")
  return np.mean(accuracy), np.mean(val_losses)


def split_data(token_ids, sentiments, vocab, split_frac = 0.8):
  """
  Split data into training and validation datasets.
  The features are the `token_ids` and the labels are the `sentiments`.
  """   

  ## split data into training, validation, and test data (features and labels, x and y)
  split_idx = int(len(token_ids)*split_frac)
  train_features, remaining_features = token_ids[:split_idx], token_ids[split_idx:]
  train_labels, remaining_labels = sentiments[:split_idx], sentiments[split_idx:]

  test_idx = int(len(remaining_features)*0.5)
  valid_features, test_features = remaining_features[:test_idx], remaining_features[test_idx:]
  valid_labels, test_labels = remaining_labels[:test_idx], remaining_labels[test_idx:]
  return train_features, train_labels, test_features, test_labels, valid_features, valid_labels


def create_model(train_features, train_labels, vocab):

  text_batch, labels = next(iter(load_data(train_features, train_labels, sequence_length=20, batch_size=64)))
  model = TextClassifier(len(vocab)+1, 200, 128, 5, dropout=0.)
  hidden = model.init_hidden(64)
  logps, hidden = model.forward(text_batch, hidden)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = TextClassifier(len(vocab)+1, 1024, 512, 5, lstm_layers=2, dropout=0.2)
  model.embedding.weight.data.uniform_(-1, 1)
  model.to(device)
  return model



class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
        """
        Initialize the model by setting up the layers.
        
        Parameters
        ----------
            vocab_size : The vocabulary size.
            embed_size : The embedding layer size.
            lstm_size : The LSTM layer size.
            output_size : The output size.
            lstm_layers : The number of LSTM layers.
            dropout : The dropout probability.
        """
        
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        
        # TODO Implement

        # Setup embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        
        # Setup additional layers
        self.lstm = nn.LSTM(self.embed_size, self.lstm_size, self.lstm_layers, dropout=self.dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_size, output_size)
        
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size):
        """ 
        Initializes hidden state
        
        Parameters
        ----------
            batch_size : The size of batches.
        
        Returns
        -------
            hidden_state
            
        """

        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.lstm_layers, batch_size,self.lstm_size).zero_(),
                         weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())
        return hidden



    def forward(self, nn_input, hidden_state):
        """
        Perform a forward pass of our model on nn_input.
        
        Parameters
        ----------
            nn_input : The batch of input to the NN.
            hidden_state : The LSTM hidden state.

        Returns
        -------
            logps: log softmax output
            hidden_state: The new hidden state.

        """

        batch_size = nn_input.size(0)
        
        embeds = self.embedding(nn_input)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        
        #lstm_out = lstm_out.contiguous().view(-1, self.lstm_size)    
        """
        do not have batch_first=True, 
        so accordingly shape a input. 
        Moreover, since now input is seq_length x batch, just need to transform lstm_out = lstm_out[-1,:,:].
        Don't have to use batch_first=True in this case, 
        nor reshape the outputs with .view just transform lstm_out as above
        """
        lstm_out = lstm_out[-1,:,:]
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        logps = self.softmax(out)
        
        
        return logps, hidden_state
