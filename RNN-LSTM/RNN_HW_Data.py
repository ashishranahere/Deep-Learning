#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import torch.nn.init as init

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


import warnings
warnings.filterwarnings("ignore")


# In[55]:


path = "/home/intern/Urvashi/A6/data/Handwriting_Data/"
os.listdir(path)


# In[56]:


def preprocess_data(path , label):
    train_path = path+'/train/'
    test_path = path+'/dev/'
    train_data , test_data = [],[]
    train_out , test_out = [],[]
    
    for i in os.listdir(train_path):
        train_sample = np.loadtxt(train_path+i)    
        num_points = int(train_sample[0])
        train_out.append(label)
        sequence=[]
        for i in range(1,num_points*2+1,2):
            point_xy = []
            point_x = point_xy.append(train_sample[i])
            point_y = point_xy.append(train_sample[i+1])
            sequence.append(point_xy)
        sequence = torch.from_numpy(np.array(sequence)).float()
        train_data.append(sequence)
        
    for i in os.listdir(test_path):
        test_sample = np.loadtxt(test_path+i)    
        num_points = int(test_sample[0])
        test_out.append(label)
        sequence=[]
        for i in range(1,num_points*2+1,2):
            point_xy = []
            point_x = point_xy.append(test_sample[i])
            point_y = point_xy.append(test_sample[i+1])
            sequence.append(point_xy)
        sequence = torch.from_numpy(np.array(sequence)).float()
        test_data.append(sequence)
   
    return train_data , test_data , train_out , test_out


# In[57]:


train_a, test_a , tr_op_a, test_op_a = preprocess_data(path+'a' , 0)
train_chA, test_chA , tr_op_chA ,  test_op_chA= preprocess_data(path+'chA' , 1)
train_dA, test_dA , tr_op_dA , test_op_dA = preprocess_data(path+'dA' , 2)
train_lA, test_lA , tr_op_lA , test_op_lA = preprocess_data(path+'lA' , 3)
train_tA, test_tA , tr_op_tA , test_op_tA = preprocess_data(path+'tA' , 4)


# In[58]:


train_data = train_a
train_data.extend(train_chA)
train_data.extend(train_dA)
train_data.extend(train_lA)
train_data.extend(train_tA)
print(len(train_data))


# In[87]:


train_data_norm=[]
for i in train_data:
#     print(np.array(i))
    min_vals = np.min(np.array(i), axis=0)
    max_vals = np.max(np.array(i), axis=0)
    data_normalized = np.array(i) - min_vals    
    range_vals = max_vals - min_vals
    i = data_normalized / range_vals
    i = torch.from_numpy(i)
    train_data_norm.append(i)
#     print(i)
len(train_data_norm)


# In[59]:


test_data = test_a
test_data.extend(test_chA)
test_data.extend(test_dA)
test_data.extend(test_lA)
test_data.extend(test_tA)
print(len(test_data))


# In[89]:


test_data_norm=[]
for i in test_data:
#     print(np.array(i))
    min_vals = np.min(np.array(i), axis=0)
    max_vals = np.max(np.array(i), axis=0)
    data_normalized = np.array(i) - min_vals    
    range_vals = max_vals - min_vals
    i = data_normalized / range_vals
    i = torch.from_numpy(i)
    test_data_norm.append(i)
#     print(i)
len(test_data_norm)


# In[60]:


train_out = tr_op_a
train_out.extend(tr_op_chA)
train_out.extend(tr_op_dA)
train_out.extend(tr_op_lA)
train_out.extend(tr_op_tA)
print(len(train_out))


# In[61]:


test_out = test_op_a
test_out.extend(test_op_chA)
test_out.extend(test_op_dA)
test_out.extend(test_op_lA)
test_out.extend(test_op_tA)
print(len(test_out))


# In[82]:


train_out = torch.from_numpy(np.array(train_out)).cuda()
test_out = torch.from_numpy(np.array(test_out)).cuda()
train_out.shape


# In[126]:
def plot_character(data,sp):
    min_vals = np.min(np.array(data), axis=0)
    max_vals = np.max(np.array(data), axis=0)
    data_normalized = np.array(data) - min_vals    
    range_vals = max_vals - min_vals
    data = data_normalized / range_vals
    x = data[:,0]
    y = data[:,1]    
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.savefig(sp)



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size , dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        
        # apply Kaiming initialization to the weights of the RNN
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                init.kaiming_normal_(param, mode='fan_in',nonlinearity='tanh')


        
    def forward(self, x, seq_lengths):        
        
        
        seq_lengths, idx_sort = torch.sort(seq_lengths, descending=True)        
        x_sort = x[idx_sort]        
        _, idx_unsort = torch.sort(idx_sort)
        x_packed = nn.utils.rnn.pack_padded_sequence(x_sort, seq_lengths.cpu(), batch_first=True)


        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through RNN
        out, _ = self.rnn(x_packed, h0)

        # Unpack the padded sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        out = self.dropout(out)

        # Get the last output for each sequence
        idx_last = (seq_lengths - 1).view(-1, 1).expand(len(seq_lengths), out.size(2)).unsqueeze(1)
        out_last = out.gather(1, idx_last).squeeze()

        # Pass through fully connected layer
        out_fc = self.fc(out_last)
        out_fc = self.softmax(out_fc)

        # Unsort the output
        out_fc = out_fc[idx_unsort]

        return out_fc


# In[ ]:


# Example usage
input_size = 2
hidden_size = 128
num_layers = 1
output_size = 5
max_epochs = 10000
dropout=0.4

convergence_threshold = 1e-4

x_padded = nn.utils.rnn.pad_sequence(train_data_norm, batch_first=True).cuda()
seq_lengths = torch.LongTensor([len(seq) for seq in train_data]).cuda()


# Create model and optimizer
model = RNN(input_size, hidden_size, num_layers, output_size,dropout).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

criterion = nn.CrossEntropyLoss()

# Train model
prev_loss = float('inf')
avg_losses = []
error = 0.0
diff = 0.0
p=0
for epoch in range(max_epochs):
    optimizer.zero_grad()
    
    outputs = model(x_padded, seq_lengths)
#     print(outputs.shape)
    loss = criterion(outputs, train_out)
#     print(loss)
    
    loss.backward()
    optimizer.step()    
    
    if epoch > 0:
        diff = (prev_loss - loss.item())
#         print(diff)
        avg_losses.append(loss.item() )
       
        if diff  < convergence_threshold :
            if p==3:
                print('Converged after {} epochs with Previous Loss {} and Current Loss {} and Differnce {}.'.format(epoch+1 , prev_loss , loss.item(), diff) )
                break
            else:
                print("Patience Up Up")
                p+=1
        else:
            p=0
    # Check parameter gradients for vanishing gradient problem
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(name, torch.mean(torch.abs(param.grad)))

        
    prev_loss = loss.item()
    
    # Compute training accuracy
    _, predicted = torch.max(outputs.data, 1)
#     print(predicted)
    
    correct = (predicted == train_out.to(predicted.device)).sum().item()
    # print(correct)
    accuracy = correct / len(predicted)

    # Print loss and accuracy
    print('Epoch [{}/{}] | Loss: {:.4f} | Accuracy: {:.2f}% | Diff: {:.4f}'.format(epoch+1, max_epochs, loss.item(), accuracy*100 , diff ))
    
    


# In[124]:


import matplotlib.pyplot as plt
plt.plot(range(len(avg_losses)), avg_losses)
plt.title('Average Error vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Average Error')
# plt.savefig("/home/intern/Urvashi/A6/RNN_HW4.png")


# In[125]:


test_padded = nn.utils.rnn.pad_sequence(test_data_norm, batch_first=True).cuda()
seq_test = torch.LongTensor([len(seq) for seq in test_data]).cuda()

# Set the model to evaluation mode
model.eval()

# Evaluate the model on the test data
with torch.no_grad():
    outputs = model(test_padded,seq_test)
    loss = nn.CrossEntropyLoss()(outputs, test_out)
    
    # Calculate the accuracy of the model
    _, predicted = torch.max(outputs, 1)
    print(predicted)
    correct = (predicted == test_out).sum().item()
    total = test_out.size(0)
    accuracy = correct / total    
    
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

    misclassified_samples=[]
    t=0
    for i in range(len(predicted)):
        if predicted[i] != test_out[i]:
            misclassified_samples.append(i)
    
    print("Missclassified indices: a2")
    print(misclassified_samples)
    
    # for i in  misclassified_samples:
    #     plot_character(test_data[i] , "/home/intern/Urvashi/A6/"+str(i)+".png" )


# In[ ]:

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate some random predictions and ground truth labels
y_pred = predicted.cpu()
y_true = test_out.cpu()


# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["a","chA","dA","lA","tA"])
disp.plot()
# plt.savefig("/home/intern/Urvashi/A6/RNN_HW_CM4.png")



# In[ ]:




