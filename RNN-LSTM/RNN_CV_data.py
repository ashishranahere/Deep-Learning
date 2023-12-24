#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import os
import glob
import torch.nn.init as init

import warnings
warnings.filterwarnings("ignore")


# In[59]:


path = "/home/intern/Urvashi/A6/data/CV_Data/"

# In[60]:


os.listdir(path)


# In[103]:


def preprocess_data(path , label):
    train, test = [], []
    train_out, test_out = [], []
    train_path = path+'/Train/'
    test_path = path+'/Test/'
    for i in os.listdir(train_path):
        train_out.append(label)
        train.append(torch.from_numpy(np.loadtxt(train_path+i)).float())
    for i in os.listdir(test_path):
        test_out.append(label)
        test.append(torch.from_numpy(np.loadtxt(test_path+i)).float())
    return train, train_out, test , test_out


# In[104]:


train_ka , tr_op_ka , test_ka , test_op_ka = preprocess_data(path+'ka' , 0)
train_kaa , tr_op_kaa , test_kaa , test_op_kaa = preprocess_data(path+'kaa' , 1)
train_ne , tr_op_ne , test_ne , test_op_ne = preprocess_data(path+'ne',2)
train_nii , tr_op_nii , test_nii , test_op_nii = preprocess_data(path+'nii',3)
train_pa , tr_op_pa , test_pa , test_op_pa = preprocess_data(path+'pa',4)


# In[105]:


train_data = train_ka
train_data.extend(train_kaa)
train_data.extend(train_ne)
train_data.extend(train_nii)
train_data.extend(train_pa)
print(len(train_data))


# In[106]:


train_out = tr_op_ka
train_out.extend(tr_op_kaa)
train_out.extend(tr_op_ne)
train_out.extend(tr_op_nii)
train_out.extend(tr_op_pa)
print(len(train_out))


# In[107]:


test_data = test_ka
test_data.extend(test_kaa)
test_data.extend(test_ne)
test_data.extend(test_nii)
test_data.extend(test_pa)
print(len(test_data))



# In[109]:


test_out = test_op_ka
test_out.extend(test_op_kaa)
test_out.extend(test_op_ne)
test_out.extend(test_op_nii)
test_out.extend(test_op_pa)
print(len(test_out))


# In[111]:


train_out = torch.from_numpy(np.array(train_out)).cuda()
test_out = torch.from_numpy(np.array(test_out)).cuda()
test_out.shape


# In[73]:


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
                init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')


        
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


# In[74]:


# Example usage
input_size = 39
hidden_size = 1024
num_layers = 1
output_size = 5
max_epochs = 10000
dropout=0.4

convergence_threshold = 1e-4

x_padded = nn.utils.rnn.pad_sequence(train_data, batch_first=True).cuda()
seq_lengths = torch.LongTensor([len(seq) for seq in train_data]).cuda()


# Create model and optimizer
model = RNN(input_size, hidden_size, num_layers, output_size,dropout).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()

# Train model
prev_loss = float('inf')
avg_losses = []
accuracy_list =[]
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
            if p==3 :
                print('Converged after {} epochs with Previous Loss {} and Current Loss {} and Differnce {}.'.format(epoch+1 , prev_loss , loss.item(), diff) )
                break
            else:
                print("Patience Up Up")
                p+=1
        else:
            p=0
        
    prev_loss = loss.item()
    
    # Compute training accuracy
    _, predicted = torch.max(outputs.data, 1)
#     print(predicted)
    
    correct = (predicted == train_out.to(predicted.device)).sum().item()
    accuracy = correct / len(predicted)
    accuracy_list.append(accuracy)

    # Print loss and accuracy
    print('Epoch [{}/{}] | Loss: {:.4f} | Accuracy: {:.2f}% | Diff: {:.4f}'.format(epoch+1, max_epochs, loss.item(), accuracy*100 , diff ))
    
    


# In[75]:


import matplotlib.pyplot as plt
plt.plot(range(len(avg_losses)), avg_losses)
# plt.plot(range(len(accuracy_list)), accuracy_list)
plt.title('Average Error vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Average Error')
# plt.savefig( "/home/intern/Urvashi/A6/data/RNN_CV6.png")



# In[114]:


test_padded = nn.utils.rnn.pad_sequence(test_data, batch_first=True).cuda()
seq_test = torch.LongTensor([len(seq) for seq in test_data]).cuda()

# Set the model to evaluation mode
model.eval()

# Evaluate the model on the test data
with torch.no_grad():
    outputs = model(test_padded,seq_test)
    loss = nn.CrossEntropyLoss()(outputs, test_out)
    
    # Calculate the accuracy of the model
    _, predicted = torch.max(outputs, 1)
#     print(predicted)
    correct = (predicted == test_out).sum().item()
    total = test_out.size(0)
    accuracy = correct / total
    print(correct)
    
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')


# In[ ]:




from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Generate some random predictions and ground truth labels
y_pred = predicted.cpu()
y_true = test_out.cpu()


# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ka","kaa","ne","nii","pa"])
disp.plot()
# plt.savefig( "/home/intern/Urvashi/A6/data/RNN_CV_CM6.png")



