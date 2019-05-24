# Copy the folder given in the master branch and paste it into the working directory


------------------------------------- PREPARING  THE  DATA ------------------------------------------------------------------------


from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))                                         # Printing the name of all the files in the folder of Current Directory/data/names/


import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"                                 # Adding the letters {'.',',',';','''} to the ascii_letters={a,b,c,d,...z,A,B,C,D,...,Z}
n_letters = len(all_letters)                                                 
print('n_letters:',n_letters)                                                # n_letters = 57

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
print(unicodeToAscii('Ślusàrski'))                  
print(unicodeToAscii('Ankit'))
category_lines = {}                                                          # A map having Key:"Categories" and Values:"Names"
all_categories = []                                                          # A list containing "Categories"

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
print('Category Lines: ',category_lines)
print('All Category: ',all_categories)
n_categories = len(all_categories)                                            # n_categories = 18 here

-------------------------------------------TURNING NAMES INTO TENSORS ---------------------------------------------------------------------------------------


def letterToIndex(letter):
    return all_letters.find(letter)                                           # Assigning Index to each letter a=0,A=26,.=54 likewise

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor                                                             # One hot encoding of each letter in a Tensor form

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor                                                             # One hot encoding of each letter of a word in a Tensor form

print(letterToIndex('A'))
print(letterToTensor('J'))
print(lineToTensor('Jones'))  

-------------------------------------------CREATING  THE  NETWORK-----------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
input = lineToTensor('This is to certify')
input = torch.cat(list(torch.split(input, 1, dim=0))*10)                     # Explicitly making "input" as a 3d tensor
hidden = (torch.randn(1, 1, 18), torch.randn(1, 1, 18))                      # "hidden" must be a 3d tensor and its product(dimension) must be 18 as per the "n_categories" 
lstm = nn.LSTM(57,18)        
output, next_hidden = lstm(input, hidden)                                    # Caution!!!! lstm always take 3d Tensor as its arguments
print('OUTPUT: ',output)
print('NEXT HIDDEN: ',hidden)

-------------------------------------------- PREPARING  FOR  TRAINING --------------------------------------------------------------------------

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i                            

-------------------------------------------------------------------------------------------------------------------------------

import random
import torch

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)                                                 # Randomnly Generated Category
    line = randomChoice(category_lines[category])                                           # Randomnly Generated Names from Randomnly Generated Category
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)      # Index of the Category Randomnly Generated
    line_tensor = lineToTensor(line)                                                        # One Hot Encoding of each and every letter present in "line" Tensor
    #print(category)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line, '/Category Tensor =',category_tensor, '/Line Tensor =',line_tensor)
    

------------------------------------------- TRAINING  THE  NETWORK ----------------------------------------------------------------------------

import torch.nn as nn
from torch.autograd import Variable
criterion = nn.CrossEntropyLoss()                                       # Using Cross Entropy Loss as LSTM does not implement LogSoftMax unlike RNN(where NLL Loss is used)
learning_rate = 0.003
def train(category_tensor, line_tensor):
    hidden = (torch.zeros(1, 1, 18),torch.zeros(1, 1, 18))
    lstm.zero_grad()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = lstm(line_tensor[i].reshape(1,1,-1), hidden)   # "line_tensor" is reshaped to a 3d tensor
        

    temp = torch.reshape(output, (1, 18))                               # "output" is reshaped to 1*18 as per dimension of "category_tensor" and stored in "temp"
    loss = criterion(temp, category_tensor)
    loss.backward()                                                     # BackPropagating the Error
    
    for p in lstm.parameters():
        p.data.add_(-learning_rate, p.grad.data)                        # Gradient Descent
        
    return output, loss.item()                                          # Dimension of the "output" is still preserved


----------------------------------------------------------------------------------------------------------------------------------

import time
import math



n_iters = 100000
print_every = 5000
plot_every = 1000

current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m*60
    return '%dm %ds' % (m, s)
start = time.time()

for iter in range(0, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s  / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
    
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
        
------------------------------------------- PLOTTING  THE  RESULTS ------------------------------------------------------------------------------------

    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    
    plt.figure()
    plt.plot(all_losses)
   
---------------------------------------------- EVALUATING  THE  RESULTS-------------------------------------------------------------------

    import torch
    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000
    
    def evaluate(line_tensor):
        hidden = (torch.randn(1, 1, 18),torch.randn(1, 1, 18))
        
        for i in range(line_tensor.size()[0]):
            output, hidden = lstm(line_tensor[i].reshape(1, 1, -1), hidden)
        
        return output
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] +=1
    
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)
    
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    plt.show()

    
------------------------------------------ PREDICTING FROM USER INPUT --------------------------------------------------------


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 2, True)
        #print(topv.shape)
        #print(topi.shape)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][0][i].item()
            category_index = topi[0][0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')
    
