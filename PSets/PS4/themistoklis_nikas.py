# %% [markdown]
# # CS541: Applied Machine Learning, Spring 2025, Problem Set 4
# 
# Problem set 4 is due in Gradescope on **April 10, Thursday at 11:59pm**.
# All the questions are in this jupyter notebook file. There are four questions in this assignment, each of which could have multiple parts and consists of a mix of coding and short answer questions. This assignment is worth a total of **100 points** (**80 pts** coding, and **20 pts** short answer).  There is a bonus question at the end which is worth an extra 10 pts but your maximum final score will be capped at 100 if you scored beyond 100. Note that each individual pset contributes the same amount to the final grade regardless of the number of points it is worth.
# 
# After completing these questions you will need to covert this notebook into a .py file named **ps4.py** and a pdf file named **ps4.pdf** in order to submit it (details below).
# 
# **Submission instructions:** please upload your completed solution files to Gradescope by the due date. **Make sure you have run all code cells and rendered all markdown/Latex without any errors.**
# 
# **THE SUBMISSION IS DUE BY April 10, Thursday at 11:59pm. **
# 
# There will be three separate submission links for the assignment:
# 1. Submit **ps3.py** to `PS4-Code`
# 2. Submit **ONLY your typed code** to `PS4-Typed Code`.
#   + The .py file should contain **ONLY your typed code** (Do not include any other code apart from what you coded for the assignment).
#   + The .py should not contain any written answers. Only the code you wrote.
#   + If your typed code falls under a function definition thats predefined by us, **ONLY include your typed code** and nothing else.
#   + For each cell block within colab/jupyter that you typed your ocde in, Add 2 new lines ("\n") before pasting your typed code in the .py file.
#   + Please name the .py file your actual name.
# 
# 3. Submit a single `.pdf` report that contains your work for all written questions to `PS4`. You can type your responses in LaTeX, or any other word processing software.  You can also hand write them on a tablet, or scan in hand-written answers. If you hand-write, please make sure they are neat and legible. If you are scanning, make sure that the scans are legible. Lastly, convert your work into a `PDF`. You can use Jupyter Notebook to convert the formats:
#   + Convert to PDF file: Go to File->Download as->PDF
#   + Convert py file: Go to File->Download as->py\
# You can take a look at an example [here](https://raw.githubusercontent.com/chaudatascience/cs599_fall2022/master/ps1/convert_py.gif)
# 
#   Your written responses in the PDF report should be self-contained. It should include all the output you want us to look at. **You will not receive credit for any results you have obtained, but failed to include directly in the PDF report file.  Please tag the reponses in your PDF with the Gradescope questions outline  as described in [Submitting an Assignment](https://youtu.be/u-pK4GzpId0). Failure to follow these instructions will result in a loss of points.**

# %% [markdown]
# **Assignment Setup**
# 
# You are strongly encouraged to use [Google Colab](https://colab.research.google.com/) for this assignment.
# 
# If you would prefer to setup your code locally on your own machine, you will need [Jupyter Notebook](https://jupyter.org/install#jupyter-notebook) or [JupyterLab](https://jupyter.org/install#jupyterlab) installation. One way to set it up is to install “Anaconda” distribution, which has Python (you should install python version >= 3.9 as this notebook is tested with python 3.9), several libraries including the Jupyter Notebook that we will use in class. It is available for Windows, Linux, and Mac OS X [here](https://docs.conda.io/en/latest/miniconda.html).
# 
# If you are not familiar with Jupyter Notebook, you can follow [this blog](https://realpython.com/jupyter-notebook-introduction/) for an introduction.  After developing your code using Jupyter, you are encouraged to test it on Google Colab to ensure it works in both settings.
# 
# 
# You cannot use packages other than the ones already imported in this assignment.
# 

# %% [markdown]
# **Jupyter Tip 1**: To run a cell, press `Shift+Enter` or click on "play" button above. To edit any code or text cell [double] click on its content. To change cell type, choose "Markdown" or "Code" in the drop-down menu above.
# 

# %% [markdown]
# **Jupyter Tip 2**: Use shortcut "Shift + Tab" to show the documentation of a function in Jupyter Notebook/ Jupterlab. Press Shift then double Tab (i.e., press Tab twice) to show the full documentation.\
# For example, type `sum(` then Shift + Tab to show the documentation for the function, as shown in this the picture below.

# %%
## import some libraries
import sklearn
from sklearn import datasets
import numpy as np
from typing import Tuple, List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from sklearn import tree
from tqdm import tqdm
import torch
from torch import Tensor
import numpy as np
from typing import Tuple, List, Dict
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from IPython.display import display,Image
import random


# %% [markdown]
# # **Question 1.** Markov Decision Process (*5 total points*)
# The Markov property states that the future depends only on the present and not on the past. The Markov chain is the probabilistic model that solely depends on the current state to predict the next state. In this section, we will join a dice game to determine a MDP.

# %% [markdown]
# ## **1.1 Short answer:** Dice game *(5 pts)*
# 
# Assume you are in Las Vegas, sitting on a playing slot machines for dice game.
# 
# For each round r = 1, 2, . . .  you choose **stay** or **quit**.
# 
#    • If **quit**, you get $10 and we end the game.
# 
#    • If **stay**, you get $4 and then roll a 6-sided dice.
# 
#        – If the dice results in 1 or 2, we end the game.
# 
#        – Otherwise, continue to the next round.
#        
# **Question:** Define states, actions, and determine the transition probabilities T(state, action, new_state) for the game.
# 

# %% [markdown]
# The MDP has two states, $s_0$ where the player is in the game and $s_{end}$ where the game ends. The MDP also has two actions, $stay$ and $quit$. When the player is in $s_0$, they choose between these two actions. If they choose $quit$, the game transitions to $s_{end}$ with probability 1. If they choose $stay$ they recieve $\$4$ and roll a dice. With probability $1/3$ (which corresponds to sides $1$ and $2$) the game ends (transitions to $s_{end}$), while with probability $2/3$ (which corresponds to sides $3$, $4$, $5$ or $6$) the player remains in $s_0$ for the next round.

# %% [markdown]
# # **Question 2.1.** Hidden Markov Models (HMMs) (*2.5 total points*)
# 
# Here is a nice brief tutorial on HMMs: https://web.stanford.edu/~jurafsky/slp3/A.pdf .
# 
# Given a 3x3 grid shown in the figure below
# 
# ![alt text](https://github.com/arijitray1993/arijitray1993.github.io/blob/main/images/HMM_question.png?raw=true?refresh)
# 
# An agent is taking actions to reach the goal and gets a score at the end based on the path it took. For example, If it went up, up, left, left -> it will get a score of -1 + 10 = 9.
# 
# The white blocks have 0 score. The traffic can have -1 or -2. Hitting the wall from a certain block keeps you in the same block, but you get -2. The terminal goal block has +10.
# 
# The actions an agent can take is `{'Up', 'Right'}`
# 
# At each block, an agent decides to take an action determined by the probabilities in the arrows.
# In each case, the "intended" action outcome occurs with probability 0.8, and the other action happens with 0.2 probability.
# 
# 
# ## Given that an agent took some actions (hidden variable), and got a final score of 7. What is the probability that the actions involved hitting a wall along the way?
# 
# 

# %% [markdown]
# Write your answer in this block.
# 
# **Answer:**

# %% [markdown]
# # **Question 2.2.** Hidden Markov Models (HMMs) (*2.5 total points*)
# 
# Based on the below diagram, Should the person move right or not ? Justify your answer elaborately.
# 

# %%
#Image("./Decision.jpg")

# %% [markdown]
# Write your answer in this block.
# 
# **Answer:**

# %% [markdown]
# # **Question 3.** Text Denoising using HMM (*45 total points*)
# 

# %% [markdown]
# In this question we will correct text errors using a hidden Markov model. To obtain text we can obtain copyright-free book in plain characters from Project Gutenberg. For this question we will be looking at Saddle room songs and hunting ballads by Frederick C. Palmer. You can download the txt version at https://www.gutenberg.org/cache/epub/74589/pg74589.txt

# %% [markdown]
# ## **3.1 Code:**  Preprocess Text *(5 pts)*
# Let us now preprocess the text by dropping all punctuation marks except for spaces, convert all capital
# letters to lower case, and mapping multiple spaces to a single space.
# The preprocessed text will have 27 symbols (26 lower case letters and the space character).

# %%
import requests
import re

# Function to download and save the file
def download_file(url, local_filename):
    response = requests.get(url, verify=False)
    response.raise_for_status()  # Ensure the download was successful
    with open(local_filename, 'wb') as f:
        f.write(response.content)
    return local_filename

def question_3_1(file_path: str) -> str:
    '''
    - file_path: path to the txt file downloaded from the URL
    Returns:
    - simplified_text: contains only lowercase letters and single spaces.
    '''
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Write your code in this block ----------------
    text = text.lower()  
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    simplified_text = text



    # End of your code ---------------------------

    return simplified_text

url = "https://www.gutenberg.org/cache/epub/74589/pg74589.txt"
file_path = "pg74589.txt"

# Download the file and save it locally
download_file(url, file_path)

# Process the downloaded file
cleaned_text = question_3_1(file_path)
print(cleaned_text[:200])


# %% [markdown]
# ## **3.2 Code:**  Letter Frequencies *(2 pts)*
# Based on the given text, count unigram, bigram, and trigram letter frequencies.

# %%
from collections import Counter

def question_3_2(text: str, n: int) -> Counter:
    '''
    - text: str, the input text to process
    - n: int, the length of the n-grams to count

    Returns:
    - Counter: a Counter object where keys are n-grams and values are their respective counts
    '''

    # Write your code in this block ----------------
    grams = []
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        grams.append(ngram)
    # End of your code ---------------------------
    return Counter(grams)

unigram_counts = question_3_2(cleaned_text, 1)
unigrams = list(unigram_counts.items())[:10]
print(unigrams)



# %% [markdown]
# ## **3.3 Code:**  Building n-gram Models *(5 pts)*
# Use the counts from question 3.2 build models of unigram, bigram, and trigram letter
# probabilities.

# %%
def question_3_3(ngram_counts: dict,  vocab_size: int) -> dict:
    '''
    - ngram_counts: dict, a dictionary containing n-gram counts
    - total_ngrams: int, the total number of n-grams in the text
    - vocab_size: int, the size of the vocabulary (unique n-grams)

    Returns:
    - ngram_probs: dict, a dictionary with n-grams as keys and their smoothed probabilities as values
    '''

    # Write your code in this block ----------------
    total_ngrams = sum(ngram_counts.values())
    ngram_probs = {}
    for ngram, count in ngram_counts.items():
        # Add-one smoothing
        ngram_probs[ngram] = (count + 1) / (total_ngrams + vocab_size)

    # End of your code ---------------------------
    return ngram_probs

vocab_size=27
unigram_probs = question_3_3(unigram_counts,  vocab_size)
print("Unigram probabilities:", list(unigram_probs.items())[:10])


# %% [markdown]
# ## **3.4 Code:**  Corrupt Input Text *(5 pts)*
# Now let us corrupt the input text through the following process: with probability $P_c$ we will replace a character with a randomly selected character,else we will keep the original character with probability $1-P_c$.

# %%
import random
import string

def question_3_4(text: str, pc: float, vocab: list, seed: int) -> str:
    '''
    Arguments:
    - text: str, the input text to be corrupted
    - pc: float, the probability of replacing each character
    - vocab: list, the vocabulary of characters to choose from when replacing text

    Returns:
    - corrupted_text:the corrupted version of the input text
    '''

    # Write your code in this block ----------------

    # Set the random seed for reproducibility
    random.seed(seed)
    corrupted_text = ""

    for char in text:
        if random.random() < pc:
            replacement = random.choice(vocab)
            corrupted_text += replacement
        else: corrupted_text += char
    return corrupted_text
    # End of your code ---------------------------


vocab = list(string.ascii_lowercase + ' ')
seed=42

pc=0.1
original_text = "this is just an example"
corrupted_text = question_3_4(original_text, pc, vocab,seed)
print("Original text:", original_text)
print("Corrupted text:", corrupted_text)

# %% [markdown]
# ## **3.5 Code:**  Building transition matrix for HMM *(5 pts)*
# Now you will build a transition matrix for a Hidden Markov Model (HMM), which is used to model the probabilities of transitioning from one character to another in a sequence of text. This matrix is essential for recovering sequence of true characters given some corrupted or noisy observations.

# %%
import numpy as np
def question3_5(bigram_counts: dict, vocab: list) -> np.ndarray:
    '''
    Arguments:
    - bigram_counts: dict, a dictionary where keys are bigrams and values are the counts of those bigrams
    - vocab:a list of all possible characters

    Returns:
    - transition_matrix: shape (vocab_size, vocab_size) where each entry represents the
                          probability of transitioning from one character to another
    '''

    # Write your code in this block ----------------
    # Initialize transition matrix
    vocab_size = len(vocab)
    transition_matrix = np.zeros((vocab_size, vocab_size))
    char_to_index = {char: i for i, char in enumerate(vocab)}

    # Calculate total bigram counts for each character
    char_totals = {char: 0 for char in vocab}
    for bigram, count in bigram_counts.items():
        first_char = bigram[0]
        char_totals[first_char] += count
    # Fill transition matrix with bigram probabilities
    for bigram, count in bigram_counts.items():
        if len(bigram) == 2:
            first_char = bigram[0]
            second_char = bigram[1]
            i = char_to_index[first_char]
            j = char_to_index[second_char]
            total_count = char_totals.get(first_char, 0)  
            transition_matrix[i, j] = count / (total_count + vocab_size)

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = transition_matrix / row_sums


    # End of your code ---------------------------
    return transition_matrix




vocab = list(string.ascii_lowercase + ' ')
bigram_counts = question_3_2(cleaned_text, 2)
transition_matrix = question3_5(bigram_counts, vocab)
print(transition_matrix[0])


# %% [markdown]
# 
# ## **3.6 Code:**  Building emission matrix for HMM *(5 pts)*
# We now need to build an emission matrix for a Hidden Markov Model (HMM). The emission matrix represents the probability of observing a certain character given the true underlying (hidden) character in a sequence of text.
# 
# The diagonal of the matrix should represent the probability that a character remains unchanged. The off-diagonal entries will represent the probability that a character is corrupted into another character.

# %%
def question3_6(vocab_size: int, corruption_prob: float) -> np.ndarray:
    '''
    Arguments:
    - vocab_size: int, the number of characters in the vocabulary
    - corruption_prob: float, the probability of a character being corrupted

    Returns:
    - emission_matrix: shape (vocab_size, vocab_size) where each entry represents the probability of observing one character given the hidden character
    '''

    # Write your code in this block ----------------
    emission_matrix = np.zeros((vocab_size, vocab_size))
    off_diagonal = corruption_prob / (vocab_size - 1)
    for i in range(vocab_size):
        for j in range(vocab_size):
            if i == j:
                emission_matrix[i, j] = 1 - corruption_prob
            else:
                emission_matrix[i, j] = off_diagonal
    # End of your code ---------------------------
    return emission_matrix
pc=0.1
emission_matrix = question3_6(len(vocab), pc)
print(emission_matrix[0])

# %% [markdown]
# ## **3.7 Code:**  Converting Text to Indices Based on Vocabulary *(2 pts)*
# Write a function that takes a string of text and a vocabulary list. The function should convert each character in the input text into its corresponding index in the vocabulary list.

# %%
def question3_7(text: str, vocab: list) -> list:
    '''
    Arguments:
    - text: str, the input text to be converted into indices
    - vocab: list, the list of characters in the vocabulary

    Returns:
    - indices: a list of integers where each integer is the index of the corresponding character in the vocabulary
    '''

    # Write your code in this block ----------------
    char_to_index = {char: i for i, char in enumerate(vocab)}
    results = []
    for char in text:
        if char in char_to_index:
            results.append(char_to_index[char])


    # End of your code ---------------------------
    return results

# %% [markdown]
# ## **3.8 Code:**  Converting Indices to Text Based on Vocabulary *(2 pts)*
# 
# 
# Write a function that takes a list of indices and converts them back into a string using the provided vocabulary. Each index in the list corresponds to a character in the vocabulary.

# %%
def question3_8(indices: list, vocab: list) -> str:
    '''
    Arguments:
    - indices: list of integers, where each integer is an index in the vocabulary
    - vocab: list of characters in the vocabulary

    Returns:
    - A string where each character corresponds to the index from the vocabulary
    '''

    # Write your code in this block ----------------
    result = ""
    for i in indices:
        if i < len(vocab):
            result += vocab[i]

    # End of your code ----------------------------

    return result

# %% [markdown]
# ## **3.9 Code:**  One-Hot Encoding Observations for HMM  *(2 pts)*
# write a function that takes a list of indices and the size of the vocabulary, and returns a one-hot encoded matrix.
# 

# %%
import numpy as np

def question3_9(indices: list, vocab_size: int) -> np.ndarray:
    '''

    - indices: list of integers, where each integer is an index in the vocabulary
    - vocab_size: the size of the vocabulary

    Returns:
    - A 2D numpy array representing one-hot encoded vectors with shape (len(indices), vocab_size)
    '''

    # Write your code in this block ----------------
    one_hot_encoded = np.zeros((len(indices), vocab_size))
    for i, index in enumerate(indices):
        if index < vocab_size:
            one_hot_encoded[i][index] = 1


    # End of your code ----------------------------

    return one_hot_encoded


# %% [markdown]
# ## **3.10 Code:**  Training HMM *(5 pts)*
# In this question we will train a Hidden Markov Model (HMM) using bigram counts from a given text to decode the corrupted text.
# 
# We will use hmmlearn library’s **MultinomialHMM** class to create the model.

# %%
# !pip install hmmlearn

# %%
from hmmlearn import hmm
def question3_10(original_text: str,bigram_counts:dict, vocab: list, pc: float) -> hmm.MultinomialHMM:
    '''
    Arguments:
    - original_text: str, the original text from which to train the HMM
    - vocab: list, the list of characters in the vocabulary
    - pc: float, the probability of character corruption (used for the emission matrix)

    Returns:
    - model: hmm.MultinomialHMM, the trained Hidden Markov Model
    '''

    # Write your code in this block ----------------

    # build the transion matrix and emission matrix. You can use question3_5 and question3_6
    transition_matrix = question3_5(bigram_counts, vocab)
    emission_matrix = question3_6(len(vocab), pc)

    # Initialize the HMM model using  hmm.MultinomialHMM and set n_trials to 1
    model = hmm.MultinomialHMM(n_components=len(vocab), n_trials=1)
    # Initialize with with uniform start probability
    start_prob = np.ones(len(vocab)) / len(vocab)
    model.startprob_ = start_prob
    model.transmat_ = transition_matrix
    model.emissionprob_ = emission_matrix


    # End of your code ---------------------------
    return model

original_text=cleaned_text
corrupted_text =question_3_4(original_text, 0.1, vocab,42)
bigram_counts = question_3_2(original_text, 2)
model = question3_10(original_text,bigram_counts, vocab, 0.1)

# Convert corrupted text to indices and one-hot encode it
corrupted_indices = question3_7(corrupted_text, vocab)
one_hot_corrupted = question3_9(corrupted_indices, len(vocab))

# Decode the corrupted text
_, recovered_indices = model.decode(one_hot_corrupted)
recovered_text = question3_8(recovered_indices, vocab)

# %% [markdown]
# ## **3.11 Code:** Estimating Error Rate Using Hamming Distance *(2 pts)*
# We will estimate the error rate between an original text and a recovered (corrupted or noisy) version of that text using the Hamming distance. The Hamming distance between two sequences is the total number of locations in which the values are different.
# 
# Hamming distance is expressed as follows:
# $$
# d_h(x, x(u)) = \sum_{k} \mathbb{1}(x_k \neq x(u)_k)
# $$
# 

# %%
def question3_11(original_text: str, recovered_text: str) -> float:
    '''
    - original_text: str, the original text
    - recovered_text: str, the recovered text

    Returns:
    - error_rate: float, the error rate as the proportion of differing characters
    '''
    # Write your code in this block ----------------
    minimum_length = min(len(original_text), len(recovered_text))
    hamming_distance = 0
    for i in range(minimum_length):
        if original_text[i] != recovered_text[i]:
            hamming_distance += 1
    error_rate = hamming_distance / minimum_length



    # End of your code ---------------------------

    return error_rate
error_rate =question3_11(original_text, recovered_text)
print(f"Error rate: {error_rate * 100:.2f}%")

# %% [markdown]
# ## **3.12 Short answer:** Estimate Error Rate *(5 pts)*
# 
# For $p_c = 0.01$ and $p_c = 0.1$ , estimate the respective error rate for the corrected text. Does the result make sense? Why or why not?
# 

# %% [markdown]
# Write your answer in this block
# 
# For $p_c = 0.01$, the estimated error rate is $0.86\%$, while for $p_c = 0.1$, it increases to $8.7\%$. This result makes sense because a higher corruption probability leads to more noise in the text, which also makes recovery more difficult. Although the HMM may be able to correct some errors, its accuracy decreases as the corruption becomes higher.
# 

# %% [markdown]
# # **Question 4.** Creating Dataloaders in Pytorch (*5 total points*)

# %% [markdown]
# This homework will introduce you to [PyTorch](https://pytorch.org), currently the fastest growing deep learning library, and the one we will use in this course.
# 
# **Before starting the homework, please go over these introductory tutorials on the PyTorch webpage**:
# [60-minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
# 
# 

# %% [markdown]
# ## **4.1 Code:** Data Loader *(5 pts)*
# 

# %% [markdown]
# For the following sections, we will work on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
# 
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
# 
# First, let's download and preprocess the dataset

# %%
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader

transform = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


batch_size = 16

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %% [markdown]
# Let's show some of the pictures in the dataset.

# %%
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
sample = iter(trainloader)
images, labels = next(sample)

# show some images
imshow(torchvision.utils.make_grid(images))

# print labels for the first row
print(' '.join('%4s' % classes[labels[j]] for j in range(8)))

# %% [markdown]
# Similar to the example above, complete the function below to create a "testloader"

# %%
def question_4_2(transform: transforms.Compose, batch_size:int, shuffle:bool, drop_last:bool) -> DataLoader:
    """
    Similar to the example above, create then return a DataLoader for testset
    """
    # Write your code in this block -----------------------------------------------------------
    ## Step 1: create `testset` using datasets.CIFAR10: similar to `trainset` above, with
    # download=True, transform=transform but set `train` to False
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    ## Step 2: create `testloader` using DataLoader and passing params `batch_size`, `shuffer`, `drop_last`
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=2, drop_last=drop_last)
    ## Step 3: return `testloader`
    return testloader
    # End of your code -----------------------------------------------------------


shuffle = False
drop_last = True
testloader = question_4_2(transform,  batch_size, shuffle, drop_last)
testloader

# %% [markdown]
# # **Question 5.** Training Models in Pytorch (*25 total points*)
# 
# 
#  In this problem, we build a 2-layer MLP to predict the class for the images.
# 
# If you're using Colab, you can use its GPU to speed up the training. To opt to GPU, go to tab [Runtime -> Change Runtime type](https://raw.githubusercontent.com/chaudatascience/cs599_fall2022/master/ps4/gpu.png) -> Choose GPU for Hardware accelerator.
# 

# %% [markdown]
# ## **5.1 Code:** 2-layer MLP *(5 pts)*
# Make a subclass of the `Module` class, called `MyMLP`.
# You will need to define the `__init__` and `forward` methods. Our neuron would have 2 linear layers and 1 ReLU layer.
# 
# You can refer to the [Neural Networks tutorial of Pytorch](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html).
# 

# %%
import torch.nn as nn

class MyMLP(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
      """
        in_dim: input dimension, usually we flatten 3d images  (num_channels, width, height) to 1d (num_channels * width * height),
              so we have in_dim = num_channels * width * height
        hid_dim: hidden dimension
        out_dim: output dimension
      """
      super().__init__()

      ## Complete the code below to initilaize self.linear_1, self.linear_2, self.relu
      # where self.linear_1 and self.linear_2 are `nn.Linear` objects, shape of (in_dim, hid_dim) and (hid_dim, out_dim) respectively,
      # and self.relu is a `nn.ReLU` object.
      # Write your code in this block -----------------------------------------------------------
      self.linear_1 = nn.Linear(in_dim, hid_dim)
      self.relu = nn.ReLU()
      self.linear_2 = nn.Linear(hid_dim, out_dim)
      # End of your code ------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
      ## Assume we want to build a model as following: input `x` -> linear_1 -> relu -> linear_2
      ## Write your forward pass
      # Write your code in this block -----------------------------------------------------------
      x = self.linear_1(x)
      x = self.relu(x)
      x = self.linear_2(x)
      return x
      # End of your code ------------------------------------------------------------------------


# %% [markdown]
# ## **5.2 Code:** Train 1 epoch *(5 pts)*
# 
# 
# For each batch in the training set, we use 5 steps to train a Pytorch model
# You will need to fill in the steps below.

# %%

def question_5_2_train_one_epoch(model: nn.Module, trainloader: DataLoader, device:torch.device,
                                 optimizer: torch.optim.SGD, criterion: torch.nn.CrossEntropyLoss, batch_size: int, flatten: bool):
    """
      Train 1 epoch on trainloader. You need to fill in after "##### [YOUR CODE]"
    """

    ## Set model to "train" model
    model = model.train()

    ## Keep track of loss and accuracy
    train_loss = 0.0
    train_acc = 0.0

    ## Loop over all the batches
    for i, (images, labels) in tqdm(enumerate(trainloader, 1), total=len(trainloader), desc=f"training 1 epoch..."):
        # For each batch, we have:
        #     + `images`: `bath_size` images in training set
        #     + `labels`: labels of the images (`batch_size` labels)


        ## Reshape the input dimension if we use MLP: instead of 3d (num_channels, width, height),
        # we flatten it to 1d (num_channels * width * height)
        if flatten:
            images = images.reshape(batch_size, -1)

        ## Move images and labels to `device` (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)

        # Write your code in this block -------------------------------------------------------------------------------------------
        ## We use 5 following steps to train a Pytorch model

        ##### [YOUR CODE] Step 1. Forward pass: pass the data forward, the model try its best to predict what the output should be
        # You need to get the output from the model, store in a new variable named `logits`
        logits = model(images)

        ##### [YOUR CODE] Step 2. Compare the output that the model gives us with the real labels
        ## You need to compute the loss, store in a new variable named `loss`
        loss = criterion(logits, labels)


        ##### [YOUR CODE] Step 3. Clear the gradient buffer
        optimizer.zero_grad()

        ##### [YOUR CODE] Step 4. Backward pass: calculate partial derivatives of the loss w.r.t parameters
        loss.backward()

        ##### [YOUR CODE] Step 5. Update the parameters by stepping in the opposite direction from the gradient
        optimizer.step()

        # End of your code --------------------------------------------------------------------------------------------------------
        ## Compute loss and accuracy for this batch
        train_loss += loss.detach().item()
        train_acc += question_5_3_compute_accuracy(logits, labels, batch_size)

    return train_loss/i, train_acc/i ## avg loss and acc over all batches


# %% [markdown]
# ## **5.3 Code:**  Compute accuracy *(5 pts)*
# 

# %%
## compute accuracy score in a batch
def question_5_3_compute_accuracy(logits: Tensor, labels: Tensor, batch_size: int) -> float:
    '''
      Obtain accuracy for a training batch
      logits: float Tensor, shape (batch_size, num_classes),  output from the model
      labels: Long Tensor, shape (batch_size, ), contains labels for the predictions
      batch_size: int, batch size

      Return accuracy for this batch, which should be a float number in [0, 100], NOT a Tensor
    '''

    # Write your code in this block ----------------
    predictions = torch.argmax(logits, dim=1)
    correct_predictions = (predictions == labels).sum().item()
    accuracy = (correct_predictions / batch_size) * 100
    return accuracy
    # End of your code ---------------------------


# %% [markdown]
# ## **5.4 Code:**  evaluate *(5 pts)*
# We will write a function to evaluate our model on test set after each epoch

# %%
## Note that we use `torch.no_grad()` here to disable gradient calculation.
# It will reduce memory consumption as we don't need to compute gradients in inference.

@torch.no_grad()
def question_5_4_evaluate(model: nn.Module, testloader: DataLoader, criterion, batch_size, device, flatten: bool):
    """
    You need to fill in after "##### [YOUR CODE]"
    """

    test_acc = 0.0
    test_loss = 0.0

    ## Turn on the evaluation mode
    model.eval()

    ## Loop through each batch on test set
    for i, (images, labels) in enumerate(testloader, 1):

        ## Flatten the image into 1d if using MLP
        if flatten:
            images = images.reshape(batch_size, -1)

        # Write your code in this block -----------------------------------------------------------

        ##### [YOUR CODE] Move data to `device`
        images = images.to(device)
        labels = labels.to(device)
        ##### [YOUR CODE] forward pass to get the output of the model
        logits = model(images)

        ##### [YOUR CODE] Compute the loss
        loss = criterion(logits, labels)

        #### [YOUR CODE]  Compute accuracy (re-use question 5.3)
        acc = question_5_3_compute_accuracy(logits, labels, batch_size)
        test_acc += acc        
        
        test_loss += loss.item()
        # End of your code ---------------------------------------------------------------------------

    return test_loss/i, test_acc/i ## avg loss and acc over all batches


# %% [markdown]
# ## **5.5 Code:**  Train model *(5 pts)*
# 
# Let's put everything together. Now we're ready to train a neural network!

# %%
import torch.optim as optim
from time import time
from collections import defaultdict

def question_5_5_train_model(model, device, num_epochs, batch_size, trainloader, testloader, flatten: bool = False):
    """
    model: Our neural net
    device: CPU/GPU
    num_epochs: How many epochs to train
    batch_size: batch size
    train/test loaders: training/test data
    flatten: whether we want to flatten the input image from 3d to 1d

    You need to fill in after "##### [YOUR CODE]"
    """

    # Write your code in this block -------------------------------------------------------------------------------------------

    ##### [YOUR CODE] create your optimizer using `optim.SGD`, set learning rate to 0.001, and momentum to 0.9
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    ##### [YOUR CODE] create criterion using `nn.CrossEntropyLoss`
    criterion = nn.CrossEntropyLoss()

    ## Measure runtime
    t_start = time()

    ## Store training log
    history = defaultdict(list)

    # We will train the model `num_epochs` times
    for i in range(1, num_epochs+1):
        ###### [YOUR CODE] train 1 epoch: call the function in question 5.2
        train_loss, train_acc = question_5_2_train_one_epoch(model, trainloader, device, optimizer, criterion, batch_size, flatten)


        ###### [YOUR CODE] call function in question 5.4 to see how it performs on test set
        test_loss, test_acc = question_5_4_evaluate(model, testloader, criterion, batch_size, device, flatten)

        # End of your code ----------------------------------------------------------------------------------------------------


        ## store train/test loss, accuracy
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        ## print out train/test loss, accuracy
        print(f'Epoch: {i} | Runtime: {((time()-t_start)/60):.2f}[m] | train_loss: {train_loss:.3f} | train_acc: {train_acc:.3f} | test_loss: {test_loss:.3f} | test_acc: {test_acc:.3f}')
    return history

# %% [markdown]
# Let's check again on the shape of each batch
# 

# %%
images.shape ## (batch size, num channels, width, height)

# %%
## Let's create our MLP
in_dim, hid_dim, out_dim = 3*32*32, 256, 10

# set device: We will use GPU if it's available, otherwise CPU
# GPU is much faster to train neural nets
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 10

my_mlp = MyMLP(in_dim, hid_dim, out_dim)

## Move model to `device`: We need everthing (i.e., model, data) on the same device, either CPU, GPU or TPU.
my_mlp.to(device)

# %% [markdown]
# Before training, let take a look at [`torchsummary`](https://pypi.org/project/torch-summary/). It provides a useful tool for a summary of the model.

# %%
from torchsummary import summary
summary(my_mlp, (in_dim, ))

# %% [markdown]
# Our model has 786k trainable parameters.
# 
# Note that **"-1"** in *Output Shape* denotes batch size dimension, which is not a part of the model architecture.

# %%
## Train MLP model
flatten = True
history_mlp = question_5_5_train_model(my_mlp, device, num_epochs, batch_size, trainloader, testloader, flatten)

# %%
## Plot train_loss, test_loss
plt.plot(np.arange(num_epochs), history_mlp["train_loss"], label='train loss')
plt.plot(np.arange(num_epochs), history_mlp["test_loss"], label='test loss')
plt.xlabel('#epochs')
plt.ylabel('loss')
plt.legend()

# %%
## Plot accuracy
plt.plot(np.arange(num_epochs), history_mlp["train_acc"], label='train acc')
plt.plot(np.arange(num_epochs), history_mlp["test_acc"], label='test acc')
plt.xlabel('#epochs')
plt.ylabel('accuracy')
plt.legend()

# %% [markdown]
# # **Question 6.** CNN  (*15 total points*)
# CNNs work well with datasets that have locality (data near each other will be more similar) and compositionality (the object consists of small parts). It has been using as a go-to method for images.
# 
# In this section, we will make a simple CNN model and train it on CIFAR10 dataset.

# %% [markdown]
# ## **6.1 Code:**  CNN model *(10 pts)*
# 
# Now Let's build the 1 CNN model.
# 
# - In our CNN model, we have a convolutional layer denoted by `nn.Conv2d(...)`. We are dealing with an image dataset that is in RGB, so we need 3 channel going in, hence `in_channels=3`. We hope to get a nice representation of this layer, so we use `out_channels=32`. Kernel size is 5, and for the rest of parameters we use the default values which you can find [here](https://pytorch.org/docs/stable/nn.html?highlight=conv2d#conv2d).
# 
# - After each one of CONV or Linear Layers, we also apply an activation function such as `ReLU`.
# 
# Denote: **block_i = (conv_i -> relu -> pool)**
# 
# Let's say we want to build a CNN model as follows:  **input x -> block_1 -> block_2 -> flatten -> fc1 -> relu -> fc2 -> relu -> fc3**  (where fc is linear layer, you can refer to the model summary below)
# 
# You need to complete \_\_init__() (fill in ##### [YOUR CODE])   and forward() of `MyCNN`.

# %%
import torch.nn as nn
import torch.nn.functional as F

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        ## Recall that input images are 3x32x32,
        # i.e., 3 channels (red, green, blue), each of size 32x32 pixels.


        ## An example of conv and pooling layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        # The first convolutional layer, `conv1`, expects 3 input channels,
        # and will convolve 32 filters each of size 3x5x5.
        # Since padding is set to 0 and stride is set to 1 as default,
        # the output size is (32, 28, 28).
        # This layer has ((3*5*5)+1)*32 = 2,432 parameters


        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        ## The first down-sampling layer uses max pooling with a (2,2) kernel
        # and a stride of 2. This effectively drops half of spatial size.

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5)
        ## Similarly, we make another conv layer


        # Write your code in this block --------------------------------------------------------------

        self.fc1 = nn.Linear(16 * 5 * 5, 64)
        ## fc1 is a Linear layer. You'll need to look at the output of conv2 and the input of fc2 to
        # determine the in_dim and out_dim for fc1. You can do this by printing out the shape of the output of conv2 in forward() function.
        # End of your code ---------------------------------------------------------------------------


        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):

        # Implement your forward pass
        # Write your code in this block --------------------------------------------------------------
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # End of your code ---------------------------------------------------------------------------

        return x

# %%
my_cnn = MyCNN().to(device)
summary(my_cnn, images.shape[1:])

# %%
## Train CNN
history_cnn = question_5_5_train_model(my_cnn, device, num_epochs, batch_size, trainloader, testloader)

# %%
## Plot train_loss, test_loss of CNN model
plt.plot(np.arange(num_epochs), history_cnn["train_loss"], label='train loss')
plt.plot(np.arange(num_epochs), history_cnn["test_loss"], label='test loss')
plt.xlabel('#epochs')
plt.ylabel('loss')
plt.legend()

# %%
## Plot accuracy

plt.plot(np.arange(num_epochs), history_cnn["train_acc"], label='train acc')
plt.plot(np.arange(num_epochs), history_cnn["test_acc"], label='test acc')
plt.xlabel('#epochs')
plt.ylabel('accuracy')
plt.legend()

# %% [markdown]
# ## **6.2.1 Short answer:** MLP vs. CNN *(2.5 pts)*
# 
# Compare to the results of MLP and CNN, which model learns better on image dataset - MLP or CNN? Explain your answer.
# 
# Hint: You can discuss on accuracy, runtime, number of parameters, etc.

# %% [markdown]
# Write your answer in this block
# 
# The CNN model learns better on the image dataset compared to the MLP. It achieves a higher test accuracy $\sim67%$ than the MLP $\sim53%$ and a lower test loss $\sim0.94$ vs $\sim1.35$, showing better generalization.
# CNNs took slightly longer to train per epoch, but it’s worth noting that I ran both models locally without using any GPUs, so slower overall training time was expected.

# %% [markdown]
# ## **6.2.2 Short answer:** You are given a MLP model. Briefly descrive in 5 steps as to how you can convert a MLP into a CNN model.  *(2.5 pts)*
# 

# %% [markdown]
# Write your answer in this block
# 
# 1. We should not flatten the input at the start but keep the image as a 3D tensor.
# 
# 2. We should replace the first fully connected layer with a convolutional layer.
# 
# 3. We should add convolutional blocks (like we did earlier Conv2d - Relu - MaxPool2d) to extract features
# 
# 4. After the final convolutional or pooling layer, we should flatten the tensor before feeding it into fully connected layers.
# 
# 5. Last but not least, we should adjust the first linear layer’s input size to match the flattened size, and keep the later fully connected layers for classification.

# %% [markdown]
# ## **6.3  (Bonus) CNN Architecture**. (10 points - Manual grading)
# 
# We have built a simple CNN model, and it works quite well on the dataset.
# Assume we fix the number of epochs (10), optimizer and the loss function, can we improve the CNN architecture to gain more accuracy?
# 
# **One very impactful and simple concept is using residuals:**
# https://arxiv.org/pdf/1512.03385.pdf
# 
# The idea is simple: When stacking multiple layers, we sometimes add in the outputs from a few layers before to the output of a certain layer.
# 
# 

# %%
import requests
from IPython.display import Image, display

url = "https://miro.medium.com/v2/resize:fit:640/format:webp/1*D0F3UitQ2l5Q0Ak-tjEdJg.png"
response = requests.get(url)

display(Image(response.content))


# %% [markdown]
# ### **5 points:** Let's implement a 3-layer CNN with a residual connection.
# The CNN should have 3 conv layers : conv1, conv2, conv3 and be implemented similar to MyCNN in 5.1. Now, add a residual connection from the output of the first conv1 layer to the output of the conv3 layer.
# 
# Hint 1: The output of the Conv1 has 32 channels - it might be benefecial to change that to 16 in this case so that it can be easily added to the output of the conv3 - which also has 16 channels. Even after this, the output shape of conv1
# 
# Hint 2: Note that the convoulution layers reduce the height and width of the input. Hence, the output of the conv3 layer will be smaller (and hence different) from conv_1. We won't be able to add them together with different shapes. Hence, one trick is padding - padding the images with 0's at every input to keep the output dimensions of the convolution layer the same as the input image wihtout the padding. You can do in Pytorch by passing the argument `padding="same"` in the conv2d init.

# %%
class ResidualCNN(nn.Module):
    """
      Feel free to experiment on CNN model.
      You only need to report the model that has the best performance on the dataset.
    """

    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding="same")
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

      self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding="same")
      self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, padding="same")

      self.fc1 = nn.Linear(1024, 64)
      self.fc2 = nn.Linear(64, 32)
      self.fc3 = nn.Linear(32, 10)
      self.relu = nn.ReLU()

    def forward(self, x):
      conv1_out = self.conv1(x)
      x = self.relu(conv1_out)
      x = self.pool(x)

      x = self.conv2(x)
      x = self.relu(x)
      x = self.pool(x)

      residual = self.pool(self.pool(conv1_out))

      x = self.conv3(x)
      x = x + residual
      x = self.relu(x)

      x = x.view(x.size(0), -1)
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      x = self.relu(x)
      x = self.fc3(x)
      return x


## You need to:
#   + Fill in the `MyBestCNN` class, use the best model you have
#   + Show the training log (train/test loss and accuracy) of the best model (you should use `question_5_5_train_model()` as the code below)
#   + Discuss about your experiment

### Initialize your model
cnn_ver2 = ResidualCNN()


#### Show the training log.
history_best_cnn = question_5_5_train_model(cnn_ver2, device, num_epochs, batch_size, trainloader, testloader)

# %% [markdown]
# ### **5 points:** Short Answer: Compare to network without the residual connections
# 
# Make the same network below and remove the residual connection. Compare the performance.
# 
# Feel free to change the number of layers of the CNN. Does the residual connection help more with more layers? What about changing the number of residual connections?
# 
# This is an open-ended exploratory answer.

# %%
class NonResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding="same")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding="same")
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, padding="same")
        
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
cnn_ver3 = ResidualCNN()


#### Show the training log.
history_best_cnn = question_5_5_train_model(cnn_ver3, device, num_epochs, batch_size, trainloader, testloader)

# %% [markdown]
# **Your discussion**
# 
# Insert your answer here

# %% [markdown]
# **Congrats! You have reached to the end of Pset4**


