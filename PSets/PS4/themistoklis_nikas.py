text = text.lower()  
text = re.sub(r'[^a-z\s]', '', text)
text = re.sub(r'\s+', ' ', text)
simplified_text = text


grams = []
for i in range(len(text) - n + 1):
    ngram = text[i:i+n]
    grams.append(ngram)


total_ngrams = sum(ngram_counts.values())
ngram_probs = {}
for ngram, count in ngram_counts.items():
    # Add-one smoothing
    ngram_probs[ngram] = (count + 1) / (total_ngrams + vocab_size)


random.seed(seed)
corrupted_text = ""

for char in text:
    if random.random() < pc:
        replacement = random.choice(vocab)
        corrupted_text += replacement
    else: corrupted_text += char
return corrupted_text


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


emission_matrix = np.zeros((vocab_size, vocab_size))
off_diagonal = corruption_prob / (vocab_size - 1)
for i in range(vocab_size):
    for j in range(vocab_size):
        if i == j:
            emission_matrix[i, j] = 1 - corruption_prob
        else:
            emission_matrix[i, j] = off_diagonal


char_to_index = {char: i for i, char in enumerate(vocab)}
results = []
for char in text:
    if char in char_to_index:
        results.append(char_to_index[char])


result = ""
for i in indices:
    if i < len(vocab):
        result += vocab[i]


one_hot_encoded = np.zeros((len(indices), vocab_size))
for i, index in enumerate(indices):
    if index < vocab_size:
        one_hot_encoded[i][index] = 1


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


minimum_length = min(len(original_text), len(recovered_text))
hamming_distance = 0
for i in range(minimum_length):
    if original_text[i] != recovered_text[i]:
        hamming_distance += 1
error_rate = hamming_distance / minimum_length


## Step 1: create `testset` using datasets.CIFAR10: similar to `trainset` above, with
# download=True, transform=transform but set `train` to False
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
## Step 2: create `testloader` using DataLoader and passing params `batch_size`, `shuffer`, `drop_last`
testloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=2, drop_last=drop_last)
## Step 3: return `testloader`
return testloader


self.linear_1 = nn.Linear(in_dim, hid_dim)
self.relu = nn.ReLU()
self.linear_2 = nn.Linear(hid_dim, out_dim)


x = self.linear_1(x)
x = self.relu(x)
x = self.linear_2(x)
return x


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


predictions = torch.argmax(logits, dim=1)
correct_predictions = (predictions == labels).sum().item()
accuracy = (correct_predictions / batch_size) * 100


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


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


criterion = nn.CrossEntropyLoss()


train_loss, train_acc = question_5_2_train_one_epoch(model, trainloader, device, optimizer, criterion, batch_size, flatten)


test_loss, test_acc = question_5_4_evaluate(model, testloader, criterion, batch_size, device, flatten)


self.fc1 = nn.Linear(16 * 5 * 5, 64)


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

