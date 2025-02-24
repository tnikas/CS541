one_hot = np.zeros((len(sentence), 100))
one_hot[np.arange(len(sentence)), sentence] = 1

rows,columns=arr.shape

array=np.zeros((rows,columns,101))
array[np.arange(rows)[:, None], np.arange(columns)[None, :], arr] = 1

X = data[:, :-1]
y = data[:, -1]
label_mapping = {'Setosa': 2, 'Versicolor': 1, 'Virginica': 0}
y = np.vectorize(label_mapping.get)(y)

X_train = []
y_train = []
X_test = []
y_test = []

classes = np.unique(y)

for label in classes:
    classes_indices = np.where(y == label)[0]
    np.random.shuffle(classes_indices)
    split_idx = int(len(classes_indices) * 0.8)

    train_indices = classes_indices[:split_idx]
    test_indices = classes_indices[split_idx:]

    X_train.append(X[train_indices])
    y_train.append(y[train_indices])
    X_test.append(X[test_indices])
    y_test.append(y[test_indices])

X_train = np.vstack(X_train)
y_train = np.hstack(y_train)
X_test = np.vstack(X_test)
y_test = np.hstack(y_test)

X_train = []
y_train = []
X_test = []
y_test = []
split_ratios = {2: 0.95, 1: 0.50, 0: 0.20} # Virginica: 0, Versicolor: 1, Setosa: 2

for label, ratio in split_ratios.items():
    classes_indices = np.where(y == label)[0]
    np.random.shuffle(classes_indices)
    split_idx = int(len(classes_indices) * ratio)

    train_indices = classes_indices[:split_idx]
    test_indices = classes_indices[split_idx:]

    X_train.append(X[train_indices])
    y_train.append(y[train_indices])
    X_test.append(X[test_indices])
    y_test.append(y[test_indices])

X_train = np.vstack(X_train)
y_train = np.hstack(y_train)
X_test = np.vstack(X_test)
y_test = np.hstack(y_test)

model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
cv_accuracy = cross_val_score(model, X, y, cv=k)
avg_accuracy = np.mean(cv_accuracy)

cv_accuracies = []
for n_neighbors in n_neighbors_list:
    model = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    cv_accuracy = cross_val_score(model, X, y, cv=k)
    avg_accuracy = np.mean(cv_accuracy)
    cv_accuracies.append(avg_accuracy)

plt.figure(figsize=(8, 5))
sns.lineplot(x=n_neighbors_list, y=cv_accuracies, marker='o')

# Formatting
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validation Accuracy")
plt.title(f"K-NN Performance on Iris Dataset ({k}-Fold CV)")
plt.xticks(n_neighbors_list)  # Show all n_neighbors values
plt.grid(True)
plt.show()

return np.sqrt(np.sum((A[:, None] - B) ** 2, axis=2))

distances = self.compute_euclidean_distance(X_test, self.X_train)
nearest_neighbors_idx = np.argsort(distances, axis=1)[:, :self.n_neighbors]
nearest_labels = np.array(self.y_train)[nearest_neighbors_idx]
predictions = np.array([np.bincount(row).argmax() for row in nearest_labels])
return predictions

y_pred = self.predict(X_test)
return np.mean(y_pred == y_test)