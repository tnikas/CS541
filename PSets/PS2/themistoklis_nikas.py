m = np.mean(X, axis=0)
s = np.std(X, axis=0)

X_norm = (X - m) / s

return X_norm
cov_matrix = np.cov(X_norm.T)
U, S, V = np.linalg.svd(cov_matrix)
U_reduced = U[:, :n_components]
X_reduced = X_norm @ U_reduced
## return the reduced data
return X_reduced
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X)
cum_var_explained = np.cumsum(pca.explained_variance_ratio_)
dist = np.linalg.norm(samples[:, None] - centroids, axis=2)
cluster_assigns = np.argmin(dist, axis=1)
return cluster_assigns
k = len(np.unique(clusters))
new_centroid = np.zeros((k, samples.shape[1]))
for i in range(k):
    new_centroid[i] = np.mean(samples[clusters == i], axis=0)
return  new_centroid
start_time = time()
kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
kmeans.fit(X)
print(f"runtime: {time() - start_time} seconds")
return kmeans
unique_labels = np.unique(y)
for label in unique_labels:
    class_means[label] = np.mean(X[y == label], axis=0)
y_pred = kmeans.labels_
mapped_pred = np.vectorize(cluster_to_label_map.get)(y_pred)
acc = accuracy_score(y, mapped_pred)
print(np.unique(y_pred), np.unique(y))
print(cluster_to_label_map)  
return acc
reshaped_img = raw_img.reshape(-1, raw_img.shape[-1])
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(reshaped_img)
imaged_scaled = scaler.transform(reshaped_img)
gmm = GMM(n_components=n_components, max_iter=60, covariance_type="tied", random_state=random_seed)
labels = gmm.fit_predict(imaged_scaled)
cluster_centers = gmm.means_
clustered_img = cluster_centers[labels].reshape(original_shape)
return clustered_img
h, w, c = shape
x = np.arange(w)
y = np.arange(h)
xx, yy = np.meshgrid(x, y)
new_img = np.dstack((raw_img, xx, yy))
return new_img
pca = PCA(n_components=var_preserve, random_state=random_seed)
X_reduced = pca.fit_transform(X)
gmm = GMM(n_components=140, random_state=random_seed)
gmm.fit(X_reduced)
X_generated, _ = gmm.sample(n_samples=36)
X_reconstruced = pca.inverse_transform(X_generated)
return X_reconstruced
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
kmeans_labels = kmeans.fit_predict(X)

gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
gmm_labels = gmm.fit_predict(X)