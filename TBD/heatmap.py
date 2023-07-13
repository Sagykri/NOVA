def plot_clusters_distance_heatmap(X, y):
    n_clusters = len(np.unique(y))
    km = KMeans(n_clusters=n_clusters, random_state=SEED).fit(X)
    
    predicted_labels = km.labels_
    dists = euclidean_distances(km.cluster_centers_)


    # Match cluster to label
    le = LabelEncoder()
    labels_true = le.fit_transform(y)

    conf_matrix = confusion_matrix(labels_true, predicted_labels)
    conf_matrix_argmax = conf_matrix.argmax(axis=0)
    y_pred_ = np.array([conf_matrix_argmax[i] for i in predicted_labels])

    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
    cluster_centers_labels = le.inverse_transform(y_pred_[closest])

    # Generate a heatmap from the distances
    # TODO: Drop bad clusters labels (dups)
    dists_df = pd.DataFrame(dists, index=cluster_centers_labels, columns=cluster_centers_labels)

    sns.clustermap(dists_df, cmap='rainbow_r', xticklabels=True, yticklabels=True, figsize=(18,18))
