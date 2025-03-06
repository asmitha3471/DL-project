from sklearn.decomposition import PCA

def apply_pca(features, n_components=6):
    pca = PCA(n_components=n_components, random_state=10)
    pca_projection = pca.fit_transform(features)
    return pca_projection, pca

# Test script
if __name__ == "__main__":
    from preprocessing import load_and_preprocess
    X, y, _ = load_and_preprocess()
    X_pca, _ = apply_pca(X)
    print("PCA Applied! New Shape:", X_pca.shape)