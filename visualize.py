# visualize.py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import numpy as np

def plot_pre_post(vecs_pre, vecs_post, labels, save_path):
    """Visualizes Contextual Transformation (CMNF) with t-SNE."""
    if not os.path.exists(os.path.dirname(save_path) or "results"):
        os.makedirs(os.path.dirname(save_path) or "results")

    combined = np.vstack([np.asarray(vecs_pre), np.asarray(vecs_post)])
    # fix perplexity relative to sample size
    n = combined.shape[0]
    perplex = min(30, max(5, n//3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplex)
    transformed = tsne.fit_transform(combined)
    mid = len(vecs_pre)
    pre_2d = transformed[:mid]
    post_2d = transformed[mid:]

    plt.figure(figsize=(10, 6))
    plt.scatter(pre_2d[:, 0], pre_2d[:, 1], c='blue', label='Original (Generic)', alpha=0.6)
    plt.scatter(post_2d[:, 0], post_2d[:, 1], c='red', label='Projected (Payment Context)', alpha=0.6)

    for i, txt in enumerate(labels[:mid]):
        plt.annotate(txt, (pre_2d[i, 0], pre_2d[i, 1]), fontsize=8)

    plt.title("CMNF Projection: Semantic Context Shift")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
