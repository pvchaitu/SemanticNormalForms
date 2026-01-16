import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def plot_pre_post(vecs_pre, vecs_post, labels, save_path):
    """Visualizes Contextual Transformation (CMNF)."""
    if not os.path.exists('results'):
        os.makedirs('results')
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(labels)-1))
    combined = tsne.fit_transform(list(vecs_pre) + list(vecs_post))
    
    mid = len(vecs_pre)
    pre_2d = combined[:mid]
    post_2d = combined[mid:]

    plt.figure(figsize=(10, 6))
    plt.scatter(pre_2d[:, 0], pre_2d[:, 1], c='blue', label='Original (Generic)', alpha=0.6)
    plt.scatter(post_2d[:, 0], post_2d[:, 1], c='red', label='Projected (Payment Context)', alpha=0.6)
    
    for i, txt in enumerate(labels):
        plt.annotate(txt, (pre_2d[i, 0], pre_2d[i, 1]), fontsize=8)
        
    plt.title("CMNF Projection: Semantic Context Shift")
    plt.legend()
    plt.savefig(save_path)
    plt.close()