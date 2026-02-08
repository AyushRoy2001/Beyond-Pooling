import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from CLIP.clip import create_model
from utils import encode_text_with_prompt_ensemble
from prompt import REAL_NAME
import matplotlib.patches as mpatches
from matplotlib import cm

# Configuration
CLASS_INDEX_INV = {3: 'Brain', 2: 'Liver', 1: 'Retina_RESC', 
                   -1: 'Retina_OCT2017', -2: 'Chest', -3: 'Histopathology'}
CLASS_IDS = [1, 2, 3, -3, -2, -1]  # Classes to visualize
MODEL_NAME = "ViT-B/16"  # CLIP model name
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "results/text_tsne"  # Output directory

from parse_options import parse_option

args = parse_option()

def load_clip_model():
    """Initialize CLIP model for text embedding"""
    model = create_model(model_name=args.model_name, img_size=args.img_size, device="cuda:0", pretrained=args.pretrain, require_pretrained=True)
    model.eval()
    return model

def get_text_embeddings(clip_model):
    """Get text embeddings for all classes and prompt types"""
    embeddings = {}
    
    with torch.no_grad():
        for class_id in CLASS_IDS:
            class_name = CLASS_INDEX_INV[class_id]
            # Get embeddings for both normal and anomaly prompts
            text_feature = encode_text_with_prompt_ensemble(
                clip_model, REAL_NAME[class_name], DEVICE
            )
            embeddings[class_name] = {'normal': text_feature[0].cpu().numpy(), 'anomaly': text_feature[1].cpu().numpy()}
        
        text_feature_agnostic = encode_text_with_prompt_ensemble(clip_model, "object", DEVICE)
        embeddings["Agnostic"] = {'normal': text_feature_agnostic[0].cpu().numpy(), 'anomaly': text_feature_agnostic[1].cpu().numpy()}
    
    return embeddings

def plot_text_tsne(text_embeddings, save_path):
    """
    Visualize text embeddings in 3D t-SNE space
    
    Args:
        text_embeddings: Dictionary of {class: {'normal': emb, 'anomaly': emb}}
        save_path: Path to save the visualization
    """
    all_embeddings = []
    class_labels = []
    prompt_types = []
    # Collect all embeddings with metadata
    for class_name, embs in text_embeddings.items():
        # Normal prompt
        all_embeddings.append(embs['normal'])
        class_labels.append(class_name)
        prompt_types.append('Normal')
        # Anomaly prompt
        all_embeddings.append(embs['anomaly'])
        class_labels.append(class_name)
        prompt_types.append('Anomaly')
    
    embeddings_array = np.array(all_embeddings)
    print(f"Embeddings shape: {embeddings_array.shape}")
    print(f"Number of points: {len(all_embeddings)}")
    
    # Apply t-SNE with parameters suitable for small datasets
    tsne = TSNE(
        n_components=3,
        random_state=42,
        perplexity=min(4, len(all_embeddings)-1),  # Perplexity must be < n_samples
        init='random',  # Avoid PCA initialization
        n_iter=5000,    # More iterations for better convergence
        learning_rate=200
    )
    tsne_results = tsne.fit_transform(embeddings_array)
    # Project onto unit hypersphere
    tsne_results = tsne_results / np.linalg.norm(tsne_results, axis=1, keepdims=True)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    unique_classes = list(set(class_labels))
    class_colors = cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_to_color = {cls: class_colors[i] for i, cls in enumerate(unique_classes)}
    class_to_color["Agnostic"] = [0, 0, 0, 1]  # Black for agnostic
    for i, (x, y, z) in enumerate(tsne_results):
        class_name = class_labels[i]
        prompt_type = prompt_types[i]
        
        # Use same marker for both prompt types
        marker = 'D'  # Diamond for all points
        size = 120  # Same size for all points
        edgecolor = 'k'  # Black border for all
        linewidth = 1.5
        if class_name == "Agnostic":
            marker = 'X'  # X marker
            size = 200  # Larger size
            linewidth = 2.5
            alpha = 1.0
        else:
            alpha = 0.9
        
        ax.scatter(
            x, y, z, 
            c=[class_to_color[class_name]],
            marker=marker,
            s=size,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha
        )
    legend_elements = []
    # Class legend
    for class_name, color in class_to_color.items():
        marker = 'X' if class_name == "Agnostic" else 'D'
        legend_elements.append(
            plt.Line2D([0], [0], marker=marker, color='w', 
                      markerfacecolor=color, markersize=10, 
                      markeredgecolor='k', label=class_name)
        )
    # Prompt type legend
    legend_elements.append(
        plt.Line2D([0], [0], marker='D', color='w', 
                  markerfacecolor='gray', markersize=10, 
                  markeredgecolor='k', label='Medical Prompts')
    )
    legend_elements.append(
        plt.Line2D([0], [0], marker='X', color='w', 
                  markerfacecolor='black', markersize=10, 
                  markeredgecolor='k', label='Agnostic Prompt')
    )
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)
    # Labels and title
    ax.set_title("3D t-SNE of Medical Text Prompts with Agnostic Embedding", fontsize=16)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_zlabel("t-SNE Dimension 3", fontsize=12)
    # Adjust viewing angle for better perspective
    ax.view_init(elev=20, azim=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Loading CLIP model...")
    clip_model = load_clip_model()
    print("Generating text embeddings...")
    text_embeddings = get_text_embeddings(clip_model)
    save_path = os.path.join(SAVE_DIR, "medical_text_prompts_tsne_with_agnostic.png")
    print(f"Creating t-SNE visualization: {save_path}")
    plot_text_tsne(text_embeddings, save_path)
    print("Visualization complete!")

if __name__ == '__main__':
    main()