import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import plotly.graph_objects as go
import pandas as pd

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def compute_cosine_similarities_with_sampling(final_pixels, final_datasets, class_names, max_samples=2000, random_seeds=[42, 123, 456]):
    inter_intra_results = defaultdict(list)
    
    for seed in random_seeds:
        np.random.seed(seed)
        sampled_indices = []

        for ds in class_names:
            ds_indices = np.where(final_datasets == ds)[0]
            if len(ds_indices) > max_samples:
                sampled = np.random.choice(ds_indices, size=max_samples, replace=False)
            else:
                sampled = ds_indices
            sampled_indices.extend(sampled)

        sampled_indices = np.array(sampled_indices)

        sample_feats = final_pixels[sampled_indices]
        sample_datasets = final_datasets[sampled_indices]

        # Normalize features
        norms = np.linalg.norm(sample_feats, axis=1, keepdims=True) + 1e-7
        sample_feats = sample_feats / norms

        ds_sample_idx = {ds: np.where(sample_datasets == ds)[0] for ds in class_names}

        # Intra-dataset similarities
        for ds in class_names:
            idx = ds_sample_idx[ds]
            ds_feats = sample_feats[idx]
            if len(ds_feats) < 2:
                mean_sim = np.nan
            else:
                sim_mat = cosine_similarity(ds_feats)
                triu_indices = np.triu_indices_from(sim_mat, k=1)
                mean_sim = np.mean(sim_mat[triu_indices])
            inter_intra_results[(ds, ds)].append(mean_sim)

        # Inter-dataset similarities
        for i in range(len(class_names)):
            for j in range(i + 1, len(class_names)):
                ds1 = class_names[i]
                ds2 = class_names[j]
                idx1, idx2 = ds_sample_idx[ds1], ds_sample_idx[ds2]
                feats1, feats2 = sample_feats[idx1], sample_feats[idx2]
                if len(feats1) == 0 or len(feats2) == 0:
                    mean_sim = np.nan
                else:
                    sim_mat = cosine_similarity(feats1, feats2)
                    mean_sim = np.mean(sim_mat)
                inter_intra_results[(ds1, ds2)].append(mean_sim)
                inter_intra_results[(ds2, ds1)].append(mean_sim)
    
    # Aggregate results over runs
    agg_results = {}
    for key, values in inter_intra_results.items():
        clean_vals = [v for v in values if not np.isnan(v)]
        if clean_vals:
            agg_results[key] = (np.mean(clean_vals), np.std(clean_vals))
        else:
            agg_results[key] = (np.nan, np.nan)
    return agg_results

def print_similarity_matrix(agg_results, class_names):
    n = len(class_names)
    print("\nCosine Similarity Matrix (Mean ± Std):")
    header = "Dataset  " + " ".join([f"{ds[:6]:>10}" for ds in class_names])
    print(header)
    for ds1 in class_names:
        row = f"{ds1[:6]:<8}"
        for ds2 in class_names:
            mean, std = agg_results.get((ds1, ds2), (np.nan, np.nan))
            if np.isnan(mean):
                row += f"{'-':>10}"
            else:
                row += f"{mean:.3f}±{std:.3f}".rjust(10)
        print(row)

CLASS_NAMES = ['Brain', 'Liver', 'Retina_RESC', 'Retina_OCT2017', 'Chest', 'Histopathology']

class MedicalTestDataset(Dataset):
    def __init__(self, data_root, class_name, img_size=240):
        self.data_paths = []
        self.labels = []

        # Load normal images
        normal_dir = os.path.join(data_root, f'{class_name}_AD', 'test', 'good', 'img')
        if os.path.exists(normal_dir):
            self.data_paths.extend([
                os.path.join(normal_dir, f)
                for f in os.listdir(normal_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.labels.extend([0] * len(self.data_paths))

        # Load anomaly images
        anomaly_dir = os.path.join(data_root, f'{class_name}_AD', 'test', 'Ungood', 'img')
        if os.path.exists(anomaly_dir):
            anomaly_paths = [
                os.path.join(anomaly_dir, f)
                for f in os.listdir(anomaly_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            self.data_paths.extend(anomaly_paths)
            self.labels.extend([1] * len(anomaly_paths))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        img = Image.open(self.data_paths[idx]).convert('RGB')
        return self.transform(img), self.labels[idx]

def extract_features(data_loader, dataset_name):
    """Extract pixels and labels with consistent lengths"""
    pixels = []
    labels = []
    datasets = []

    for images, batch_labels in tqdm(data_loader):
        # Flatten and store
        flattened = images.view(images.size(0), -1).numpy()
        pixels.append(flattened)
        labels.append(batch_labels.numpy())
        datasets.extend([dataset_name] * images.size(0))

    if not pixels:
        return np.empty((0, 1)), np.array([]), []

    return (
        np.concatenate(pixels, axis=0),
        np.concatenate(labels, axis=0),
        np.array(datasets)
    )

def save_legend_as_image(handles, labels, title, save_path, ncol=2):
    """Save legend as a separate image"""
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    legend = ax.legend(handles, labels, loc="center", ncol=ncol, title=title)
    fig.canvas.draw()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_tsne(tsne_results, labels, datasets, save_dir):
    """Create three separate 3D t-SNE visualizations (no legends inside plots, legends saved separately)."""
    assert len(tsne_results) == len(labels) == len(datasets), \
        f"Length mismatch: t-SNE({len(tsne_results)}), labels({len(labels)}), datasets({len(datasets)})"

    df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'z': tsne_results[:, 2],
        'label': labels,
        'dataset': datasets
    })

    marker = 'o'
    alpha = 0.7
    size = 50
    edge_color = 'w'
    linewidth = 0.5

    colors = plt.cm.tab20.colors
    colors_10 = plt.cm.tab10.colors

    # ---------- 1. Combined ----------
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    handles, legend_labels = [], []
    for i, dataset in enumerate(CLASS_NAMES):
        for label in [0, 1]:
            subset = df[(df.dataset == dataset) & (df.label == label)]
            if not subset.empty:
                sc = ax.scatter(
                    subset.x, subset.y, subset.z,
                    c=[colors[i*2 + label]],
                    marker=marker,
                    alpha=alpha,
                    s=size,
                    edgecolors=edge_color,
                    linewidth=linewidth
                )
                handles.append(sc)
                legend_labels.append(f'{dataset} - {"Normal" if label == 0 else "Anomaly"}')

    ax.set_title("Combined 3D t-SNE Visualization")
    plt.savefig(os.path.join(save_dir, 'combined_tsne_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()

    save_legend_as_image(handles, legend_labels, "Dataset - Class", os.path.join(save_dir, "combined_legend.png"), ncol=2)

    # ---------- 2. Normal only ----------
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    norm_df = df[df.label == 0]
    handles, legend_labels = [], []
    for i, dataset in enumerate(CLASS_NAMES):
        subset = norm_df[norm_df.dataset == dataset]
        if not subset.empty:
            sc = ax.scatter(
                subset.x, subset.y, subset.z,
                c=[colors_10[i]],
                marker=marker,
                alpha=alpha,
                s=size,
                edgecolors=edge_color,
                linewidth=linewidth
            )
            handles.append(sc)
            legend_labels.append(dataset)

    ax.set_title("Normal Samples 3D t-SNE Visualization")
    plt.savefig(os.path.join(save_dir, 'normal_tsne_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()

    save_legend_as_image(handles, legend_labels, "Normal Samples", os.path.join(save_dir, "normal_legend.png"), ncol=2)

    # ---------- 3. Anomaly only ----------
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    anomaly_df = df[df.label == 1]
    handles, legend_labels = [], []
    for i, dataset in enumerate(CLASS_NAMES):
        subset = anomaly_df[anomaly_df.dataset == dataset]
        if not subset.empty:
            sc = ax.scatter(
                subset.x, subset.y, subset.z,
                c=[colors_10[i]],
                marker=marker,
                alpha=alpha,
                s=size,
                edgecolors=edge_color,
                linewidth=linewidth
            )
            handles.append(sc)
            legend_labels.append(dataset)

    ax.set_title("Anomaly Samples 3D t-SNE Visualization")
    plt.savefig(os.path.join(save_dir, 'anomaly_tsne_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()

    save_legend_as_image(handles, legend_labels, "Anomaly Samples", os.path.join(save_dir, "anomaly_legend.png"), ncol=2)

def main():
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Setup and data loading
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    all_pixels, all_labels, all_datasets = [], [], []

    for class_name in CLASS_NAMES:
        dataset = MedicalTestDataset(args.data_root, class_name, args.img_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        pixels, labels, datasets = extract_features(loader, class_name)
        if pixels.size > 0:
            all_pixels.append(pixels)
            all_labels.append(labels)
            all_datasets.append(datasets)

    # Concatenate with validation
    try:
        final_pixels = np.concatenate(all_pixels)
        final_labels = np.concatenate(all_labels)
        final_datasets = np.concatenate(all_datasets)
    except ValueError as e:
        print(f"Data concatenation failed: {e}")
        return

    scaler = MinMaxScaler()
    norm_pixels = scaler.fit_transform(final_pixels)
    # Project onto unit hypersphere
    # norm_pixels = final_pixels / np.linalg.norm(final_pixels, axis=1, keepdims=True)
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(norm_pixels)
    plot_tsne(tsne_results, final_labels, final_datasets, args.save_dir)

if __name__ == '__main__':
    main()
