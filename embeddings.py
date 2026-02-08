import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

from CLIP.clip import create_model
from utils import encode_text_with_prompt_ensemble
from prompt import REAL_NAME
from parse_options import parse_option

# ---------------------------
# Configuration
# ---------------------------
CLASS_NAMES = ['Brain', 'Liver', 'Retina_RESC', 'Retina_OCT2017', 'Chest', 'Histopathology']
CLASS_INDEX_INV = {3: 'Brain', 2: 'Liver', 1: 'Retina_RESC', -1: 'Retina_OCT2017', -2: 'Chest', -3: 'Histopathology'}
CLASS_IDS = [1, 2, 3, -3, -2, -1]
args_parse = parse_option()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Dataset Definition
# ---------------------------
class MedicalTestDataset(Dataset):
    def __init__(self, data_root, class_name, img_size=240):
        self.data_paths = []
        self.labels = []

        normal_dir = os.path.join(data_root, f'{class_name}_AD', 'test', 'good', 'img')
        if os.path.exists(normal_dir):
            self.data_paths.extend([
                os.path.join(normal_dir, f)
                for f in os.listdir(normal_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.labels.extend([0] * len(self.data_paths))

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

# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features(data_loader, dataset_name):
    pixels, labels, datasets = [], [], []
    for images, batch_labels in tqdm(data_loader):
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

# ---------------------------
# Load CLIP and Encode Text
# ---------------------------
def load_clip_model():
    model = create_model(model_name=args_parse.model_name, img_size=args_parse.img_size, device="cuda:0", pretrained=args_parse.pretrain, require_pretrained=True)
    model.eval()
    return model

def get_text_embeddings(clip_model):
    embeddings = {}
    with torch.no_grad():
        for class_id in CLASS_IDS:
            class_name = CLASS_INDEX_INV[class_id]
            text_feature = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[class_name], DEVICE)
            embeddings[class_name] = {
                'normal': text_feature[0].cpu().numpy(),
                'anomaly': text_feature[1].cpu().numpy()
            }
        text_feature_agnostic = encode_text_with_prompt_ensemble(clip_model, "object", DEVICE)
        embeddings["Agnostic"] = {
            'normal': text_feature_agnostic[0].cpu().numpy(),
            'anomaly': text_feature_agnostic[1].cpu().numpy()
        }
    return embeddings

# ---------------------------
# t-SNE Plotting
# ---------------------------
def plot_tsne(tsne_results, labels, datasets, save_dir):
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

    # Combined plot
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, dataset in enumerate(sorted(set(df.dataset))):
        for label in [0, 1]:
            subset = df[(df.dataset == dataset) & (df.label == label)]
            if not subset.empty:
                ax.scatter(
                    subset.x, subset.y, subset.z,
                    c=[colors[i % len(colors)]],
                    marker=marker,
                    label=f'{dataset} - {"Normal" if label == 0 else "Anomaly"}',
                    alpha=alpha,
                    s=size,
                    edgecolors=edge_color,
                    linewidth=linewidth
                )
    ax.legend(bbox_to_anchor=(1.05, 1), ncol=2)
    ax.set_title("Combined 3D t-SNE Visualization")
    plt.savefig(os.path.join(save_dir, 'combined_tsne_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Normal-only
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    norm_df = df[df.label == 0]
    for i, dataset in enumerate(sorted(set(norm_df.dataset))):
        subset = norm_df[norm_df.dataset == dataset]
        if not subset.empty:
            ax.scatter(subset.x, subset.y, subset.z, c=[colors[i % len(colors)]], marker=marker,
                       label=dataset, alpha=alpha, s=size, edgecolors=edge_color, linewidth=linewidth)
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_title("Normal Samples 3D t-SNE Visualization")
    plt.savefig(os.path.join(save_dir, 'normal_tsne_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Anomaly-only
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    anomaly_df = df[df.label == 1]
    for i, dataset in enumerate(sorted(set(anomaly_df.dataset))):
        subset = anomaly_df[anomaly_df.dataset == dataset]
        if not subset.empty:
            ax.scatter(subset.x, subset.y, subset.z, c=[colors[i % len(colors)]], marker=marker,
                       label=dataset, alpha=alpha, s=size, edgecolors=edge_color, linewidth=linewidth)
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_title("Anomaly Samples 3D t-SNE Visualization")
    plt.savefig(os.path.join(save_dir, 'anomaly_tsne_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------
# Main Function
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args_cli = parser.parse_args()

    torch.manual_seed(args_cli.seed)
    os.makedirs(args_cli.save_dir, exist_ok=True)

    all_pixels, all_labels, all_datasets = [], [], []
    for class_name in CLASS_NAMES:
        dataset = MedicalTestDataset(args_cli.data_root, class_name, args_cli.img_size)
        loader = DataLoader(dataset, batch_size=args_cli.batch_size, shuffle=False)
        pixels, labels, datasets = extract_features(loader, class_name)
        if pixels.size > 0:
            all_pixels.append(pixels)
            all_labels.append(labels)
            all_datasets.append(datasets)

    final_pixels = np.concatenate(all_pixels)
    final_labels = np.concatenate(all_labels)
    final_datasets = np.concatenate(all_datasets)

    scaler = MinMaxScaler()
    norm_pixels = scaler.fit_transform(final_pixels)

    print("Loading CLIP model and generating text embeddings...")
    clip_model = load_clip_model()
    text_embeddings = get_text_embeddings(clip_model)

    text_vecs, text_labels, text_datasets = [], [], []
    for class_name, embs in text_embeddings.items():
        text_vecs.append(embs['normal'])
        text_labels.append(0)
        text_datasets.append(class_name + '_text')
        text_vecs.append(embs['anomaly'])
        text_labels.append(1)
        text_datasets.append(class_name + '_text')

    all_vecs = np.concatenate([norm_pixels, np.stack(text_vecs)], axis=0)
    all_lbls = np.concatenate([final_labels, np.array(text_labels)], axis=0)
    all_dsets = np.concatenate([final_datasets, np.array(text_datasets)], axis=0)

    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(all_vecs)

    plot_tsne(tsne_results, all_lbls, all_dsets, args_cli.save_dir)

if __name__ == '__main__':
    main()
