import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from dataset.medical_zero import MedTestDataset, MedTrainDataset
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
from CLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import precision_recall_curve
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, encode_text_with_prompt_ensemble
from prompt import REAL_NAME

import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.stats import multivariate_normal
from scipy.linalg import eigh
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import griddata
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")

def geodesic_update(A, B, t):
    dot_product = np.dot(A, B)
    u = B - dot_product * A
    u_norm = u / (np.linalg.norm(u) + 1e-7)
    theta = np.arccos(np.clip(dot_product, -1+1e-7, 1-1e-7))
    return np.cos(t * theta) * A + np.sin(t * theta) * u_norm

def compute_geodesic_centroid(points):
    if len(points) == 0:
        return None
    centroid = points[0].copy()
    for i in range(1, len(points)):
        weight = 1/(i+1)  # Sequential update weight
        centroid = geodesic_update(centroid, points[i], weight)
    return centroid

def compute_global_cosine(features, batch_size=1000):
    """Compute average pairwise cosine similarity without full matrix"""
    n = features.shape[0]
    total_sum = 0.0
    total_pairs = 0
    # Process in batches to avoid O(n²) memory
    for i in range(0, n, batch_size):
        start_i = i
        end_i = min(i + batch_size, n)
        batch = features[start_i:end_i]
        # Compute dot products with all features
        batch_dots = batch @ features.T  # [batch_size, n]
        # Remove self-similarities
        for j in range(start_i, end_i):
            batch_dots[j - start_i, j] = 0
        
        total_sum += batch_dots.sum()
        total_pairs += batch_dots.size - (end_i - start_i)  # Account for diagonal removal
    
    return total_sum / total_pairs if total_pairs > 0 else 0.0

def plot_dimension_contributions(image_embeddings, text_features, layer, save_path):
    """
    Plot dimension-wise contribution to cosine similarity with text prompts.
    
    Args:
        image_embeddings: Tensor of shape (n_samples, embed_dim)
        text_features: Tensor of shape (embed_dim, 2) [good, bad]
        layer: Layer index for title
        save_path: Path to save the plot
    """
    embed_dim = image_embeddings.shape[1]
    
    # Calculate cosine similarity contributions per dimension
    # Normalize both embeddings
    norm_img = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    norm_text = text_features / text_features.norm(dim=0, keepdim=True)
    
    # Compute contribution of each dimension to cosine similarity
    # (n_samples, embed_dim) * (embed_dim, 2) -> (n_samples, 2)
    # Instead, we want per-dimension contribution: (n_samples, embed_dim, 2)
    dim_contrib = norm_img.unsqueeze(2) * norm_text.unsqueeze(0)
    
    # Average across samples to get mean contribution per dimension
    mean_contrib = dim_contrib.mean(dim=0)  # (embed_dim, 2)
    
    # Calculate the absolute difference between contributions to good vs bad
    diff_contrib = (mean_contrib[:, 0] - mean_contrib[:, 1]).abs()
    
    # Sort dimensions by their contribution difference
    sorted_indices = torch.argsort(diff_contrib, descending=True)
    sorted_contrib = mean_contrib[sorted_indices].cpu().detach().numpy()
    sorted_diff = diff_contrib[sorted_indices].cpu().detach().numpy()
    
    # Create plots
    plt.figure(figsize=(15, 10))
    # Plot contribution to good class
    plt.subplot(2, 1, 1)
    plt.plot(range(embed_dim), mean_contrib[:, 0].cpu().detach().numpy(), 
             'g-', linewidth=2, label='Contribution to "Good"')
    plt.fill_between(range(embed_dim), 0, mean_contrib[:, 0].cpu().detach().numpy(), 
                     color='green', alpha=0.1)
    plt.title(f'Dimension Contribution to Good Class (Layer {layer+1})')
    plt.xlabel('Dimension Index')
    plt.ylabel('Contribution Score')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    # Plot contribution to bad class
    plt.subplot(2, 1, 2)
    plt.plot(range(embed_dim), mean_contrib[:, 1].cpu().detach().numpy(),
             'r-', linewidth=2, label='Contribution to "Bad"')
    plt.fill_between(range(embed_dim), 0, mean_contrib[:, 1].cpu().detach().numpy(),
                     color='red', alpha=0.1)
    plt.title(f'Dimension Contribution to Bad Class (Layer {layer+1})')
    plt.xlabel('Dimension Index')
    plt.ylabel('Contribution Score')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Create sorted plot showing most important dimensions
    plt.figure(figsize=(15, 6))
    # Plot sorted contributions
    plt.subplot(1, 2, 1)
    plt.plot(range(embed_dim), sorted_contrib[:, 0], 'g-', label='Good')
    plt.plot(range(embed_dim), sorted_contrib[:, 1], 'r-', label='Bad')
    plt.plot(range(embed_dim), sorted_diff, 'b-', label='Difference')
    plt.title(f'Sorted Dimension Contributions (Layer {layer+1})')
    plt.xlabel('Dimension Index (Sorted by Importance)')
    plt.ylabel('Contribution Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Plot cumulative contribution difference
    plt.subplot(1, 2, 2)
    total_diff = sorted_diff.sum()
    cum_contrib = np.cumsum(sorted_diff) / total_diff
    plt.plot(range(embed_dim), cum_contrib, 'b-', linewidth=2)
    plt.fill_between(range(embed_dim), 0, cum_contrib, color='blue', alpha=0.1)
    plt.title(f'Cumulative Contribution Difference (Layer {layer+1})')
    plt.xlabel('Number of Dimensions (Sorted by Importance)')
    plt.ylabel('Fraction of Total Contribution Difference')
    # Mark 80% and 90% contribution points
    for percentile in [0.8, 0.9]:
        idx = np.where(cum_contrib >= percentile)[0][0]
        plt.axvline(idx, color='red', linestyle='--')
        plt.text(idx+5, percentile-0.1, f'{idx} dims\n({percentile*100:.0f}%)', color='red')
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_sorted.png'))
    plt.close()

from parse_options import parse_option
args = parse_option()

use_cuda = torch.cuda.is_available()
device = torch.device(args.gpu if use_cuda else "cpu")
CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}
CLASS_INDEX_INV = {3:'Brain', 2:'Liver', 1:'Retina_RESC', -1:'Retina_OCT2017', -2:'Chest', -3:'Histopathology'}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_segmentation_results(image, ground_truth, heatmap, segmented_output, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    image = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    ground_truth = ground_truth.transpose(1, 2, 0)
    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(heatmap, cmap="hot")
    axes[2].set_title("Anomaly Heatmap")
    axes[2].axis("off")

    axes[3].imshow(segmented_output, cmap="gray")
    axes[3].set_title("Segmented Output")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tsne(image_features, text_features, labels, level, save_path_tsne, poly_order=4):
    all_features = torch.cat([image_features, text_features], dim=0).detach().cpu().numpy()
    #all_features = image_features.detach().cpu().numpy()
    scaler = MinMaxScaler()
    all_features = scaler.fit_transform(all_features)

    tsne = TSNE(n_components=3, random_state=36)
    tsne_results = tsne.fit_transform(all_features)

    num_image_samples = len(labels)
    image_coords = tsne_results[:num_image_samples]
    text_coords = tsne_results[num_image_samples:]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['orange' if label == 1 else 'blue' for label in labels]

    ax.scatter(image_coords[:, 0], image_coords[:, 1], image_coords[:, 2],
               c=colors, label='Image Features', alpha=0.6, s=40)
    ax.scatter(text_coords[:, 0], text_coords[:, 1], text_coords[:, 2],
               c=['red', 'green'], label='Text Features', s=120, edgecolors='black', linewidths=1.5)

    # Smooth 2D polynomial fit (Z = f(X, Y))
    X = tsne_results[:, 0]
    Y = tsne_results[:, 1]
    Z = tsne_results[:, 2]

    # Create a design matrix for 2D polynomial fit
    def poly_features(x, y, order):
        terms = []
        for i in range(order + 1):
            for j in range(order + 1 - i):
                terms.append((x**i) * (y**j))
        return np.stack(terms, axis=1)

    Phi = poly_features(X, Y, poly_order)
    model = LinearRegression()
    model.fit(Phi, Z)

    # Evaluate on grid
    xi = np.linspace(np.percentile(X, 1), np.percentile(X, 99), 100)
    yi = np.linspace(np.percentile(Y, 1), np.percentile(Y, 99), 100)
    xi, yi = np.meshgrid(xi, yi)
    phi_grid = poly_features(xi.ravel(), yi.ravel(), poly_order)
    zi = model.predict(phi_grid).reshape(xi.shape)

    # Plot surface
    ax.plot_surface(xi, yi, zi, alpha=0.25, cmap='viridis', linewidth=0, antialiased=True)

    # Clip axis ranges
    ax.set_xlim(np.percentile(X, 1), np.percentile(X, 99))
    ax.set_ylim(np.percentile(Y, 1), np.percentile(Y, 99))
    ax.set_zlim(np.percentile(Z, 1), np.percentile(Z, 99))

    ax.set_title(f"3D t-SNE with Smooth Polynomial Surface (Level {level})")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_zlabel("t-SNE Dim 3")
    plt.tight_layout()
    plt.savefig(save_path_tsne)
    plt.close()

def main():
    setup_seed(args.seed)
    
    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    checkpoint = torch.load(os.path.join(f'{args.save_path}', f'{args.obj}.pth'))
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])
    model.seg_channel_attention_mlp.load_state_dict(checkpoint["seg_mlp"])
    model.det_channel_attention_mlp.load_state_dict(checkpoint["det_mlp"])

    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # load dataset and loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = MedTrainDataset(args.data_path, args.obj, args.img_size, args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    text_feature_list = [0]
    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        for i in [1,2,3,-3,-2,-1]:
            text_feature = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[CLASS_INDEX_INV[i]], device)
            text_feature_list.append(text_feature)

    model.set_text_features(text_feature_list[CLASS_INDEX[args.obj]])
    score = test(args, model, test_loader, text_feature_list[CLASS_INDEX[args.obj]])
        
def test(args, seg_model, test_loader, text_features):
    gt_list = []
    gt_mask_list = []
    image_scores = []
    segment_scores = []
    image_features_all_levels = [[] for _ in range(4)]
    # print(text_features[:10,0])
    # print(text_features[:10,1])
    
    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        y_sca = y.item()
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, ori_seg_patch_tokens, ori_det_patch_tokens, _, _, _, _ = seg_model(image)
            ori_seg_patch_tokens = [p[0, 1:, :] for p in ori_seg_patch_tokens]
            ori_det_patch_tokens = [p[0, 1:, :] for p in ori_det_patch_tokens]

            # Process segmentation features incrementally
            for layer, layer_feat in enumerate(ori_seg_patch_tokens):
                H = int(math.sqrt(layer_feat.size(0)))
            
            for layer in range(len(ori_det_patch_tokens)):
                ori_det_patch_tokens[layer] /= ori_det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                image_features_all_levels[layer].append(ori_det_patch_tokens[layer].mean(dim=0)) # Average pooling
            
            # image
            anomaly_score = 0
            patch_tokens = ori_det_patch_tokens.copy()
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                anomaly_score += anomaly_map.mean()
            image_scores.append(anomaly_score.cpu())

            # pixel
            patch_tokens = ori_seg_patch_tokens
            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.img_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            final_score_map = np.sum(anomaly_maps, axis=0)
            
            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            segment_scores.append(final_score_map)

            if random.random() < 0.1:
                save_path = os.path.join("results", args.obj, "output", f"segmentation_vis_{random.randint(0, 10000)}.png")
                plot_segmentation_results(image[0],
                                        mask[0].cpu().numpy(),
                                        final_score_map[0],
                                        (final_score_map[0] > 0.5).astype(np.int_),
                                        save_path)  

    for level in range(4):
        image_features_level = torch.stack(image_features_all_levels[level], dim=0)
        save_path_tsne = os.path.join("results", args.obj, "tsne_cm", f"tsne_level_{level + 1}.png")
        plot_tsne(image_features_level, text_features.T, gt_list, level + 1, save_path_tsne)

    for level in range(4):
        if image_features_all_levels[level]:
            image_embeddings = torch.stack(image_features_all_levels[level], dim=0)
            save_path_contrib = os.path.join("results", args.obj, "tsne_cm", f"dim_contrib_layer_{level+1}.png")
            plot_dimension_contributions(
                image_embeddings,
                text_features,
                level,
                save_path_contrib
            )
        
    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)

    segment_scores = np.array(segment_scores)
    image_scores = np.array(image_scores)

    segment_scores = (segment_scores - segment_scores.min()) / (segment_scores.max() - segment_scores.min())
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())

    img_roc_auc_det = roc_auc_score(gt_list, image_scores)
    print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')

    if CLASS_INDEX[args.obj] > 0:
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')
        return seg_roc_auc + img_roc_auc_det
    else:
        return img_roc_auc_det

if __name__ == '__main__':
    main()