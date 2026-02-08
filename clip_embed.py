import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix
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
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from parse_options import parse_option

warnings.filterwarnings("ignore")

args = parse_option()
use_cuda = torch.cuda.is_available()
device = torch.device(args.gpu if use_cuda else "cpu")
CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}
CLASS_INDEX_INV = {v: k for k, v in CLASS_INDEX.items()}
dataset_names = list(CLASS_INDEX.keys())

# Global storage for combined t-SNE (4 levels)
all_image_features = [[] for _ in range(4)]  # One list per level
all_labels = [[] for _ in range(4)]          # One list per level
all_dataset_ids = [[] for _ in range(4)]     # One list per level

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def project_to_sphere(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-7
    return X / norms

def plot_combined_tsne(image_features, labels, dataset_ids, dataset_names, level, save_path):
    features = image_features.cpu().numpy()
    labels_arr = labels.cpu().numpy()
    datasets_arr = dataset_ids.cpu().numpy()

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(features)
    tsne_results = project_to_sphere(tsne_results) 

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    cmap = {
        ('Brain', 0): 'blue', ('Brain', 1): 'lightblue',
        ('Liver', 0): 'green', ('Liver', 1): 'lightgreen',
        ('Chest', 0): 'red', ('Chest', 1): 'salmon',
        ('Histopathology', 0): 'purple', ('Histopathology', 1): 'violet',
        ('Retina_RESC', 0): 'orange', ('Retina_RESC', 1): 'gold',
        ('Retina_OCT2017', 0): 'brown', ('Retina_OCT2017', 1): 'peru',
    }

    for name in dataset_names:
        for label in [0, 1]:
            idx = (datasets_arr == dataset_names.index(name)) & (labels_arr == label)
            if np.any(idx):
                ax.scatter(tsne_results[idx, 0], tsne_results[idx, 1], tsne_results[idx, 2],
                           c=cmap.get((name, label), 'gray'), 
                           label=f"{name} - {'Normal' if label == 0 else 'Anomaly'}",
                           alpha=0.3, s=30)

    ax.set_title(f"Combined 3D t-SNE (Level {level+1})")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_zlabel("t-SNE Dim 3")
    ax.legend(loc="best", fontsize='small')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def main():
    setup_seed(args.seed)

    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    text_feature_list = [0]
    with torch.cuda.amp.autocast(), torch.no_grad():
        for i in [1,2,3,-3,-2,-1]:
            text_feature = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[CLASS_INDEX_INV[i]], device)
            text_feature_list.append(text_feature)

    for dataset in dataset_names:
        args.obj = dataset
        print(f"\nProcessing {dataset} dataset...")

        checkpoint = torch.load(os.path.join(f'{args.save_path}', f'{args.obj}.pth'))
        # Skip adapter loading to use raw CLIP features
        # model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
        # model.det_adapters.load_state_dict(checkpoint["det_adapters"])

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

        text_features = text_feature_list[CLASS_INDEX[args.obj]]
        test(args, model, test_loader, text_features)

    for level in range(4):
        features = torch.cat(all_image_features[level], dim=0)
        labels = torch.cat(all_labels[level], dim=0)
        dataset_ids = torch.cat(all_dataset_ids[level], dim=0)
        
        save_path = os.path.join("results", "combined", f"combined_3d_tsne_level_{level+1}.png")
        plot_combined_tsne(features, labels, dataset_ids, dataset_names, level, save_path)

def test(args, seg_model, test_loader, text_features):
    image_features_all_levels = [[] for _ in range(4)]
    gt_list = []

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        y_sca = y.item()

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, _, ori_det_patch_tokens = seg_model(image)
            ori_det_patch_tokens = [p[0, 1:, :] for p in ori_det_patch_tokens]

            for layer in range(len(ori_det_patch_tokens)):
                # Use raw CLIP features without adapter modifications
                ori_det_patch_tokens[layer] /= ori_det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                feat = ori_det_patch_tokens[layer].mean(dim=0)
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-7)
                image_features_all_levels[layer].append(feat)

        gt_list.append(y_sca)

    dataset_id = dataset_names.index(args.obj)
    
    for level in range(4):
        if image_features_all_levels[level]:  # Check if list is not empty
            image_features_tensor = torch.stack(image_features_all_levels[level], dim=0)
            label_tensor = torch.tensor(gt_list)
            dataset_id_tensor = torch.tensor([dataset_id] * len(gt_list))
            
            all_image_features[level].append(image_features_tensor)
            all_labels[level].append(label_tensor)
            all_dataset_ids[level].append(dataset_id_tensor)

if __name__ == '__main__':
    main()