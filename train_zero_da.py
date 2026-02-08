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
from loss import FocalLoss, BinaryDiceLoss, CosineSimilarityLoss
from utils import augment, encode_text_with_prompt_ensemble, encode_text_simple
from prompt import REAL_NAME

import warnings
warnings.filterwarnings("ignore")

from parse_options import parse_option

args = parse_option()

use_cuda = torch.cuda.is_available()
device = torch.device(args.gpu if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}
CLASS_INDEX_INV = {3:'Brain', 2:'Liver', 1:'Retina_RESC', -1:'Retina_OCT2017', -2:'Chest', -3:'Histopathology'}


def check_optimizer_params(optimizer, model, param_group_name):
    optimizer_params = set(p for group in optimizer.param_groups for p in group['params'])
    model_params = set(dict(model.named_parameters()).values())

    if not optimizer_params.issubset(model_params):
        print(f"Warning: {param_group_name} optimizer contains parameters not in the model!")

    print(f"\n{param_group_name} Optimizer is optimizing the following parameters:")
    for name, param in model.named_parameters():
        if param in optimizer_params:
            print(f"  {name} (requires_grad={param.requires_grad})")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def geodesic_update(A, B, t):
    dot_product = torch.dot(A, B)
    u = B - dot_product * A
    u_norm = u / (u.norm() + 1e-7)
    theta = torch.acos(torch.clamp(dot_product, -1+1e-7, 1-1e-7))
    return torch.cos(t * theta) * A + torch.sin(t * theta) * u_norm

def weighted_geodesic_update(C, F, w, t=0.5):
    weighted_F = w * F
    dot_product = torch.dot(weighted_F, C)
    u = weighted_F - dot_product * C
    u_norm = u / (u.norm() + 1e-7)
    theta = torch.acos(torch.clamp(dot_product, -1+1e-7, 1-1e-7))
    return torch.cos(t * theta) * C + torch.sin(t * theta) * u_norm

def main():
    setup_seed(args.seed)
    
    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_mlp_optimizer = torch.optim.Adam(list(model.det_channel_attention_mlp.parameters()), lr=args.learning_rate*args.lr_variance, betas=(0.5, 0.999))
    seg_mlp_optimizer = torch.optim.Adam(list(model.seg_channel_attention_mlp.parameters()), lr=args.learning_rate*args.lr_variance, betas=(0.5, 0.999))

    # load dataset and loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = MedTrainDataset(args.data_path, args.obj, args.img_size, args.batch_size, exclude_classes=args.exclude)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()
    loss_cos = CosineSimilarityLoss()

    text_feature_list = []
    text_feature_list_unlearn = []
    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        for i in [1,2,3,-3,-2,-1]:
            text_feature_unlearn = encode_text_simple(clip_model, REAL_NAME[CLASS_INDEX_INV[i]], device)
            text_feature = encode_text_with_prompt_ensemble(clip_model, "object", device)
            text_feature_unlearn = torch.tensor(text_feature_unlearn, dtype=torch.float16).to(device)
            text_feature = torch.tensor(text_feature, dtype=torch.float16).to(device)
            text_feature_list.append(text_feature)
            text_feature_list_unlearn.append(text_feature_unlearn)

    text_feature_list[CLASS_INDEX[args.obj]].requires_grad = True
    text_feature_list_unlearn[CLASS_INDEX[args.obj]].requires_grad = True
    save_score = 0.0

    model.set_text_features(text_feature_list[CLASS_INDEX[args.obj]])
    for name, param in model.named_parameters():
        param.requires_grad = True

    # Load checkpoint and initialize centroids with requires_grad=False
    checkpoint = torch.load(os.path.join(f'{args.load_path}', f'{args.obj}.pth'))
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])
    model.seg_channel_attention_mlp.load_state_dict(checkpoint["seg_mlp"])
    model.det_channel_attention_mlp.load_state_dict(checkpoint["det_mlp"])

    # PER-LAYER CENTROIDS: Initialize as lists of 4 centroids each
    # seg_normal_centroids = [nn.Parameter(torch.zeros(768).to(device), requires_grad=True) for _ in range(4)]
    # seg_anomaly_centroids = [nn.Parameter(torch.zeros(768).to(device), requires_grad=True) for _ in range(4)]
    # det_normal_centroids = [nn.Parameter(torch.zeros(768).to(device), requires_grad=True) for _ in range(4)]
    # det_anomaly_centroids = [nn.Parameter(torch.zeros(768).to(device), requires_grad=True) for _ in range(4)]
    seg_normal_centroids = [centroid.to(device).requires_grad_(False) for centroid in checkpoint['normal_seg_centroid']]
    seg_anomaly_centroids = [centroid.to(device).requires_grad_(False) for centroid in checkpoint['anomaly_seg_centroid']]
    det_normal_centroids = [centroid.to(device).requires_grad_(False) for centroid in checkpoint['normal_det_centroid']]
    det_anomaly_centroids = [centroid.to(device).requires_grad_(False) for centroid in checkpoint['anomaly_det_centroid']]

    # No of matched samples each epoch
    matched_samples_det = []
    matched_samples_seg = []

    for epoch in range(args.epoch):
        print('epoch', epoch, ':')
        # No of matched samples
        ms_det = 0
        ms_seg = 0

        loss_list = []
        idx = 0
        for (image, image_label, mask, seg_idx) in tqdm(train_loader):
            if idx % (len(train_loader) // 5) == 0:
                score = test(args, model, test_loader, text_feature_list[CLASS_INDEX[args.obj]])
                if score >= save_score:
                    save_score = score
                    ckp_path = f'{args.save_path}/{args.obj}.pth'
                    torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                                'det_adapters': model.det_adapters.state_dict(),
                                'seg_mlp': model.seg_channel_attention_mlp.state_dict(),
                                'det_mlp': model.det_channel_attention_mlp.state_dict(),
                                'normal_det_centroid': det_normal_centroids,
                                'anomaly_det_centroid': det_anomaly_centroids,
                                'normal_seg_centroid': seg_normal_centroids,
                                'anomaly_seg_centroid': seg_anomaly_centroids}, 
                                ckp_path)
                    print(f'best epoch found: epoch {epoch} batch {idx}')
                print('\n')
            idx += 1

            image = image.squeeze(0).to(device)
            seg_idx = seg_idx.item()

            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens, seg_var, det_var, avg_seg_diff, avg_det_diff = model(image)           
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

                skip_sample_det = False
                skip_sample_seg = False

                # image level
                det_loss = 0
                image_label = image_label.squeeze(0).to(device)
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] = det_patch_tokens[layer] / (det_patch_tokens[layer].norm(dim=-1, keepdim=True) + 1e-7)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_feature_list[seg_idx]).unsqueeze(0)    
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)

                    # DETECTION CENTROID HANDLING
                    det_features = det_patch_tokens[layer].mean(dim=0).unsqueeze(0)  # [B, D]
                    det_labels = image_label  # [B]
                    normal_features = det_features[det_labels == 0]
                    anomaly_features = det_features[det_labels == 1]
                    normal_features = normal_features / (normal_features.norm(dim=-1, keepdim=True) + 1e-7) # for hypersphere
                    anomaly_features = anomaly_features / (anomaly_features.norm(dim=-1, keepdim=True) + 1e-7) # for hypersphere
            
                    intra_loss = 0
                    intra_inter_loss = 0
                    if len(normal_features) > 0:
                        if det_normal_centroids[layer].sum() == 0:
                            det_normal_centroid_temp = normal_features.squeeze(0)
                        else:
                            det_normal_centroid_temp = weighted_geodesic_update(det_normal_centroids[layer].data,normal_features.squeeze(0),avg_det_diff,0.5)
                        
                        # Use centroids without gradient tracking
                        normalized = torch.cat((
                            normal_features, 
                            det_normal_centroids[layer].data.unsqueeze(0), 
                            det_anomaly_centroids[layer].data.unsqueeze(0)
                        ), dim=0)
                        normal_features_n = normalized[0]
                        det_normal_centroid_n = normalized[1].squeeze(0)
                        det_anomaly_centroid_n = normalized[2].squeeze(0)
                        
                        #dNN = torch.norm(normal_features_n - det_normal_centroid_n.unsqueeze(0), p=2)
                        #dNN = F.cosine_similarity(normal_features_n, det_normal_centroid_n.unsqueeze(0), dim=-1).mean()
                        dNN = torch.acos(torch.clamp(F.cosine_similarity(normal_features_n, det_normal_centroid_n.unsqueeze(0), dim=-1), min=-1+1e-7, max=1-1e-7)).mean()
                        #dNA = torch.norm(normal_features_n - det_anomaly_centroid_n.unsqueeze(0), p=2)
                        #dNA = F.cosine_similarity(normal_features_n, det_anomaly_centroid_n.unsqueeze(0), dim=-1).mean()
                        dNA = torch.acos(torch.clamp(F.cosine_similarity(normal_features_n, det_anomaly_centroid_n.unsqueeze(0), dim=-1), min=-1+1e-7, max=1-1e-7)).mean()
                        intra_loss += dNN
                        #intra_loss += (1-dNN)
                        intra_inter_loss += dNA
                        if dNN > dNA:
                            skip_sample_det = True
                        else:
                            # Update centroid data without affecting graph
                            det_normal_centroids[layer].data = det_normal_centroid_temp.data
                            ms_det += 1
                    
                    if len(anomaly_features) > 0:
                        if det_anomaly_centroids[layer].sum() == 0:
                            det_anomaly_centroid_temp = anomaly_features.squeeze(0)
                        else:
                            det_anomaly_centroid_temp = weighted_geodesic_update(det_anomaly_centroids[layer].data,anomaly_features.squeeze(0),avg_det_diff,0.5)
                        
                        normalized = torch.cat((
                            anomaly_features, 
                            det_normal_centroids[layer].data.unsqueeze(0), 
                            det_anomaly_centroids[layer].data.unsqueeze(0)
                        ), dim=0)
                        anomaly_features_n = normalized[0]
                        det_normal_centroid_n = normalized[1].squeeze(0)
                        det_anomaly_centroid_n = normalized[2].squeeze(0)
                        
                        #dAA = torch.norm(anomaly_features_n - det_anomaly_centroid_n.unsqueeze(0), p=2)
                        #dAA = F.cosine_similarity(anomaly_features_n, det_anomaly_centroid_n.unsqueeze(0), dim=-1).mean()
                        dAA = torch.acos(torch.clamp(F.cosine_similarity(anomaly_features_n, det_anomaly_centroid_n.unsqueeze(0), dim=-1), min=-1+1e-7, max=1-1e-7)).mean()
                        #dAN = torch.norm(anomaly_features_n - det_normal_centroid_n.unsqueeze(0), p=2)
                        #dAN = F.cosine_similarity(anomaly_features_n, det_normal_centroid_n.unsqueeze(0), dim=-1).mean()
                        dAN = torch.acos(torch.clamp(F.cosine_similarity(anomaly_features_n, det_normal_centroid_n.unsqueeze(0), dim=-1), min=-1+1e-7, max=1-1e-7)).mean()
                        intra_loss += dAA
                        #intra_loss += (1-dAA)
                        intra_inter_loss += dAN
                        if dAA > dAN:
                            skip_sample_det = True
                        else:   
                            det_anomaly_centroids[layer].data = det_anomaly_centroid_temp.data
                            ms_det += 1
                    
                    # inter_loss = torch.norm(
                    #     det_normal_centroids[layer].data.unsqueeze(0) - 
                    #     det_anomaly_centroids[layer].data.unsqueeze(0), 
                    #     p=2
                    # )
                    # inter_loss = F.cosine_similarity(
                    #     det_normal_centroids[layer].data.unsqueeze(0), 
                    #     det_anomaly_centroids[layer].data.unsqueeze(0), 
                    #     dim=-1
                    # ).mean()
                    inter_loss = torch.acos(torch.clamp(F.cosine_similarity(
                        det_normal_centroids[layer].data.unsqueeze(0), 
                        det_anomaly_centroids[layer].data.unsqueeze(0), 
                        dim=-1
                    ), min=-1+1e-7, max=1-1e-7)).mean()
                    intra_inter_loss = 1 / (intra_inter_loss + 1e-6)
                    inter_loss = 1 / (inter_loss + 1e-6)
                    centroid_loss = args.intra_weight * intra_loss + args.inter_weight * intra_inter_loss + args.inter_weight * inter_loss 
                    det_loss += args.ratio * centroid_loss
                
                if seg_idx > 0:
                    # pixel level
                    seg_loss = 0
                    mask = mask.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    
                    for layer in range(len(seg_patch_tokens)):
                        seg_patch_tokens[layer] = seg_patch_tokens[layer] / (seg_patch_tokens[layer].norm(dim=-1, keepdim=True) + 1e-7)
                        anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_feature_list[seg_idx]).unsqueeze(0)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)

                        # SEGMENTATION CENTROID HANDLING
                        if seg_patch_tokens[layer].ndim == 2:
                            num_patches, D = seg_patch_tokens[layer].shape
                            B = 1
                        else:
                            B, num_patches, D = seg_patch_tokens[layer].shape
                        current_mask = mask.clone()
                        while current_mask.ndim > 3:
                            current_mask = current_mask[0]
                        H_patch = int(math.sqrt(num_patches))
                        mask_pooled = F.adaptive_avg_pool2d(current_mask.unsqueeze(1), (H_patch, H_patch)).squeeze(1)
                        patch_labels = mask_pooled.view(B, -1) > 0.5
                        all_tokens = seg_patch_tokens[layer].view(-1, D)
                        all_labels = patch_labels.view(-1)
                        
                        normal_tokens = all_tokens[~all_labels]
                        anomaly_tokens = all_tokens[all_labels]
                        normal_tokens = normal_tokens / (normal_tokens.norm(dim=-1, keepdim=True) + 1e-7)
                        anomaly_tokens = anomaly_tokens / (anomaly_tokens.norm(dim=-1, keepdim=True) + 1e-7)
                        
                        intra_loss_seg = 0
                        intra_inter_loss = 0
                        if len(normal_tokens) > 0:
                            current_normal = normal_tokens.mean(dim=0)
                            if seg_normal_centroids[layer].sum() == 0:
                                seg_normal_centroid_temp = current_normal
                            else:
                                seg_normal_centroid_temp = weighted_geodesic_update(seg_normal_centroids[layer].data,current_normal,avg_seg_diff,0.5)
                            
                            normalized = torch.cat((
                                current_normal, 
                                seg_normal_centroids[layer].data,
                                seg_anomaly_centroids[layer].data
                            ), dim=0)
                            current_normal_n = normalized[0]
                            seg_normal_centroid_n = normalized[1]
                            seg_anomaly_centroid_n = normalized[2]
                            
                            #dNN = torch.norm(current_normal_n - seg_normal_centroid_n, p=2)
                            #dNN = F.cosine_similarity(current_normal_n, seg_normal_centroid_n, dim=-1).mean()
                            dNN = torch.acos(torch.clamp(F.cosine_similarity(current_normal_n, seg_normal_centroid_n, dim=-1), min=-1+1e-7, max=1-1e-7)).mean()
                            #dNA = torch.norm(current_normal_n - seg_anomaly_centroid_n, p=2)
                            #dNA = F.cosine_similarity(current_normal_n, seg_anomaly_centroid_n, dim=-1).mean()
                            dNA = torch.acos(torch.clamp(F.cosine_similarity(current_normal_n, seg_anomaly_centroid_n, dim=-1), min=-1+1e-7, max=1-1e-7)).mean()
                            intra_loss_seg += dNN
                            #intra_loss_seg += (1-dNN)
                            intra_inter_loss += dNA
                            if dNN > dNA:
                                skip_sample_seg = True
                            else:
                                seg_normal_centroids[layer].data = seg_normal_centroid_temp.data
                                ms_seg += 1
                        
                        if len(anomaly_tokens) > 0:
                            current_anomaly = anomaly_tokens.mean(dim=0)
                            if seg_anomaly_centroids[layer].sum() == 0:
                                seg_anomaly_centroid_temp = current_anomaly
                            else:
                                seg_anomaly_centroid_temp = weighted_geodesic_update(seg_anomaly_centroids[layer].data,current_anomaly,avg_seg_diff,0.5)
                            
                            normalized = torch.cat((
                                current_anomaly, 
                                seg_normal_centroids[layer].data,
                                seg_anomaly_centroids[layer].data
                            ), dim=0)
                            current_anomaly_n = normalized[0]
                            seg_normal_centroid_n = normalized[1]
                            seg_anomaly_centroid_n = normalized[2]
                            
                            #dAA = torch.norm(current_anomaly_n - seg_anomaly_centroid_n, p=2)
                            #dAA = F.cosine_similarity(current_anomaly_n, seg_anomaly_centroid_n, dim=-1).mean()
                            dAA = torch.acos(torch.clamp(F.cosine_similarity(current_anomaly_n, seg_anomaly_centroid_n, dim=-1), min=-1+1e-7, max=1-1e-7)).mean()
                            #dAN = torch.norm(current_anomaly_n - seg_normal_centroid_n, p=2)
                            #dAN = F.cosine_similarity(current_anomaly_n, seg_normal_centroid_n, dim=-1).mean()
                            dAN = torch.acos(torch.clamp(F.cosine_similarity(current_anomaly_n, seg_normal_centroid_n, dim=-1), min=-1+1e-7, max=1-1e-7)).mean()
                            intra_loss_seg += dAA
                            #intra_loss_seg += (1-dAA)
                            intra_inter_loss += dAN
                            if dAA > dAN:
                                skip_sample_seg = True
                            else:
                                seg_anomaly_centroids[layer].data = seg_anomaly_centroid_temp.data
                                ms_seg += 1
                        
                        # inter_loss_seg = torch.norm(
                        #     seg_normal_centroids[layer].data - 
                        #     seg_anomaly_centroids[layer].data, 
                        #     p=2
                        # )
                        # inter_loss_seg = F.cosine_similarity(
                        #     seg_normal_centroids[layer].data, 
                        #     seg_anomaly_centroids[layer].data, 
                        #     dim=-1
                        # ).mean()
                        inter_loss_seg = torch.acos(torch.clamp(F.cosine_similarity(
                            seg_normal_centroids[layer].data, 
                            seg_anomaly_centroids[layer].data, 
                            dim=-1
                        ), min=-1+1e-7, max=1-1e-7)).mean()
                        intra_inter_loss = 1 / (intra_inter_loss + 1e-6)
                        inter_loss_seg = 1 / (inter_loss_seg + 1e-6)
                        centroid_seg_loss = args.intra_weight * intra_loss_seg + args.inter_weight * intra_inter_loss + args.inter_weight * inter_loss_seg
                        seg_loss += args.ratio * centroid_seg_loss
                    
                    if skip_sample_seg:
                        loss = torch.zeros(1, device=args.gpu, requires_grad=True)
                    else:
                        loss = seg_loss + det_loss
                        seg_optimizer.zero_grad()
                        det_optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(model.seg_adapters.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(model.det_adapters.parameters(), max_norm=1.0)
                        seg_optimizer.step()
                        det_optimizer.step()

                        det_mlp_optimizer.zero_grad()
                        seg_mlp_optimizer.zero_grad()
                        (loss+det_var+seg_var).backward()
                        det_mlp_optimizer.step()
                        seg_mlp_optimizer.step()

                else:
                    if skip_sample_det:
                        loss = torch.zeros(1, device=args.gpu, requires_grad=True) 
                    else:
                        loss = det_loss
                        det_optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(model.det_adapters.parameters(), max_norm=1.0)
                        det_optimizer.step()

                        det_mlp_optimizer.zero_grad()
                        (loss+det_var).backward()
                        det_mlp_optimizer.step()
                
                loss_list.append(loss.item())

        train_dataset.shuffle_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

        # logs
        print("Loss: ", np.mean(loss_list))
        matched_samples_det.append(ms_det // 4)
        matched_samples_seg.append(ms_seg // 4)
        print(f'Matched Samples Det: {sum(matched_samples_det)}, Seg: {sum(matched_samples_seg)}')
        
def test(args, seg_model, test_loader, text_features):
    gt_list = []
    gt_mask_list = []
    image_scores = []
    segment_scores = []

    for (image, y, mask) in tqdm(test_loader):
        image = image.to(device)
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, ori_seg_patch_tokens, ori_det_patch_tokens, seg_var, det_var, _, _ = seg_model(image)
            ori_seg_patch_tokens = [p[0, 1:, :] for p in ori_seg_patch_tokens]
            ori_det_patch_tokens = [p[0, 1:, :] for p in ori_det_patch_tokens]
            
            # image
            anomaly_score = torch.tensor(0.0, device=args.gpu)
            patch_tokens = ori_det_patch_tokens.copy()
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= (patch_tokens[layer].norm(dim=-1, keepdim=True) + 1e-7)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                anomaly_score += anomaly_map.mean()
                if torch.isnan(anomaly_score).any():
                    anomaly_score = torch.tensor(0.5, device=args.gpu)
            image_scores.append(anomaly_score.cpu())

            # pixel
            patch_tokens = ori_seg_patch_tokens
            anomaly_maps = []
            for layer in range(len(patch_tokens)):
                patch_tokens[layer] /= (patch_tokens[layer].norm(dim=-1, keepdim=True) + 1e-7)
                anomaly_map = (100.0 * patch_tokens[layer] @ text_features).unsqueeze(0)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=args.img_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            final_score_map = np.sum(anomaly_maps, axis=0)
            final_score_map = np.nan_to_num(final_score_map, nan=0.0)
            
            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            segment_scores.append(final_score_map)
        
    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)

    segment_scores = np.array(segment_scores)
    image_scores = np.array(image_scores)
    image_scores = np.nan_to_num(image_scores, nan=0.5)
    segment_scores = np.nan_to_num(segment_scores, nan=0.0)

    segment_scores = (segment_scores - segment_scores.min()) / (segment_scores.max() - segment_scores.min() + 1e-7)
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min() + 1e-7)

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