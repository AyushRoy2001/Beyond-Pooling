<p align="center">
  <img width="600" height="400" alt="thumbnail" src="https://github.com/user-attachments/assets/6ceaa052-9518-4f7a-9d57-ce2e2dac48ab" />
</p>


# [AISTATS 2026] Beyond Pooling: Matching for Robust Generalization under Data Heterogeneity

<p align="center">
  <strong>Ayush Roy</strong>¹ &middot; 
  <strong>Rudrasis Chakraborty</strong>² &middot;
  <strong>Lav Varshney</strong>³ &middot;
  <strong>Vishnu Suresh Lokhande</strong>¹
</p>

<p align="center">
  ¹ University at Buffalo, SUNY &bull; ² Lawrence Livermore National Lab (LLNL) &bull; ³ Stony Brook University, SUNY
</p>

## Abstract

Pooling heterogeneous datasets across domains is a common strategy in representation learning, but naive pooling can amplify distributional asymmetries and yield biased estimators, especially in settings where zero-shot generalization is required. We propose a matching framework that selects samples relative to an adaptive centroid and iteratively refines the representation distribution. The double robustness and the propensity score matching for the inclusion of data domains make matching more robust than naive pooling and uniform subsampling by filtering out the confounding domains (the main cause of heterogeneity). Theoretical and empirical analyses show that, unlike naive pooling or uniform subsampling, matching achieves better results under asymmetric meta-distributions, which are also extended to non-Gaussian and multimodal real-world settings. Most importantly, we show that these improvements translate to zero-shot medical anomaly detection, one of the extreme forms of data heterogeneity and asymmetry.

## Installation
Clone the repository and run the following commands.

```bash
conda create --name beyondpooling python=3.10
conda activate beyondpooling
pip install -r requirements.txt
```

## Datasets
Please follow the instruction given in https://github.com/MediaBrain-SJTU/MVFA-AD for downloading the datasets.

## Training
For the first step of domain addition (warm starting), please run the following commands to create the model weights which will be re-utilized in the subsequent steps of domain addition.
```
python train_zero.py --obj Retina_OCT2017 --gpu 'cuda:0' --exclude Chest Histopathology Brain Liver --intra_weight [PLEASE TRY 1.5, 1.0, 0.5] --inter_weight [PLEASE TRY 1.5, 1.0, 0.5] --save_path 'YOUR_MODEL_CHECKPOINT_PATH'
python train_zero.py --obj Retina_RESC --gpu 'cuda:1' --exclude Histopathology Liver Brain Chest --intra_weight [PLEASE TRY 1.5, 1.0, 0.5] --inter_weight [PLEASE TRY 1.5, 1.0, 0.5] --save_path 'YOUR_MODEL_CHECKPOINT_PATH'
python train_zero.py --obj Brain --gpu 'cuda:2' --exclude Liver Chest Retina_OCT2017 Histopathology --intra_weight [PLEASE TRY 1.5, 1.0, 0.5] --inter_weight [PLEASE TRY 1.5, 1.0, 0.5] --save_path 'YOUR_MODEL_CHECKPOINT_PATH'
python train_zero.py --obj Chest --gpu 'cuda:3' --exclude Brain Liver Retina_RESC Retina_OCT2017 --intra_weight [PLEASE TRY 1.5, 1.0, 0.5] --inter_weight [PLEASE TRY 1.5, 1.0, 0.5] --save_path 'YOUR_MODEL_CHECKPOINT_PATH'
python train_zero.py --obj Liver --gpu 'cuda:4' --exclude Histopathology Retina_RESC Chest Retina_OCT2017 --intra_weight [PLEASE TRY 1.5, 1.0, 0.5] --inter_weight [PLEASE TRY 1.5, 1.0, 0.5] --save_path 'YOUR_MODEL_CHECKPOINT_PATH'
python train_zero.py --obj Histopathology --gpu 'cuda:5' --exclude Liver Retina_RESC Brain Retina_OCT2017 --intra_weight [PLEASE TRY 1.5, 1.0, 0.5] --inter_weight [PLEASE TRY 1.5, 1.0, 0.5] --save_path 'YOUR_MODEL_CHECKPOINT_PATH'
```
For the next step of domain addition, please run the following commands.
```
#### ADDITION OF THE SECOND DOMAIN
python train_zero_da.py --obj Retina_OCT2017 --gpu 'cuda:0' --exclude Chest Histopathology Retina_RESC Liver --intra_weight 1.5 --inter_weight 0.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_WARM_START'
python train_zero_da.py --obj Retina_RESC --gpu 'cuda:1' --exclude Retina_OCT2017 Liver Brain Chest --intra_weight 0.5 --inter_weight 1.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_WARM_START'
python train_zero_da.py --obj Brain --gpu 'cuda:2' --exclude Liver Retina_RESC Retina_OCT2017 Histopathology --intra_weight 1.0 --inter_weight 1.0 --load_path 'MODEL_CHECKPOINT_PATH_OF_WARM_START'
python train_zero_da.py --obj Chest --gpu 'cuda:3' --exclude Histopathology Liver Retina_RESC Retina_OCT2017 --intra_weight 1.5 --inter_weight 0.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_WARM_START'
python train_zero_da.py --obj Liver --gpu 'cuda:4' --exclude Histopathology Retina_RESC Brain Retina_OCT2017 --intra_weight 1.5 --inter_weight 0.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_WARM_START'
python train_zero_da.py --obj Histopathology --gpu 'cuda:5' --exclude Liver Retina_RESC Brain Chest --intra_weight 1.0 --inter_weight 1.0 --load_path 'MODEL_CHECKPOINT_PATH_OF_WARM_START'
```
```
#### ADDITION OF THE THIRD DOMAIN
python train_zero_da.py --obj Retina_OCT2017 --gpu 'cuda:0' --exclude Brain Retina_RESC Liver --intra_weight 1.5 --inter_weight 0.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_SECOND_DOMAIN_ADDITION'
python train_zero_da.py --obj Retina_RESC --gpu 'cuda:1' --exclude Retina_OCT2017 Liver Histopathology Chest --intra_weight 0.5 --inter_weight 1.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_SECOND_DOMAIN_ADDITION'
python train_zero_da.py --obj Brain --gpu 'cuda:2' --exclude Liver Retina_RESC Chest Histopathology --intra_weight 1.0 --inter_weight 1.0 --load_path 'MODEL_CHECKPOINT_PATH_OF_SECOND_DOMAIN_ADDITION'
python train_zero_da.py --obj Chest --gpu 'cuda:3' --exclude Histopathology Brain Retina_RESC Retina_OCT2017 --intra_weight 1.5 --inter_weight 0.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_SECOND_DOMAIN_ADDITION'
python train_zero_da.py --obj Liver --gpu 'cuda:4' --exclude Histopathology Chest Brain Retina_RESC --intra_weight 1.5 --inter_weight 0.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_SECOND_DOMAIN_ADDITION'
python train_zero_da.py --obj Histopathology --gpu 'cuda:5' --exclude Liver Retina_OCT2017 Brain Chest --intra_weight 1.0 --inter_weight 1.0 --load_path 'MODEL_CHECKPOINT_PATH_OF_SECOND_DOMAIN_ADDITION'
```
```
#### ADDITION OF THE FOURTH DOMAIN
python train_zero_da.py --obj Retina_OCT2017 --gpu 'cuda:5' --exclude Histopathology Brain Retina_RESC Liver --intra_weight 1.5 --inter_weight 0.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_THIRD_DOMAIN_ADDITION'
python train_zero_da.py --obj Retina_RESC --gpu 'cuda:6' --exclude Retina_OCT2017 Liver Histopathology Brain --intra_weight 0.5 --inter_weight 1.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_THIRD_DOMAIN_ADDITION'
python train_zero_da.py --obj Brain --gpu 'cuda:7' --exclude Retina_OCT2017 Retina_RESC Chest Histopathology --intra_weight 1.0 --inter_weight 1.0 --load_path 'MODEL_CHECKPOINT_PATH_OF_THIRD_DOMAIN_ADDITION'
python train_zero_da.py --obj Chest --gpu 'cuda:5' --exclude Histopathology Brain Retina_RESC Liver --intra_weight 1.5 --inter_weight 0.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_THIRD_DOMAIN_ADDITION'
python train_zero_da.py --obj Liver --gpu 'cuda:6' --exclude Histopathology Chest Brain Retina_OCT2017 --intra_weight 1.5 --inter_weight 0.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_THIRD_DOMAIN_ADDITION'
python train_zero_da.py --obj Histopathology --gpu 'cuda:7' --exclude Liver Retina_OCT2017 Retina_RESC Chest --intra_weight 1.0 --inter_weight 1.0 --load_path 'MODEL_CHECKPOINT_PATH_OF_THIRD_DOMAIN_ADDITION'
```
```
#### ADDITION OF THE FIFTH DOMAIN
python train_zero_da.py --obj Retina_OCT2017 --gpu 'cuda:5' --exclude Histopathology Brain Retina_RESC Chest --intra_weight 1.5 --inter_weight 0.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_FOURTH_DOMAIN_ADDITION'
python train_zero_da.py --obj Retina_RESC --gpu 'cuda:6' --exclude Retina_OCT2017 Chest Histopathology Brain --intra_weight 0.5 --inter_weight 1.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_FOURTH_DOMAIN_ADDITION'
python train_zero_da.py --obj Brain --gpu 'cuda:7' --exclude Retina_OCT2017 Retina_RESC Chest Liver --intra_weight 1.0 --inter_weight 1.0 --load_path 'MODEL_CHECKPOINT_PATH_OF_FOURTH_DOMAIN_ADDITION'
python train_zero_da.py --obj Chest --gpu 'cuda:5' --exclude Histopathology Brain Retina_OCT2017 Liver --intra_weight 1.5 --inter_weight 0.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_FOURTH_DOMAIN_ADDITION'
python train_zero_da.py --obj Liver --gpu 'cuda:6' --exclude Retina_OCT2017 Chest Brain Retina_RESC --intra_weight 1.5 --inter_weight 0.5 --load_path 'MODEL_CHECKPOINT_PATH_OF_FOURTH_DOMAIN_ADDITION'
python train_zero_da.py --obj Histopathology --gpu 'cuda:7' --exclude Brain Retina_OCT2017 Retina_RESC Chest --intra_weight 1.0 --inter_weight 1.0 --load_path 'MODEL_CHECKPOINT_PATH_OF_FOURTH_DOMAIN_ADDITION'
```

## Testing
Modify --obj for testing the required dataset and --save_path to load the model weights saved at your specific local path.
```
python test_zero.py --obj Liver --gpu 'cuda:4' --save_path 'YOUR_MODEL_CHECKPOINT_PATH'
```

## Qualitative Results
<table align="center">
<tr>
<td><img width="400" height="300" src="https://github.com/user-attachments/assets/6128ab14-c9ae-4afd-8867-4f072414be66"></td>
<td><img width="400" height="300" src="https://github.com/user-attachments/assets/9d8c44f2-440a-4af9-9251-ce11d4d4d2fb"></td>
</tr>
</table>

<p align="center">
<img width="705" height="181" alt="ablation" src="https://github.com/user-attachments/assets/578ce8f1-fb62-4cde-96f0-79d0272a9678" />
</p>
<p align="center">
<img width="715" height="500" alt="Screenshot 2026-05-04 180549" src="https://github.com/user-attachments/assets/689f83ee-0f57-4465-85af-f353275b0931" />
</p>

## Acknowledgements
We borrow code from MVFA (https://arxiv.org/pdf/2403.12570) and thank the authors for making the code public.

# Citation
```bibtex
@article{roy2026beyond,
  title={Beyond Pooling: Matching for Robust Generalization under Data Heterogeneity},
  author={Roy, Ayush and Chakraborty, Rudrasis and Varshney, Lav and Lokhande, Vishnu Suresh},
  journal={arXiv preprint arXiv:2602.07154},
  year={2026}
}
```
