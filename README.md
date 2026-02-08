<p align="center">
  <img width="600" height="400" alt="Screenshot 2026-02-08 083947" src="https://github.com/user-attachments/assets/4215d0ea-e33c-4d3d-a469-8b6464c75006" />
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
```
For the next step of domain addition, please run the following commands.
```
```

## Testing
Modify --obj for testing the required dataset and --save_path to load the model weights saved at your specific local path.
```
python test_zero.py --obj Liver --gpu 'cuda:4' --save_path 'YOUR_MODEL_CHECKPOINT_PATH'
```

## Qualitative Results
<img width="400" height="350" alt="variance" src="https://github.com/user-attachments/assets/6128ab14-c9ae-4afd-8867-4f072414be66" />
<img width="300" height="350" alt="mind_the_gap" src="https://github.com/user-attachments/assets/9d8c44f2-440a-4af9-9251-ce11d4d4d2fb" />
<img width="705" height="181" alt="ablation" src="https://github.com/user-attachments/assets/578ce8f1-fb62-4cde-96f0-79d0272a9678" />

## Acknowledgements
We borrow code from MVFA (https://arxiv.org/pdf/2403.12570) and thank the authors for making the code public.

# Citation
```bibtex
@article{roy2025exchangeability,
  title={Is Exchangeability better than IID to handle Data Distribution Shifts while Pooling Data for Data-scarce Medical image segmentation?},
  author={Roy, Ayush and Enam, Samin and Xia, Jun and Lokhande, Vishnu Suresh and Kim, Won Hwa},
  journal={arXiv preprint arXiv:2507.19575},
  year={2025}
}
```
