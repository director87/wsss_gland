# Weakly Supervised Gland Segmentation with Class Semantic Consistency and Purified Labels Filtration (AAAI2025)

## Abstract
Image-level weakly supervised semantic segmentation (WSSS) reduces the dependence on high-quality data annotation, which plays a crucial role in computational pathology. Benefit from the ability to localize the objects with only binary labels, Class Activation Map (CAM) is a widely used method to initial pseudo masks. However, due to the low contrast among different tissues in histopathological images, most existing CAM-based methods perform poorly in gland segmentation. We retrospect this process and find that class consistency and semantic consistency can guide the network to effectively distinguish confusing pixels and generate fine-grained pseudo masks. Specifically, for class consistency, we propose Consistency Correlation Attention (CCA) to encourage the network to focus on the contribution of class features to semantic dependencies. For semantic consistency, we propose Multi-scale Pyramid Fusion Pooling (MPFP) to aggregate coarse-to-fine global semantic information from CAMs at multiple spatial resolutions, thus identifying class localization. Additionally, we introduce a Purified Labels Filtration (PLF) strategy during the segmentation phase to mitigate the noisy supervision signal and improve the segmentation quality of the model. Extensive experiments show that the our method achieves new state-of-the-art results on three publicly available gland datasets. Furthermore, our method demonstrates impressive domain adaptation capability, achieving satisfactory results with only a small portion of samples when faced with unseen domain data.

## Environment
Our code based on
* Ubuntu 20.04
* NVIDIA RTX 4090 GPU
* Python 3.8
* Pytorch 2.0.1

The necessary packages can be installed through the following command:
```
pip install -r requirements.txt
```

## Datasets
* [ProG](https://data.mendeley.com/datasets/h8bdwrtnr5/1) \[1\]
* [GlaS](https://pan.baidu.com/share/init?surl=htY5nZacceXj_m2FlY8uXw) \[2\] (code: snb3) 
* [EBHI](https://figshare.com/articles/dataset/EBHI-SEG/21540159/1?file=38179080) \[3\]

Download the datasets and put them into `datasets` folder with following formats:
```
|-- datasets  
    |-- RINGS/  
        |-- train/  
        |-- val/  
            |-- img/  
            |-- mask/  
        |-- test/  
            |-- img/  
            |-- mask/  
    |-- GlaS/  
        |-- train/  
        |-- val/  
            |-- img/  
            |-- mask/  
        |-- test/  
            |-- img/  
            |-- mask/
    |-- EBHI/  
        |-- train/  
        |-- val/  
            |-- img/  
            |-- mask/  
        |-- test/  
            |-- img/  
            |-- mask/
```


## Pretrained weights
Download the pretained weight of classification stage from [this link](https://drive.google.com/drive/folders/1oc4BNEREZG2gZ78IzEwZZjC4Y0fTshRC?usp=drive_link), and put it into `init_weights` folder.

## Run the whole pipeline
### 1. Train the classification model
```
python 1_train_stage1.py --dataset ring --trainroot datasets/RINGS/train/ --testroot dataset/RINGS/test/
```
### 2. Generate pseudo masks
```
python 2_generate_PM.py --dataroot datasets/RINGS --dataset ring
```
### 3. Train the segmentation model
```
python 3_train_stage2.py --dataset ring --dataroot datasets/RINGS
```

## References
\[1\] Salvi et al. A hybrid deep learning approach for gland segmentation in prostate histopathological images. In *Artificial Intelligence in Medicine*, 2021

\[2\] Sirinukunwattana et al. Gland segmentation in colon histology images: The glas challenge contest. In *Medical Image Analysis*, 2017

\[3\] Shi et al. EBHI-Seg: A novel enteroscope biopsy histopathological hematoxylin and eosin image dataset for image segmentation tasks. In *Frontiers in Medicine*, 2023

## Contact
If you have any question, please contact <director87@foxmail.com>.

## Citation
If you find this code repository useful, please consider citing our paper:
```
@inproceedings{feng2025weakly,
  title={Weakly Supervised Gland Segmentation with Class Semantic Consistency and Purified Labels Filtration},
  author={Feng, Siyang and Wang, Huadeng and Han, Chu and Liu, Zhenbing and Zhang, Hualong and Lan, Rushi and Pan, Xipeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2987--2995},
  year={2025}
}
```
