# A Lightweight Recurrent Aggregation Network for Satellite Video Super-Resolution
### [**Paper**](https://doi.org/10.1109/jstars.2023.3332449) | [**Dataset**](https://zenodo.org/records/10939736) | [**Models**](https://pan.baidu.com/s/1U1AyTI3mdPzZ-2SalYJjhw)

Codes and dataset for "A Lightweight Recurrent Aggregation Network for Satellite Video Super-Resolution", Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS), 2023.

## Abstract
>Intelligent processing and analysis of satellite video has become one of the research hotspots in the representation of remote sensing, and satellite video super-resolution (SVSR) is an important research direction which can improve the image quality of satellite video. However, existing approaches for SVSR often underutilize a notable advantage inherent to satellite video: the presence of extensive sequential imagery capturing a consistent scene. Presently, the majority of SVSR methods merely harness a limited number of adjacent frames for enhancing the resolution of individual frames, thus resulting in suboptimal information utilization. In response, we introduce the Recurrent Aggregation Network for Satellite Video Super-Resolution (RASVSR). This innovative framework leverages a bidirectional recurrent neural network to propagate extracted features from each frame across the entire video sequence. It relies on an alignment method based on optical flow and deformable convolution (DCN) to realize the alignment of the features, and a Temporal Feature Fusion Module (TFF) to realize effective feature fusion over time. Notably, our research underscores the positive influence of employing lengthier image sequences in SVSR. In the context of RASVSR, with better alignment and fusion, we make the perceptual field of each frame spanning 100 frames of the video, thus acquiring richer information, and information between different images can be complementary. This strategic approach culminates in superior performance compared to alternative methods, as evidenced by a noteworthy 1.15 dB improvement in PSNR, with very few parameters.

## Network  
![image](/assets/overall.png)

## Dataset
### SAT-MTB-VSR
SAT-MTB-VSR is a large-scale dataset for satellite video super-resolution made from original videos of Jilin-1, which is a subset of the satellite video multitasking dataset SAT-MTB.It contains 431 videos, each of which is 100 consecutive frames, of which 413 are used as the training set and 18 as the validation set, and all the 18 validation sets are from different original videos, while the size of the images is 640 × 640. These images are downsampled 4× by bicubic interpolation to get 160 × 160 low-resolution images, thus obtaining the LR-HR training pairs.

### Download
The dataset is publicly available at [zenodo](https://zenodo.org/records/10939736) and [Baidu Netdisk](https://pan.baidu.com/s/1iyhPrpUdyoHZVIj1U5xvUg).

![image](/assets/dataset.png)
## Install
1. Clone the repo

    ```bash
    git clone https://github.com/Alioth2000/RASVSR.git
    ```

1. Install dependent packages

    ```bash
    cd RASVSR
    pip install -r requirements.txt
    ```

1. Install BasicSR<br>
    Please run the following command in the root path of the project to install BasicSR:<br>

    ```bash
    python setup.py develop
    ```
   
## Dataset Preparation
Download the SAT-MTB-VSR dataset from [zenodo](https://zenodo.org/records/10939736) or [Baidu Netdisk](https://pan.baidu.com/s/1iyhPrpUdyoHZVIj1U5xvUg) and unzip it to `datasets/SAT-MTB-VSR/`.<br>
We recommend using the following command to convert the files to lmdb format to speed up training:
```bash
  python scripts/data_preparation/create_lmdb.py
```

## Pretrained Models
Download the pretrained models from [Baidu Netdisk](https://pan.baidu.com/s/1U1AyTI3mdPzZ-2SalYJjhw) and put them in `experiments/pretrained_models/SAT-MTB-VSR/`.

## Training
- Single GPU
    ```
    python basicsr/train.py -opt options/train_RASVSR.yml
    ```
- Multiple GPU
    ```
    CUDA_VISIBLE_DEVICES=0,1 ./scripts/dist_train.sh 2 options/train_RASVSR.yml
    ```

## Test
- Single GPU
    ```
    python basicsr/test.py -opt options/test_RASVSR.yml
    ```
- Multiple GPU
    ```
    CUDA_VISIBLE_DEVICES=0,1 ./scripts/dist_test.sh 2 options/test_RASVSR.yml
    ```

## Results
![image](/assets/table.png)
![image](/assets/compare.png)

## Citation
```
@ARTICLE{10316591,
  author={Wang, Han and Li, Shengyang and Zhao, Manqi},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={A Lightweight Recurrent Aggregation Network for Satellite Video Super-Resolution}, 
  year={2024},
  volume={17},
  number={},
  pages={685-695},
  doi={10.1109/JSTARS.2023.3332449}}

```

## Acknowledgement
This work is built upon [BasicSR](https://github.com/XPixelGroup/BasicSR).