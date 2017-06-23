# cervical-cancer-screening

Top 23% (191st of 848) solution for [Kaggle Intel &amp; MobileODT Cervical Cancer Screening](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening).

## Introduction

In this competition, [Intel](https://www.kaggle.com/intel) is partnering with [MobileODT](http://www.mobileodt.com/) to challenge Kagglers to develop an algorithm which accurately identifies a woman’s cervix type based on images. Doing so will prevent ineffectual treatments and allow healthcare providers to give proper referral for cases that require more advanced treatment.

## Basic Idea with Step by Step Implementation

1. [[pre-processing]](./code/cervix-part-crop-GMM-Method-on-train-additional-train-and-test-set.ipnb) Due to big image (`4000*4000`, etc), I resize the original images to fixed size (`224*224`) and crop out the no-relevant part.
2. [Generate MXNet format binary file of images] Prepare `.lst` and `.rec` files referring [Prepare Datasets | MXNet](https://github.com/dmlc/mxnet/tree/master/example/image-classification#prepare-datasets).
3. [[Train models from scratch]](./train-or-finetune-model/models/) use `run_train_scripts.sh` and `train_ccs-train.py` to train a network from scratch.
4. [[fine-tune models]](./train-or-finetune-model/finetune-models/) use `run_finetune_script.sh`, `run_finetune.py` and `run_finetune_data-aug.py` to fine-tune pre-trained models.
5. [[prepare submission]](./train-or-finetune-model/) use `get_result_for_one_network.sh` and `run_inference.py` scripts to prepare submission based on test set.
6. [[Boosting multi-sub-models and prepare submission]](./code/train-boost-model-based-on-multi-features.ipynb) train a boosting model based on multi-sub-models. Of course, you should select some sub-models from last step and put preparing-boosting sub-models in `./models` directory.


## Repo. Structure

```
├── `code` contains pre-processing step (image crop), boosting Jupyter notebook
├── `intel` contains remote host environments intel provided (useless for me)
├── `model` MXNet models, which prepare to make boosting based on these models
├── `pre-submit` preparing submit file
│   ├── [result]finetune-resnet-152-train-add-seg-224-lr-0.001-momentum-0.9
│   ├── [result]finetune-resnet-200-train-add-seg-224-lr-0.001-momentum-0.9
│   └── [result]inception-resnet-v2-152-train-add-seg-224-lr-0.01
├── `submitted` stage1 submitted file
│   ├── [result]finetune-resnet-101-train-add-seg-224-lr-0.001-momentum-0.9
│   ├── [result]finetune-resnet-152-imagenet-11k-365-ch-train-add-seg-224-lr-0.01
│   ├── [result]finetune-resnet-152-imagenet-11k-train-add-seg-224-lr-0.01
│   ├── [result]finetune-resnet-200-train-add-seg-224
│   ├── [result]finetune-resnet-34-train-add-seg-224-lr-0.01
│   ├── [result]finetune-resnet-50-imagenet-11k-365-ch-train-add-seg-224-lr-0.01
│   ├── [result]finetune-resnet-50-trian-add-seg-224-lr-0.01
│   ├── [result]finetune-resnext-101-train-add-seg-224-lr-0.001-momentum-0.9
│   ├── [result]finetune-resneXt-101-train-add-seg-224-lr-0.01
│   ├── [result]finetune-resnext-50-train-add-seg-224-lr-0.001-momentum-0.9
│   ├── [result]finetune-resneXt-50-train-add-seg-224-lr-0.01
│   ├── [result]inception-resnet-v2-18-train-add-seg-224-lr-0.01
│   ├── [result]inception-resnet-v2-50-train-add-seg-224-lr-0.01
│   ├── [result]inception-resnet-v2-train-add-seg-224-lr-0.01
│   ├── [result]inception-resnet-v2-train-seg-224-lr-0.05
│   ├── [result]inception-resnt-v2-train-224-lr-0.05
│   ├── [result]resnet-50-train-seg-224-lr-0.05
│   └── [result-v2?]inception-resnet-v2-50-train-add-seg-224-lr-0.01
└── `train-or-finetune-model` those train-from-scratched and fine-tuned models
    ├── `finetune-models` those checkpoints and logs of fine-tuning models
    ├── `models` those checkpoints and logs of train-from-scratch models
    ├── `submitted` stage1 submission file
    └── `tmp` useless files
```

## 0. Data

* 3-class classification
* Training set has 1700+ images( type1: 250, type2: 781, type3: 450 ).
* Training + Additional set have 8000+ images ( all type1: 1440,  all type2: 4346, all type3: 2426 ) .

blank files (0 KB) :

1. `additional/Type_1/5893.jpg`
2. `additional/Type_2/5892.jpg`
3. `additional/Type_2/2845.jpg`
4. `additional/Type_2/6360.jpg`

Non-cervix images:

1. `additional/Type_1/746.jpg`
2. `additional/Type_1/2030.jpg`
3. `additional/Type_1/4065.jpg`
4. `additional/Type_1/4706.jpg`
5. `additional/Type_2/1813.jpg`
6. `additional/Type_2/3086.jpg`

## 1. Train from scratch

First, I tried train `MLP`, `LeNet`, `GoogLeNet`, `AlexNet`, `ResNet-50`, `ResNet-152`, `inception-ResNet-v2`, and `ResNeXt` models from scratch based on training and additional data.
*  `MLP` and `LeNet` doesn't converge in 30 even more epochs, logging as the accuracy of validation and training set is between 17%~30%;
*  `ResNeXt`, `AlexNet`, `GoogLeNet` in 30 even more epochs, logging as the accuracy of validation and training set is between 30%~50%;
*  I found that `inception-ResNet-v2`, `ResNet` performance are best, in 300 epochs val-acc can reach 60% above even 70%.

According to some papers, resolution of image is also significant for performance. Due to limited GPU RAM, three GPUs (0  GeForce GTX TIT 6082MiB, 1  Tesla K20c 4742MiB, 2  TITAN X (Pascal) 12189MiB) , I set batch size (not batch number) between 10 and 30 (10+ images per gpu) and resize original image to `224*224`.

However, the best submission is not those models, which have highest val-acc (such as 70%), but those models whose train-acc and val-acc are similar and just reach a not bad val-acc (such as 60%).

What a pity! I don't try to make augmentation based on original training and additional images. I think it must make sense.

Note: I found that the index order of GPU in `MXNet` (when declaring `mx.gpu(i)`) is opposite to `nvidia-smi` printed order( below ). In MXNet, the 0 is not GeForce GTX TITAN but TITAN X (Pascal).

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.39                 Driver Version: 375.39                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX TIT...  Off  | 0000:02:00.0     Off |                  N/A |
| 26%   36C    P8    14W / 250W |      2MiB /  6082MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla K20c          Off  | 0000:03:00.0     Off |                    0 |
| 30%   36C    P8    26W / 225W |      2MiB /  4742MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  TITAN X (Pascal)    Off  | 0000:82:00.0     Off |                  N/A |
| 44%   73C    P2   156W / 250W |   8713MiB / 12189MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    2     31661    C   python                                        8707MiB |
+-----------------------------------------------------------------------------+
```


## 2. Fine-tune from pre-trained model

Although results of training `inception-ResNet-v2` and `ResNet` from scratch are good, but I found the results from fine-tuning pre-trained models (based on ImageNet data set) are better.

### 2.1 Easily Over-fitting and Great ResNet Model

* I fine-tuned `ResNet-18`, `ResNet-34`, `ResNet-50`, `ResNet-101`, `ResNet-152`, `ResNet-200`, `ResNeXt-50`, `ResNeXt-101` models.
* It's very easily over-fitting to fine-tuning on pre-trained model. After three or four epoch, model have apparently over-fitting evidence.

### 2.2 Hyper-Parameter Optimization

Besides, I only made parameter optimization about learning rate, which I find **smaller the learning rate is, more easily over-fitting the model is.** Of course, you can make some regularization such as `early stopping` to delay this procedure.

### 2.3 Network Depth

Generally speaking, I found deeper the network is, better the result I get, but it's not always true. For instance. Below networks are great:

 |        model        |best val-acc epoch|submission score (log-loss) | val-acc   |train-acc |
 |---------------------| ---------------- | -------------------------  |---------  |--------- |
 |ResNeXt-50-lr-0.01   |         3        | 0.74195                    |  0.695312 | 0.875    |
 |ResNeXt-101-lr-0.01  |         3        | 0.77904                    |  0.630208 | 0.864583 |
 |ResNet-101-lr-0.01   |         2        | 0.77222                    |  0.605769 | 0.645833 |
 |ResNet-152-lr-0.01   |         3        | 0.74618                    |  0.673077 | 0.822917 |
 |ResNet-200-lr-0.01   |         3        | 0.76409                    |  0.666667 | 0.84375  |

so far, I make some parameter modified about learning rate and adding momentum. However, after reducing the learning rate to 0.001 and adding momentum as 0.9, the validation accuracy and submission score (log-loss) have no improvement but submission score dropped.

### 2.4 Pre-trained model

All pre-trained models're from [data.dmlc.ml/models](http://data.dmlc.ml/models/).

Different pre-trained data sets make fine-tuned model different performance.

I tried pre-trained models based on two kind images: the one is `ImageNet-11k`, the other is `ImageNet-11k-place365-ch`.

I don't know what's the `ImageNet-11k-place365-ch` image, it seems place or street-view images. The performance of this kind pre-trained model is not good, same as train from scratch.



## 3. Boosting

After fine-tuning those networks, I think I can make more progress on submission score using boosting based on fine-tuned models.

However, it seems no improvement but dropped a lot (dropped 0.4~0.6 log-loss). I think maybe I have something wrong with use of `XGBoost`. Of course, I have to admit I'm, in fact, new to use `XGBoost`.
