# cervical-cancer-screening
Top 23% (191st of 848) solution for Intel &amp; MobileODT Cervical Cancer Screening. Below is repo. structure:
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

* Training set has 1700+ images.
* Training + Additional set have 8000+ images.

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

However, the best submission is not those models, which have highest val-acc (such as 70%), but those models whose train-acc and val-acc are similar and just reach a not bad val-acc (such as 60%).

What a pity! I don't try to make augmentation based on original training and additional images. I think it must make sense.

Note: I found that the index order of GPU in `MXNet` (when declaring `mx.gpu(i)`) is opposite to `nvidia-smi` printed order.

## 2. Fine-tune from pre-trained model

Although results of training `inception-ResNet-v2` and `ResNet` from scratch are good, but I found the results from fine-tuning pre-trained models (based on ImageNet data set) are better.

### 2.1 Easily Over-fitting and Great ResNet Model

* I fine-tuned `ResNet-18`, `ResNet-34`, `ResNet-50`, `ResNet-101`, `ResNet-152`, `ResNet-200`, `ResNeXt-50`, `ResNeXt-101` models.
* It's very easily over-fitting to fine-tuning on pre-trained model. After three or four epoch, model have apparently over-fitting evidence.

### 2.2 Parameter Optimization

Besides, I only made parameter optimization about learning rate, which I find **smaller the learning rate is, more easily over-fitting the model is.** Of course, you can make some regularization such as `early stopping` to delay this procedure.

### 2.3 Network Depth

Generally speaking, I found deeper the network is, better the result I get, but it's not always true. For instance. Below networks are great:

 |        model        |best val-acc epoch|submission score (log-loss) | val-acc   |train-acc |
 |---------------------|------------------|--------------------------  |---------  |--------- |
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