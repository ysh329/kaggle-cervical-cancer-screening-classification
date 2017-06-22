# cervical-cancer-screening
Solution for Intel &amp; MobileODT Cervical Cancer Screening

## Data:

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

## Train from scratch

First, I tried train `MLP`, `LeNet`, `GoogLeNet`, `AlexNet`, `ResNet-50`, `ResNet-152`, `inception-ResNet-v2`, and `ResNeXt` models from scratch based on training and additional data.
*  `MLP` and `LeNet` doesn't converge in 30 even more epochs, logging as the accuracy of validation and training set is between 17%~30%;
*  `ResNeXt`, `AlexNet`, `GoogLeNet` in 30 even more epochs, logging as the accuracy of validation and training set is between 30%~50%;
*  I found that `inception-ResNet-v2`, `ResNet` performance are best, in 300 epochs val-acc can reach 60% above even 70%.

However, the best submission is not those models, which have highest val-acc (such as 70%), but those models whose train-acc and val-acc are similar and just reach a not bad val-acc (such as 60%).

What a pity! I don't try to make augmentation based on original training and additional images. I think it must make sense.

Note: I found that the index order of GPU in `MXNet` (when declaring `mx.gpu(i)`) is opposite to `nvidia-smi` printed order.

## Fine-tune from pre-trained model

Although results of training `inception-ResNet-v2` and `ResNet` from scratch are good, but I found the results from fine-tuning pre-trained models (based on ImageNet data set) are better.

I fine-tuned `ResNet-18`, `ResNet-34`, `ResNet-50`, `ResNet-101`, `ResNet-152`, `ResNet-200`, `ResNeXt-50`, `ResNeXt-101` models.

Besides, I only made parameter optimization about learning rate, which I find **smaller the learning rate is, more easily over-fitting the model is.**

