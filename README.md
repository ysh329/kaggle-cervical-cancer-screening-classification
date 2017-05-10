# cervical-cancer-screening
Solution for Intel &amp; MobileODT Cervical Cancer Screening

## submit result

| submit file |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|:-----------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|a|a|a|a|a|a|a|a|

Note:

1. `5893.jpg` in Additional_Type_1 is NOT a valid image.
2. `5892.jpg` `2845.jpg` in Additional_Type_2 are NOT valid images.

首先，我用end2end的方法，测试了mlp,lenet,googlenet,alexnet,resnet-50,resnet-152,inception-resnet-v2,resnext。  
*  mlp还有lenet基本不收敛，30个epoch内val-acc一直在17%~30%之间，或许训练再长时间也和随机猜差不多；
*  resnext,alexnet,googlenet在30个epoch内val-acc在30%~50%之间；
*  inception-resnet-v2,resnet效果不错，300个epoch内val-accc可以达到60%~70%之间，但是提交的结果最好的rank不是val-acc最高的模型，而是val-acc刚上60%的；
