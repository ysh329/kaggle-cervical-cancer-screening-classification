#!/bin/bash


printf "[inception-resnet-v2-101]\n"
python train_ccs-train.py --network inception-resnet-v2 --num-layers 101 --model-prefix ./models/inception-resnet-v2-101-train-add-seg-224-lr-0.01

#printf "[resnet]\n"
#python ./train_ccs-train.py --network resnet --model-prefix ./models/resnet-50-train-add-seg-224-lr-0.01

#printf "[resnext]\n"
#python train_ccs-train.py --network resnext --model-prefix ./models/resnext-train-seg-224

#printf "[alexnet]\n"
#python train_ccs-train.py --network alexnet | grep *accuracy*

#printf "[inception-bn]\n"
#python train_ccs-train.py --network inception-bn | grep *accuracy*

#printf "[googlenet]\n"
#python train_ccs-train.py --network googlenet | grep *accuracy*
