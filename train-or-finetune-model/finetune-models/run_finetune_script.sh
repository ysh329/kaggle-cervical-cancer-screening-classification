#!/bin/bash


#printf '[resnet-50-train-add-seg-224-lr-0.01]\n'
#python run_finetune.py './resnet-50/resnet-50' '0' './resnet-50/finetune-resnet-50-train-add-seg-224' '20'

#printf '[resnet-34-train-add-seg-224-lr-0.01]\n'
#python run_finetune.py './resnet-34/resnet-34' '0' './resnet-34/finetune-resnet-34-train-add-seg-224' '30'

#printf '[resnet-18-train-add-seg-224-lr-0.01]\n'
#python run_finetune.py './resnet-18/resnet-18' '0' './resnet-18/finetune-resnet-18-train-add-seg-224' '30'

#printf '[resnet-200-train-add-seg-224-lr-0.01]\n'
#python run_finetune.py './resnet-200/resnet-200' '0' './resnet-200/finetune-resnet-200-train-add-seg-224' '30'


#python run_finetune.py './resneXt-101/resnext-101' '0' './resneXt-101/resnext-101' '100'

#python run_finetune.py './resneXt-50/resnext-50' '19' './resneXt-50/finetune-resnext-50-train-add-seg-224-lr-0.01' '100'

#python run_finetune.py './resnet-152-imagenet-11k/resnet-152' '20' './resnet-152-imagenet-11k/finetune-resnet-152-imagenet-11k-train-add-seg-224-lr-0.01' '100'

#python run_finetune.py './resnet-152-imagenet-11k-365-ch/resnet-152' '20' './resnet-152-imagenet-11k-365-ch/finetune-resnet-152-imagenet-11k-365-ch-train-add-seg-224-lr-0.01' '100'

#python run_finetune.py './resnet-152-imagenet-11k-365-ch/resnet-152' '0' './resnet-152-imagenet-11k-365-ch/finetune-resnet-152-imagenet-11k-365-ch-train-add-seg-224-lr-0.01' '150'

#python run_finetune.py './inception-v3/Inception-7' '1' './inception-v3/finetune-Inception-7-train-add-seg-224-lr-0.01' '30'

#python run_finetune.py './caffenet/caffenet' '0' './caffenet/caffenet' '30'

#python run_finetune.py './squeezenet-v1.0/squeezenet_v1.0' '0' './squeezenet-v1.0/finetune-squeezenet_v1.0-train-add-seg-224-lr-0.01' '30'

#python run_finetune.py './inception-bn/Inception-BN' '126' './inception-bn/finetune-Inception-BN-train-add-seg-224-lr-0.01' '30'

#python run_finetune.py './resneXt-101-lr-0.001/resnext-101' '0' './resneXt-101-lr-0.001/finetune-resnext-101-train-add-seg-224-lr-0.001-momentum-0.9' '200'

#python run_finetune.py './resnet-101-lr-0.001/resnet-101' '0' './resnet-101-lr-0.001/finetune-resnet-101-train-add-seg-224-lr-0.001-momentum-0.9' '200'

#python run_finetune.py './resnet-152-lr-0.001/./resnet-152' '0' './resnet-152-lr-0.001/finetune-resnet-152-train-add-seg-224-lr-0.001-momentum-0.9' '40'

#python run_finetune.py './resnet-200-lr-0.001/resnet-200' '0' './resnet-200-lr-0.001/finetune-resnet-200-train-add-seg-224-lr-0.001-momentum-0.9' '25'

python run_finetune_data-aug.py './resnet-50-aug/resnet-50' '0' './resnet-50-aug/finetune-resnet-50-train-add-aug-seg-224-lr-0.01' '1000'
