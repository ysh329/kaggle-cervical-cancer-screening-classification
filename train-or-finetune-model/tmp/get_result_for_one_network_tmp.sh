#!/bin/bash

# initialization
#model_prefix_arr=("./finetune-models/resnet-101/finetune-resnet-101-train-add-seg-224")
#model_prefix_arr=("./models/inception-resnet-v2-152-train-add-seg-224-lr-0.01/inception-resnet-v2-152-train-add-seg-224-lr-0.01")
#model_prefix_arr=("./finetune-models/resnet-200/finetune-resnet-200-train-add-seg-224")
#model_prefix_arr=("./finetune-models/resneXt-50/finetune-resnext-50-train-add-seg-224-lr-0.01")
model_prefix_arr=("./finetune-models/resnet-152-imagenet-11k/finetune-resnet-152-imagenet-11k-train-add-seg-224-lr-0.01")
#model_prefix_arr=("./finetune-models/resnet-152-imagenet-11k-365-ch/finetune-resnet-152-imagenet-11k-365-ch-train-add-seg-224-lr-0.01")
model_max_epoch_num_arr=(10)
#model_num=${#model_prefix_arr[@]}-1
#printf "${model_num}\n"


# start loop
for idx in 0 1
do
    model_prefix=${model_prefix_arr[idx]}
    max_epoch_num=${model_max_epoch_num_arr[idx]}
    #printf "model prefix:${model_prefix}\n"
    #printf "model epoch:${max_epoch_num}\n"
    for ((epoch_num = 1; epoch_num <= $max_epoch_num ; epoch_num++))
    do
        printf "model prefix:${model_prefix} model epoch:${epoch_num}\n"
        python ./run_inference_tmp.py $model_prefix $epoch_num
        #printf "model prefix:${model_prefix}\n"
        #printf "model epoch:${epoch_num}\n"
    done
done
