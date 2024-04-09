#!/bin/bash

python train.py --split_name clr_1 --model_name inceptionv3
python test.py --split_name clr_1 --model_name inceptionv3
python train.py --split_name clr_2 --model_name inceptionv3
python test.py --split_name clr_2 --model_name inceptionv3
python train.py --split_name clr_3 --model_name inceptionv3
python test.py --split_name clr_3 --model_name inceptionv3
python train.py --split_name clr_4 --model_name inceptionv3
python test.py --split_name clr_4 --model_name inceptionv3
python train.py --split_name all --model_name inceptionv3

python train.py --split_name clr_1 --model_name vgg16
python test.py --split_name clr_1 --model_name vgg16
python train.py --split_name clr_2 --model_name vgg16
python test.py --split_name clr_2 --model_name vgg16
python train.py --split_name clr_3 --model_name vgg16
python test.py --split_name clr_3 --model_name vgg16
python train.py --split_name clr_4 --model_name vgg16
python test.py --split_name clr_4 --model_name vgg16
python train.py --split_name all --model_name vgg16

python train.py --split_name clr_1 --model_name resnet18
python test.py --split_name clr_1 --model_name resnet18
python train.py --split_name clr_2 --model_name resnet18
python test.py --split_name clr_2 --model_name resnet18
python train.py --split_name clr_3 --model_name resnet18
python test.py --split_name clr_3 --model_name resnet18
python train.py --split_name clr_4 --model_name resnet18
python test.py --split_name clr_4 --model_name resnet18
python train.py --split_name all --model_name resnet18

python train.py --split_name clr_1 --model_name resnet50
python test.py --split_name clr_1 --model_name resnet50
python train.py --split_name clr_2 --model_name resnet50
python test.py --split_name clr_2 --model_name resnet50
python train.py --split_name clr_3 --model_name resnet50
python test.py --split_name clr_3 --model_name resnet50
python train.py --split_name clr_4 --model_name resnet50
python test.py --split_name clr_4 --model_name resnet50
python train.py --split_name all --model_name resnet50

# python train.py --split_name clr_1 --model_name mobelnetv3_large
# python test.py --split_name clr_1 --model_name mobelnetv3_large
# python train.py --split_name clr_2 --model_name mobelnetv3_large
# python test.py --split_name clr_2 --model_name mobelnetv3_large
# python train.py --split_name clr_3 --model_name mobelnetv3_large
# python test.py --split_name clr_3 --model_name mobelnetv3_large
# python train.py --split_name clr_4 --model_name mobelnetv3_large
# python test.py --split_name clr_4 --model_name mobelnetv3_large
# python train.py --split_name all --model_name mobelnetv3_large

python train.py --split_name clr_1 --model_name mobelnetv3_small
python test.py --split_name clr_1 --model_name mobelnetv3_small
python train.py --split_name clr_2 --model_name mobelnetv3_small
python test.py --split_name clr_2 --model_name mobelnetv3_small
python train.py --split_name clr_3 --model_name mobelnetv3_small
python test.py --split_name clr_3 --model_name mobelnetv3_small
python train.py --split_name clr_4 --model_name mobelnetv3_small
python test.py --split_name clr_4 --model_name mobelnetv3_small
python train.py --split_name all --model_name mobelnetv3_small

python train.py --split_name clr_1 --model_name efficientnet_b7
python test.py --split_name clr_1 --model_name efficientnet_b7
python train.py --split_name clr_2 --model_name efficientnet_b7
python test.py --split_name clr_2 --model_name efficientnet_b7
python train.py --split_name clr_3 --model_name efficientnet_b7
python test.py --split_name clr_3 --model_name efficientnet_b7
python train.py --split_name clr_4 --model_name efficientnet_b7
python test.py --split_name clr_4 --model_name efficientnet_b7
python train.py --split_name all --model_name efficientnet_b7

