#!/bin/bash

python train.py --split_name clr_1 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.001
python test.py --split_name clr_1 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.001
python train.py --split_name clr_2 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.001
python test.py --split_name clr_2 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.001
python train.py --split_name clr_3 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.001
python test.py --split_name clr_3 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.001
python train.py --split_name clr_4 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.001
python test.py --split_name clr_4 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.001
python train.py --split_name clr_1 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_1 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_2 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_2 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_3 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_3 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_4 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_4 --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_1 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.001
python test.py --split_name clr_1 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.001
python train.py --split_name clr_2 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.001
python test.py --split_name clr_2 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.001
python train.py --split_name clr_3 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.001
python test.py --split_name clr_3 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.001
python train.py --split_name clr_4 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.001
python test.py --split_name clr_4 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.001
python train.py --split_name clr_1 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.0001
python test.py --split_name clr_1 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.0001
python train.py --split_name clr_2 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.0001
python test.py --split_name clr_2 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.0001
python train.py --split_name clr_3 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.0001
python test.py --split_name clr_3 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.0001
python train.py --split_name clr_4 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.0001
python test.py --split_name clr_4 --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.0001
python train.py --split_name clr_1 --model_name vgg16 --wd 0.0001 --lr 0.001
python test.py --split_name clr_1 --model_name vgg16 --wd 0.0001 --lr 0.001
python train.py --split_name clr_2 --model_name vgg16 --wd 0.0001 --lr 0.001
python test.py --split_name clr_2 --model_name vgg16 --wd 0.0001 --lr 0.001
python train.py --split_name clr_3 --model_name vgg16 --wd 0.0001 --lr 0.001
python test.py --split_name clr_3 --model_name vgg16 --wd 0.0001 --lr 0.001
python train.py --split_name clr_4 --model_name vgg16 --wd 0.0001 --lr 0.001
python test.py --split_name clr_4 --model_name vgg16 --wd 0.0001 --lr 0.001
python train.py --split_name clr_1 --model_name vgg16 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_1 --model_name vgg16 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_2 --model_name vgg16 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_2 --model_name vgg16 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_3 --model_name vgg16 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_3 --model_name vgg16 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_4 --model_name vgg16 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_4 --model_name vgg16 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_1 --model_name vgg16 --wd 0.001 --lr 0.001
python test.py --split_name clr_1 --model_name vgg16 --wd 0.001 --lr 0.001
python train.py --split_name clr_2 --model_name vgg16 --wd 0.001 --lr 0.001
python test.py --split_name clr_2 --model_name vgg16 --wd 0.001 --lr 0.001
python train.py --split_name clr_3 --model_name vgg16 --wd 0.001 --lr 0.001
python test.py --split_name clr_3 --model_name vgg16 --wd 0.001 --lr 0.001
python train.py --split_name clr_4 --model_name vgg16 --wd 0.001 --lr 0.001
python test.py --split_name clr_4 --model_name vgg16 --wd 0.001 --lr 0.001
python train.py --split_name clr_1 --model_name vgg16 --wd 0.001 --lr 0.0001
python test.py --split_name clr_1 --model_name vgg16 --wd 0.001 --lr 0.0001
python train.py --split_name clr_2 --model_name vgg16 --wd 0.001 --lr 0.0001
python test.py --split_name clr_2 --model_name vgg16 --wd 0.001 --lr 0.0001
python train.py --split_name clr_3 --model_name vgg16 --wd 0.001 --lr 0.0001
python test.py --split_name clr_3 --model_name vgg16 --wd 0.001 --lr 0.0001
python train.py --split_name clr_4 --model_name vgg16 --wd 0.001 --lr 0.0001
python test.py --split_name clr_4 --model_name vgg16 --wd 0.001 --lr 0.0001
python train.py --split_name clr_1 --model_name resnet18 --wd 0.0001 --lr 0.001
python test.py --split_name clr_1 --model_name resnet18 --wd 0.0001 --lr 0.001
python train.py --split_name clr_2 --model_name resnet18 --wd 0.0001 --lr 0.001
python test.py --split_name clr_2 --model_name resnet18 --wd 0.0001 --lr 0.001
python train.py --split_name clr_3 --model_name resnet18 --wd 0.0001 --lr 0.001
python test.py --split_name clr_3 --model_name resnet18 --wd 0.0001 --lr 0.001
python train.py --split_name clr_4 --model_name resnet18 --wd 0.0001 --lr 0.001
python test.py --split_name clr_4 --model_name resnet18 --wd 0.0001 --lr 0.001
python train.py --split_name clr_1 --model_name resnet18 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_1 --model_name resnet18 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_2 --model_name resnet18 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_2 --model_name resnet18 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_3 --model_name resnet18 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_3 --model_name resnet18 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_4 --model_name resnet18 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_4 --model_name resnet18 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_1 --model_name resnet18 --wd 0.001 --lr 0.001
python test.py --split_name clr_1 --model_name resnet18 --wd 0.001 --lr 0.001
python train.py --split_name clr_2 --model_name resnet18 --wd 0.001 --lr 0.001
python test.py --split_name clr_2 --model_name resnet18 --wd 0.001 --lr 0.001
python train.py --split_name clr_3 --model_name resnet18 --wd 0.001 --lr 0.001
python test.py --split_name clr_3 --model_name resnet18 --wd 0.001 --lr 0.001
python train.py --split_name clr_4 --model_name resnet18 --wd 0.001 --lr 0.001
python test.py --split_name clr_4 --model_name resnet18 --wd 0.001 --lr 0.001
python train.py --split_name clr_1 --model_name resnet18 --wd 0.001 --lr 0.0001
python test.py --split_name clr_1 --model_name resnet18 --wd 0.001 --lr 0.0001
python train.py --split_name clr_2 --model_name resnet18 --wd 0.001 --lr 0.0001
python test.py --split_name clr_2 --model_name resnet18 --wd 0.001 --lr 0.0001
python train.py --split_name clr_3 --model_name resnet18 --wd 0.001 --lr 0.0001
python test.py --split_name clr_3 --model_name resnet18 --wd 0.001 --lr 0.0001
python train.py --split_name clr_4 --model_name resnet18 --wd 0.001 --lr 0.0001
python test.py --split_name clr_4 --model_name resnet18 --wd 0.001 --lr 0.0001
python train.py --split_name clr_1 --model_name resnet50 --wd 0.0001 --lr 0.001
python test.py --split_name clr_1 --model_name resnet50 --wd 0.0001 --lr 0.001
python train.py --split_name clr_2 --model_name resnet50 --wd 0.0001 --lr 0.001
python test.py --split_name clr_2 --model_name resnet50 --wd 0.0001 --lr 0.001
python train.py --split_name clr_3 --model_name resnet50 --wd 0.0001 --lr 0.001
python test.py --split_name clr_3 --model_name resnet50 --wd 0.0001 --lr 0.001
python train.py --split_name clr_4 --model_name resnet50 --wd 0.0001 --lr 0.001
python test.py --split_name clr_4 --model_name resnet50 --wd 0.0001 --lr 0.001
python train.py --split_name clr_1 --model_name resnet50 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_1 --model_name resnet50 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_2 --model_name resnet50 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_2 --model_name resnet50 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_3 --model_name resnet50 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_3 --model_name resnet50 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_4 --model_name resnet50 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_4 --model_name resnet50 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_1 --model_name resnet50 --wd 0.001 --lr 0.001
python test.py --split_name clr_1 --model_name resnet50 --wd 0.001 --lr 0.001
python train.py --split_name clr_2 --model_name resnet50 --wd 0.001 --lr 0.001
python test.py --split_name clr_2 --model_name resnet50 --wd 0.001 --lr 0.001
python train.py --split_name clr_3 --model_name resnet50 --wd 0.001 --lr 0.001
python test.py --split_name clr_3 --model_name resnet50 --wd 0.001 --lr 0.001
python train.py --split_name clr_4 --model_name resnet50 --wd 0.001 --lr 0.001
python test.py --split_name clr_4 --model_name resnet50 --wd 0.001 --lr 0.001
python train.py --split_name clr_1 --model_name resnet50 --wd 0.001 --lr 0.0001
python test.py --split_name clr_1 --model_name resnet50 --wd 0.001 --lr 0.0001
python train.py --split_name clr_2 --model_name resnet50 --wd 0.001 --lr 0.0001
python test.py --split_name clr_2 --model_name resnet50 --wd 0.001 --lr 0.0001
python train.py --split_name clr_3 --model_name resnet50 --wd 0.001 --lr 0.0001
python test.py --split_name clr_3 --model_name resnet50 --wd 0.001 --lr 0.0001
python train.py --split_name clr_4 --model_name resnet50 --wd 0.001 --lr 0.0001
python test.py --split_name clr_4 --model_name resnet50 --wd 0.001 --lr 0.0001
python train.py --split_name clr_1 --model_name mobelnetv3_small --wd 0.0001 --lr 0.001
python test.py --split_name clr_1 --model_name mobelnetv3_small --wd 0.0001 --lr 0.001
python train.py --split_name clr_2 --model_name mobelnetv3_small --wd 0.0001 --lr 0.001
python test.py --split_name clr_2 --model_name mobelnetv3_small --wd 0.0001 --lr 0.001
python train.py --split_name clr_3 --model_name mobelnetv3_small --wd 0.0001 --lr 0.001
python test.py --split_name clr_3 --model_name mobelnetv3_small --wd 0.0001 --lr 0.001
python train.py --split_name clr_4 --model_name mobelnetv3_small --wd 0.0001 --lr 0.001
python test.py --split_name clr_4 --model_name mobelnetv3_small --wd 0.0001 --lr 0.001
python train.py --split_name clr_1 --model_name mobelnetv3_small --wd 0.0001 --lr 0.0001
python test.py --split_name clr_1 --model_name mobelnetv3_small --wd 0.0001 --lr 0.0001
python train.py --split_name clr_2 --model_name mobelnetv3_small --wd 0.0001 --lr 0.0001
python test.py --split_name clr_2 --model_name mobelnetv3_small --wd 0.0001 --lr 0.0001
python train.py --split_name clr_3 --model_name mobelnetv3_small --wd 0.0001 --lr 0.0001
python test.py --split_name clr_3 --model_name mobelnetv3_small --wd 0.0001 --lr 0.0001
python train.py --split_name clr_4 --model_name mobelnetv3_small --wd 0.0001 --lr 0.0001
python test.py --split_name clr_4 --model_name mobelnetv3_small --wd 0.0001 --lr 0.0001
python train.py --split_name clr_1 --model_name mobelnetv3_small --wd 0.001 --lr 0.001
python test.py --split_name clr_1 --model_name mobelnetv3_small --wd 0.001 --lr 0.001
python train.py --split_name clr_2 --model_name mobelnetv3_small --wd 0.001 --lr 0.001
python test.py --split_name clr_2 --model_name mobelnetv3_small --wd 0.001 --lr 0.001
python train.py --split_name clr_3 --model_name mobelnetv3_small --wd 0.001 --lr 0.001
python test.py --split_name clr_3 --model_name mobelnetv3_small --wd 0.001 --lr 0.001
python train.py --split_name clr_4 --model_name mobelnetv3_small --wd 0.001 --lr 0.001
python test.py --split_name clr_4 --model_name mobelnetv3_small --wd 0.001 --lr 0.001
python train.py --split_name clr_1 --model_name mobelnetv3_small --wd 0.001 --lr 0.0001
python test.py --split_name clr_1 --model_name mobelnetv3_small --wd 0.001 --lr 0.0001
python train.py --split_name clr_2 --model_name mobelnetv3_small --wd 0.001 --lr 0.0001
python test.py --split_name clr_2 --model_name mobelnetv3_small --wd 0.001 --lr 0.0001
python train.py --split_name clr_3 --model_name mobelnetv3_small --wd 0.001 --lr 0.0001
python test.py --split_name clr_3 --model_name mobelnetv3_small --wd 0.001 --lr 0.0001
python train.py --split_name clr_4 --model_name mobelnetv3_small --wd 0.001 --lr 0.0001
python test.py --split_name clr_4 --model_name mobelnetv3_small --wd 0.001 --lr 0.0001
python train.py --split_name clr_1 --model_name efficientnet_b7 --wd 0.0001 --lr 0.001
python test.py --split_name clr_1 --model_name efficientnet_b7 --wd 0.0001 --lr 0.001
python train.py --split_name clr_2 --model_name efficientnet_b7 --wd 0.0001 --lr 0.001
python test.py --split_name clr_2 --model_name efficientnet_b7 --wd 0.0001 --lr 0.001
python train.py --split_name clr_3 --model_name efficientnet_b7 --wd 0.0001 --lr 0.001
python test.py --split_name clr_3 --model_name efficientnet_b7 --wd 0.0001 --lr 0.001
python train.py --split_name clr_4 --model_name efficientnet_b7 --wd 0.0001 --lr 0.001
python test.py --split_name clr_4 --model_name efficientnet_b7 --wd 0.0001 --lr 0.001
python train.py --split_name clr_1 --model_name efficientnet_b7 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_1 --model_name efficientnet_b7 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_2 --model_name efficientnet_b7 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_2 --model_name efficientnet_b7 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_3 --model_name efficientnet_b7 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_3 --model_name efficientnet_b7 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_4 --model_name efficientnet_b7 --wd 0.0001 --lr 0.0001
python test.py --split_name clr_4 --model_name efficientnet_b7 --wd 0.0001 --lr 0.0001
python train.py --split_name clr_1 --model_name efficientnet_b7 --wd 0.001 --lr 0.001
python test.py --split_name clr_1 --model_name efficientnet_b7 --wd 0.001 --lr 0.001
python train.py --split_name clr_2 --model_name efficientnet_b7 --wd 0.001 --lr 0.001
python test.py --split_name clr_2 --model_name efficientnet_b7 --wd 0.001 --lr 0.001
python train.py --split_name clr_3 --model_name efficientnet_b7 --wd 0.001 --lr 0.001
python test.py --split_name clr_3 --model_name efficientnet_b7 --wd 0.001 --lr 0.001
python train.py --split_name clr_4 --model_name efficientnet_b7 --wd 0.001 --lr 0.001
python test.py --split_name clr_4 --model_name efficientnet_b7 --wd 0.001 --lr 0.001
python train.py --split_name clr_1 --model_name efficientnet_b7 --wd 0.001 --lr 0.0001
python test.py --split_name clr_1 --model_name efficientnet_b7 --wd 0.001 --lr 0.0001
python train.py --split_name clr_2 --model_name efficientnet_b7 --wd 0.001 --lr 0.0001
python test.py --split_name clr_2 --model_name efficientnet_b7 --wd 0.001 --lr 0.0001
python train.py --split_name clr_3 --model_name efficientnet_b7 --wd 0.001 --lr 0.0001
python test.py --split_name clr_3 --model_name efficientnet_b7 --wd 0.001 --lr 0.0001
python train.py --split_name clr_4 --model_name efficientnet_b7 --wd 0.001 --lr 0.0001
python test.py --split_name clr_4 --model_name efficientnet_b7 --wd 0.001 --lr 0.0001
python ring.py