#!/bin/bash

python train_sz.py --split_name all --model_name inceptionv3 --resize 299
python train_sz.py --split_name all --model_name vgg16
python train_sz.py --split_name all --model_name vgg16
python train_sz.py --split_name all --model_name resnet18
python train_sz.py --split_name all --model_name resnet50
python train_sz.py --split_name all --model_name mobelnetv3_small
python train_sz.py --split_name all --model_name efficientnet_b7
