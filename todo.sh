#!/bin/bash

python train_sz.py --split_name all --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.001
python train_sz.py --split_name all --model_name inceptionv3 --resize 299 --wd 0.0001 --lr 0.0001
python train_sz.py --split_name all --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.001
python train_sz.py --split_name all --model_name inceptionv3 --resize 299 --wd 0.001 --lr 0.0001
python train_sz.py --split_name all --model_name vgg16 --wd 0.0001 --lr 0.001
python train_sz.py --split_name all --model_name vgg16 --wd 0.0001 --lr 0.0001
python train_sz.py --split_name all --model_name vgg16 --wd 0.001 --lr 0.001
python train_sz.py --split_name all --model_name vgg16 --wd 0.001 --lr 0.0001
python train_sz.py --split_name all --model_name resnet18 --wd 0.0001 --lr 0.001
python train_sz.py --split_name all --model_name resnet18 --wd 0.0001 --lr 0.0001
python train_sz.py --split_name all --model_name resnet18 --wd 0.001 --lr 0.001
python train_sz.py --split_name all --model_name resnet18 --wd 0.001 --lr 0.0001
python train_sz.py --split_name all --model_name resnet50 --wd 0.0001 --lr 0.001
python train_sz.py --split_name all --model_name resnet50 --wd 0.0001 --lr 0.0001
python train_sz.py --split_name all --model_name resnet50 --wd 0.001 --lr 0.001
python train_sz.py --split_name all --model_name resnet50 --wd 0.001 --lr 0.0001
python train_sz.py --split_name all --model_name mobelnetv3_small --wd 0.0001 --lr 0.001
python train_sz.py --split_name all --model_name mobelnetv3_small --wd 0.0001 --lr 0.0001
python train_sz.py --split_name all --model_name mobelnetv3_small --wd 0.001 --lr 0.001
python train_sz.py --split_name all --model_name mobelnetv3_small --wd 0.001 --lr 0.0001
python train_sz.py --split_name all --model_name efficientnet_b7 --wd 0.0001 --lr 0.001
python train_sz.py --split_name all --model_name efficientnet_b7 --wd 0.0001 --lr 0.0001
python train_sz.py --split_name all --model_name efficientnet_b7 --wd 0.001 --lr 0.001
python train_sz.py --split_name all --model_name efficientnet_b7 --wd 0.001 --lr 0.0001
python ring.py