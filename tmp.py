# 定义所有的model_name
model_names = [
    "inceptionv3",
    "vgg16",
    "resnet18",
    "resnet50",
    "mobelnetv3_large",
    "mobelnetv3_small",
    "efficientnet_b7"
]

# 定义所有的split_name
split_names = ["clr_1", "clr_2", "clr_3", "clr_4", "all"]

# 打开一个新的shell脚本文件写入
with open("run_models.sh", "w") as file:
    file.write("#!/bin/bash\n\n")
    for model_name in model_names:
        for split_name in split_names:
            # 为除了"all"之外的split_name添加训练和测试命令
            if split_name != "all":
                file.write(f"python train.py --split_name {split_name} --model_name {model_name}\n")
                file.write(f"python test.py --split_name {split_name} --model_name {model_name}\n")
            else:
                # "all" split只进行训练
                file.write(f"python train.py --split_name {split_name} --model_name {model_name}\n")
        file.write("\n")

print("Shell script 'run_models.sh' has been generated.")
