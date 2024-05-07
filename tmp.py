# Define the parameters for the training script
models = [
    "inceptionv3",
    "vgg16",
    "resnet18",
    "resnet50",
    "mobelnetv3_small",
    "efficientnet_b7"
]
weight_decays = [1e-4, 1e-3]
learning_rates = [1e-3, 1e-4]

# Open a file to save the commands
with open("todo.sh", "w") as f:
    f.write("#!/bin/bash\n\n")

    # Iterate over each model
    for model in models:
        # Iterate over each combination of weight decay and learning rate
        for wd in weight_decays:
            for lr in learning_rates:
                # Check if the model is inceptionv3 to add the resize parameter
                if model == "inceptionv3":
                    command = f"python train_sz.py --split_name all --model_name {model} --resize 299 --wd {wd} --lr {lr}\n"
                else:
                    command = f"python train_sz.py --split_name all --model_name {model} --wd {wd} --lr {lr}\n"
                # Write each command to the file
                f.write(command)

print("Shell script generated successfully.")
