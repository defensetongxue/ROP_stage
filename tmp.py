def generate_shell_script():
    # Define the ranges for each parameter
    ridge_seg_numbers = range(1, 9)  # 1 to 8
    sample_distances = range(60, 141, 20)  # 60 to 140, step 20
    sample_low_thresholds = [i / 100.0 for i in range(38, 49)]  # 0.38 to 0.48, step 0.01
    patch_size_range=[380,420]
    with open('./todo.sh', 'w') as file:
        for param_name, values in [
            # ('ridge_seg_number', ridge_seg_numbers),
            # ('sample_distance', sample_distances),
            # ('sample_low_threshold', sample_low_thresholds)
            ('patch_size',patch_size_range)
        ]:
            for val in values:
                # Write the resample.py command
                file.write(f"python cleansing.py --{param_name} {val}\n")

                # Write the test_stage.py commands for each split
                for split in range(1, 5):  # Split 1 to 4
                    file.write(f"python train.py --split_name {split} --{param_name} {val}\n")
                    
                    file.write(f"python test_stage.py --split_name {split} --{param_name} {val}\n")

                # Add an empty line for readability between parameter sets
                file.write("\n")

if __name__ == '__main__':
    generate_shell_script()
    print("Shell script todo.sh generated successfully.")
