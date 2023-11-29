import argparse,json

def get_config():
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--data_path', type=str, default='../autodl-tmp/dataset_ROP',
                        help='Path to the target folder to store the processed datasets.')
    # split
    parser.add_argument('--split_name', type=str, default='1',
                        help='which split to use.')
    parser.add_argument('--word_size', type=int, default=4,
                        help='which split to use.')
    parser.add_argument('--aux_r', type=float, default=1.,
                        help='which split to use.')
    parser.add_argument('--patch_size', type=int, default=160,
                        help='which split to use.')
    parser.add_argument('--hybird', type=str, default='resnet50',
                        help='which split to use.')
    
    # train and test
    parser.add_argument('--save_dir', type=str, default="./checkpoints",
                        help='Name of the file to save the best model during training.')
    parser.add_argument('--result_path', type=str, default="experiments",
                        help='Path to the visualize result or the pytorch model will be saved.')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='which split to use.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='which split to use.')
    # config file 
    parser.add_argument('--cfg', help='experiment configuration filename',
                        default="./configs/defalut.json", type=str)
    
    args = parser.parse_args()
    # Merge args and config file 
    with open(args.cfg,'r') as f:
        args.configs=json.load(f)
    return args