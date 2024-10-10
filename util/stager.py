import torch
from PIL import Image
from .tools import  crop_patches
from torchvision import transforms
from .dataset import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Parse arguments


class Stager:
    def __init__(self, model, data_dict,resize=224,patch_size=400,sample_low_threshold=0.42):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.eval()
        self.img_norm = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
        self.visual_patch_size = 200
        self.patch_size= patch_size
        self.sample_low_threshold=sample_low_threshold
        self.data_dict=data_dict
    def _stage(self,image_name):
        result=[]
        with torch.no_grad():
           
            data = self.data_dict[image_name]
            label = int(data['stage'])
            img = Image.open(data["image_path"]).convert("RGB")
            inputs = []
            if data['ridge_seg']["max_val"] < 0.5:
                return False
            else:
                sample_visual = []
                for (x, y), val in zip(data['ridge_seg']['point_list'], data['ridge_seg']["value_list"]):
                
                    if val < self.sample_low_threshold:
                        break
                    sample_visual.append([x, y])
                    _, patch = crop_patches(img, self.patch_size, x, y,
                                            abnormal_mask=None, stage=0, save_dir=None)
                    patch = self.img_norm(patch)
                    inputs.append(patch.unsqueeze(0))
                if len(inputs) <= 0:
                    print(f"{image_name} do not have enougph data, value list is ",
                          data['ridge_seg']["value_list"])
                    raise ValueError("Not enough data")
                inputs = torch.cat(inputs, dim=0)

                outputs = self.model(inputs.to(self.device))
                probs = torch.softmax(outputs.cpu(), axis=1)
                # output shape is bc,num_class
                # get pred for each patch
                pred_labels = torch.argmax(probs, dim=1)
                for center_coor, prob,label_pred in zip(sample_visual ,probs,pred_labels):
                    x, y = center_coor
                    x_min = max(0, x - self.patch_size // 2)
                    x_max = min(img.width, x + self.patch_size // 2)
                    y_min = max(0, y - self.patch_size // 2)
                    y_max = min(img.height, y + self.patch_size // 2)

                    # Append result for each patch
                    result.append({
                        'x_min': x_min,
                        'x_max': x_max,
                        'y_min': y_min,
                        'y_max': y_max,
                        'prob': prob.tolist(),
                        'label': label_pred.item()
                    })
        return result
                    


# Set up the device
if __name__ == '__main__':
    pass
