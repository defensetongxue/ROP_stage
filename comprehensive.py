from util.tools import visual_sentences, visual_error
import numpy as np
from util.stager import Stager
from models import build_model
import sys
import os
import json
import torch
from util.zoner import ZoneProcesser
from shutil import copy


class comprehensive_handler:
    def __init__(self, model_config, data_path='../autodl-tmp/dataset_ROP', model_save_path='./checkpoints/clr_1nlp.pth', visual_patch_size=200, visual_dir='./experiments/comprehensive/'):
        self.data_path = data_path
        self.angle_caler = ZoneProcesser(threshold=0.42)
        with open(os.path.join(data_path, 'annotations.json'), 'r') as f:
            self.data_dict = json.load(f)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model(model_config)
        model = model.to(device)
        model.load_state_dict(torch.load(model_save_path))
        self.stager = Stager(model, self.data_dict)
        self.visual_patch_size = visual_patch_size
        os.makedirs(visual_dir, exist_ok=True)
        self.visual_dir = visual_dir

    def bbox_info(self, sample_list, x_min, x_max, y_min, y_max, stage, image_name=''):
        if len(sample_list) == 0:
            raise ValueError(f"No samples found in the image {image_name}")
        angle_list = []
        # print(sample_list, x_min, x_max, y_min, y_max)
        for x, y, angle in sample_list:
            if x > x_min and x < x_max and y > y_min and y < y_max:
                angle_list.append(angle)
        if len(angle_list) == 0:
            visual_error(self.data_dict[image_name]['image_path'],
                         sample_list=sample_list,
                         x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, save_path='experiments/comprehensive/error/'+image_name)
            copy(self.data_dict[image_name]["ridge_seg"]["ridge_seg_path"],
                 'experiments/comprehensive/error/'+image_name[:-4]+'_ridge_seg.png')
            raise ValueError(
                f"No samples found in the box {x_min}-{x_max}-{y_min}-{y_max} in the image {image_name} and visualized in experiments/comprehensive/error/{image_name}")

        # 计算统计量
        angle_array = np.array(angle_list)
        return {
            'min': int(np.min(angle_array)),
            'max':  int(np.max(angle_array)),
            'mean':  int(np.mean(angle_array)),
            'std': int(np.std(angle_array)),
            'label': stage
        }

    def _visual_selected(self, box_info):
        min_angle = 200
        selected_i = -1
        for i, info in enumerate(box_info):
            if info['mean'] < min_angle:
                selected_i = i
                min_angle = info['mean']
        return selected_i

    def _fund_minist_box(self, image_name):
        stage_record = self.stager._stage(image_name)
        if not stage_record:  # ridge not detected
            return False
        data = self.data_dict[image_name]
        optic_disc = data['optic_disc_pred']
        sample_list = self.angle_caler._get_sample_list(
            data["ridge_seg"]["ridge_seg_path"], optic_disc["position"])

        if len(sample_list) == 0:
            raise ValueError("No samples found in {}".format(image_name))

        pred_stage = set()
        for record in stage_record:
            pred_stage.add(record['label'])
        pred_stage = list(pred_stage)

        box_angle_info = []
        for record in stage_record:
            try:
                box_angle_info.append(
                    self.bbox_info(
                        sample_list,
                        x_min=record['x_min'],
                        x_max=record['x_max'],
                        y_min=record['y_min'],
                        y_max=record['y_max'], stage=record['label'], image_name=image_name))
            except (ValueError):
                raise ValueError(f"Error in {image_name}")
        visual_points = []
        text_list = []
        label_list = []
        if len(pred_stage) == 1:
            if pred_stage[0] == 0:
                return False
            seleted_box = self._visual_selected(box_angle_info)
            x = (stage_record[seleted_box]['x_min'] +
                 stage_record[seleted_box]['x_max'])/2
            y = (stage_record[seleted_box]['y_min'] +
                 stage_record[seleted_box]['y_max'])/2
            visual_points.append((int(x), int(y)))
            text_list.append(
                f"[{box_angle_info[seleted_box]['min']}-{box_angle_info[seleted_box]['max']}]({box_angle_info[seleted_box]['mean']})")
            label_list.append(pred_stage[0]+1)
        elif len(pred_stage) > 1:
            # 按照label，每个不同的label有一个selected box
            for label in pred_stage:
                selected_box = self._visual_selected(
                    [box_info for box_info in box_angle_info if box_info['label'] == label])
                x = (stage_record[selected_box]['x_min'] +
                     stage_record[selected_box]['x_max'])/2
                y = (stage_record[selected_box]['y_min'] +
                     stage_record[selected_box]['y_max'])/2
                visual_points.append((int(x), int(y)))
                text_list.append(
                    f"[{box_angle_info[selected_box]['min']}-{box_angle_info[selected_box]['max']}]({box_angle_info[selected_box]['mean']})")
                label_list.append(label+1)
        else:
            raise
        visual_sentences(
            self.data_dict[image_name]['image_path'],
            points=visual_points,
            patch_size=self.visual_patch_size,
            box_text=text_list,
            labels=label_list,
            save_path=os.path.join(self.visual_dir, str(
                self.data_dict[image_name]['stage'])+'_'+image_name)
        )
        return True


if __name__ == '__main__':
    from configs import get_config
    args = get_config()
    save_model_name = args.split_name+args.configs['save_name']
    handler = comprehensive_handler(model_save_path=os.path.join(
        args.save_dir, save_model_name), model_config=args.configs['model'])
    with open(os.path.join(args.data_path, 'split', f"{str(args.split_name)}.json")) as f:
        image_list = json.load(f)['test']
    cnt = 0
    for image_name in image_list:
        if handler._fund_minist_box(image_name):
            cnt += 1
            # if cnt > 30:
            #     break
