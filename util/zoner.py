import os,json
import numpy as np
import math
from PIL import Image
class ZoneProcesser():
    def __init__(self,image_size=(1600,1200),
                 camrea_weight=1570,
                 camera_angle=math.pi/2,
                 sample_dense=5,
                 threshold=0.5):
        self.weight,self.height=image_size
        self.camera_angle=camera_angle
        self.radius=camrea_weight/(2*math.sin(camera_angle/2))
        self.sample_dense=sample_dense
        self.threshold=threshold

        self.crop_padded=CropPadding()
    def _get_z(self,x,y):
        tmp=self.radius**2-x**2-y**2
        if tmp>=0:
            return round(math.sqrt(tmp))
        return 0
    def _get_xy(self,coordinate_x,coordinate_y):
        return coordinate_x-(self.weight/2),coordinate_y-(self.height/2)
    
    def _xy2coor(self,x,y):
        return x+(self.weight/2),y+(self.height/2)
    
    def calculate_angle(self,ridge_x,ridge_y,optic_x,optic_y):
        z_ridge = self._get_z(ridge_x, ridge_y)
        z_optic = self._get_z(optic_x, optic_y)
        l = math.sqrt((ridge_x - optic_x)**2 + (ridge_y - optic_y)**2 + (z_ridge - z_optic)**2)
        # in circle l ischord length, cal the angle with l and radius
        try:
            angle=2*math.asin(l/(2*self.radius))
        except:
            print(l)
            raise
        return math.degrees(angle)
    
    def ridge_sample(self,mask):
        """
        For each point in mask 2D numpy array, if coordinates x, y satisfy 
        x % sample_dense == 0 and y % sample_dense == 0 and mask[x][y] > threshold, 
        sample it. Return a list of coordinates (x, y).
        """
        rows, cols = np.where((mask > self.threshold) & 
                              (np.arange(mask.shape[0])[:, None] % self.sample_dense == 0) & 
                              (np.arange(mask.shape[1])[None, :] % self.sample_dense == 0))
        samples_coordinate = list(zip(rows, cols))
        samples=[]
        for i,j in samples_coordinate:
            x,y =self._get_xy(j,i)
            samples.append([x,y])
        return samples
    
    def _get_angle(self, ridge_path, optic_disc_coordinate, ridge_threshold=0.5):
        optic_x, optic_y = self._get_xy(optic_disc_coordinate[0], optic_disc_coordinate[1])
        ridge = Image.open(ridge_path).convert('L')
        ridge = np.array(ridge)
        ridge=np.where(ridge>int(255*ridge_threshold),1,0)
        samples = self.ridge_sample(ridge)

        angles = []
        for x, y in samples:
            angle = self.calculate_angle(x, y, optic_x, optic_y)
            angles.append((angle, self._xy2coor(x,y)))

        if not angles:
            return json.dumps({"error": "No samples found."})

        # Calculate min and avg angles
        min_angle, min_coor = min(angles, key=lambda x: x[0])
        avg_angle = sum(angle for angle, _ in angles) / len(angles)
        
        # Find the coordinate closest to the average angle
        avg_coor = min(angles, key=lambda x: abs(x[0] - avg_angle))[1]

        result = {
            "min_angle": int(min_angle),
            "min_coor": min_coor,
            "avg_angle": int(avg_angle),
            "avg_coor": avg_coor
        }
        
        return result
    def _get_sample_list(self,ridge_path, optic_disc_coordinate, ridge_threshold=0.42):
        optic_x, optic_y = self._get_xy(optic_disc_coordinate[0], optic_disc_coordinate[1])
        ridge = Image.open(ridge_path).resize((1600,1200)).convert('L')
        ridge = np.array(ridge)
        ridge=np.where(ridge>int(255*ridge_threshold),1,0)
        # print(ridge.shape)
        samples = self.ridge_sample(ridge)

        angle_list=[]
        for x, y in samples:
            angle = self.calculate_angle(x, y, optic_x, optic_y)
            coor_x,coor_y=self._xy2coor(x,y)
            angle_list.append([coor_x,coor_y,int(angle)])
        return angle_list
class CropPadding:
    def __init__(self,box=(80, 0, 1570, 1200)):
        self.box=box
    def __call__(self,img) :
        return img.crop(self.box)
if __name__=='__main__':
    data_path='../autodl-tmp/dataset_ROP'
    with open(os.path.join(data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    
    # simplify the data 
    processer=ZoneProcesser()
    for image_name in data_dict:
    # for image_name in test_list:
        data=data_dict[image_name]
        optic_disc=data['optic_disc_pred']
        # if optic_disc['distance']!='visible':
        #     continue
        if "ridge_seg" in data and "ridge_seg_path" in data["ridge_seg"]:
            angle=processer._get_angle(data["ridge_seg"]["ridge_seg_path"],optic_disc["position"])
            # zone[data['zone']-1].append((image_name,angle))
            data_dict[image_name]['zone_pred']=angle
    with open(os.path.join(data_path,'annotations.json'),'w') as f:
        json.dump(data_dict,f)
    