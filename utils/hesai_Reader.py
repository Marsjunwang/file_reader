from curses.ascii import isdigit
import os 
import glob
from pathlib import Path
from tracemalloc import start

# from cv2 import split
from utils.progress_bar import progress_bar_iter as prog_bar
import numpy as np
import shutil
from PIL import Image
import json
import time

from tqdm import tqdm
import concurrent.futures as futures
from tqdm.contrib.concurrent import process_map
import multiprocessing
from functools import partial
import re
import warnings
from easydict import EasyDict

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)
hesai_name_mapping = {
                    "car":"vehicle",
                    "trucks":"big_vehicle",
                    "van":"big_vehicle",
                    "mini_truck":"big_vehicle",
                    "motorcyclist":"motorcycle",
                    "cyclist":"bicycle", 
                    "bicyclist":"bicycle",
                    "bicycle":"bicycle",
                    # "electric_bicycle":4,
                    "bus":"huge_vehicle",
                    "traffic_cone":"cone",
                    "pedestrian":"pedestrian",
                    'tricycle':"tricycle",
                    'other':'ignore',
                    'noise':'ignore',
                    'small_unmovable':'ignore',
                    "small_movable":'ignore',
                    "special_vehicle":'ignore',
                    "crash_barrel":'ignore',
                    "construction_sign":'ignore',
                    "water_horse":'ignore'
                }
class RawDataReader():
    def __init__(self,
                 raw_data_type = None,
                 root_path = '/media/jw11/jw11/data/6018_frames_data/raw_data/PCDx1_CAMx4_1',
                 save_path = '/media/jw11/jw11/data/converter_data',
                 labe_path = '/media/jw11/jw11/data/6018_frames_data/label_baidu_0610_6018/PCDx1_CAMx4_1',
                 label_file_tree = None,
                 pcd_file_tree = None,
                 sensor_lib = None):
        if not (os.path.exists(root_path) and os.path.exists(labe_path)):
            raise ValueError("Input path is Empty!!!")
        self.raw_data_type = raw_data_type
        self.root_path = root_path
        self.save_path = save_path
        self.label_path = labe_path
        self.all_sensor_path = self.create_kitti_tree(save_path,sensor_lib)
        self.label_list = self.label_reader_new(labe_path,label_file_tree)
        self.pc_hash_table = self.pc_table(root_path,pcd_file_tree)
        
    def data_reader(self):
        root_path = self.root_path
        data_list = os.listdir(root_path)
        ext = '.pcd'
        data_list = glob.glob(str(root_path / '*{}'.format(ext))) if root_path.is_dir() else [root_path]
        return data_list
    

    def parse_header(self,lines):
        """ Parse header of PCD files.
        """
        metadata = {}
        for ln in lines:
            if ln.startswith('#') or len(ln) < 2:
                continue
            match = re.match('(\w+)\s+([\w\s\.]+)', ln)
            if not match:
                warnings.warn("warning: can't understand line: %s" % ln)
                continue
            key, value = match.group(1).lower(), match.group(2)
            if key == 'version':
                metadata[key] = value
            elif key in ('fields', 'type'):
                metadata[key] = value.split()
            elif key in ('size', 'count'):
                metadata[key] = map(int, value.split())
            elif key in ('width', 'height', 'points'):
                metadata[key] = int(value)
            elif key == 'viewpoint':
                metadata[key] = map(float, value.split())
            elif key == 'data':
                metadata[key] = value.strip().lower()
            # TODO apparently count is not required?
        # add some reasonable defaults
        if 'count' not in metadata:
            metadata['count'] = [1]*len(metadata['fields'])
        if 'viewpoint' not in metadata:
            metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if 'version' not in metadata:
            metadata['version'] = '.7'
        return metadata
    
    def _build_dtype(self,metadata):
        """ Build numpy structured array dtype from pcl metadata.

        Note that fields with count > 1 are 'flattened' by creating multiple
        single-count fields.

        *TODO* allow 'proper' multi-count fields.
        """
        fieldnames = []
        typenames = []
        for f, c, t, s in zip(metadata['fields'],
                            metadata['count'],
                            metadata['type'],
                            metadata['size']):
            np_type = pcd_type_to_numpy_type[(t, s)]
            if c == 1:
                fieldnames.append(f)
                typenames.append(np_type)
            else:
                fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
                typenames.extend([np_type]*c)
        # dtype = np.dtype(zip(fieldnames, typenames))
        # dtype = tuple(typenames)
        dtype = np.dtype({'names':tuple(fieldnames),
                 'formats':tuple(typenames)})
        return dtype  
    def parse_ascii_pc_data(self,f, dtype, metadata):
        """ Use numpy to parse ascii pointcloud data.
        """
        return np.loadtxt(f, dtype=dtype, delimiter=' ')

    def parse_binary_pc_data(self,f, dtype, metadata):
        rowstep = metadata['points']*dtype.itemsize
        # for some reason pcl adds empty space at the end of files
        buf = f.read(rowstep)
        # return np.fromstring(buf, dtype=dtype)
        return np.frombuffer(buf, dtype=dtype)
    
    def pcd_reader(self,pcd_path):
        """ Parse pointcloud coming from file object f
        """
        with open(pcd_path, 'rb') as f:
            header = []
            while True:
                ln = f.readline().strip().decode('utf-8')
                header.append(ln)
                if ln.startswith('DATA'):
                    metadata = self.parse_header(header)
                    dtype = self._build_dtype(metadata)
                    break
            if metadata['data'] == 'ascii':
                pc_data = self.parse_ascii_pc_data(f, dtype, metadata)
            elif metadata['data'] == 'binary':
                pc_data = self.parse_binary_pc_data(f, dtype, metadata)
        # meta =[]
        # points = []
        # data_buffer = False
        # with open(pcd_path, 'rb') as f:
        #     for line in f:
        #         line = line.strip().decode('utf-8')
        #         meta.append(line)
        #         if data_buffer:
        #             point = list(map(float, [x for x in line.split(' ')]))
        #             points.append(point)
        #         if line.startswith('DATA'):
        #             data_buffer =True
        return pc_data
        

    def label_reader(raw_label_path):
        if raw_data_type == 'data_6018':
            return glob.glob(os.path.join(raw_label_path,'*','*','*.json'))
        elif raw_data_type == 'data_60' or raw_data_type == 'data_6187':
            return glob.glob(os.path.join(raw_label_path,'*.json'))
        
    # @staticmethod
    def label_reader_new(self,root_path, file_tree):
        return glob.glob(str(Path(root_path) / file_tree))
    
    
    # @staticmethod  
    def pc_table(self,root_path, file_tree):
        all_pc_paths = self.label_reader_new(root_path, file_tree)
        pc = EasyDict({})
        if self.raw_data_type == "data_hesai_6000":
            for pc_path in all_pc_paths:
                pc_path_list = pc_path.split('/')
                if pc_path_list[-4] not in pc.keys():
                    pc[pc_path_list[-4]] = {}
                pc[pc_path_list[-4]][pc_path_list[-2]] = pc_path

        else:  
            for pc_path in all_pc_paths:
                pc[os.path.basename(pc_path).split('.')[0]] = pc_path
            
        return pc
    
    def json_reader(label_path):
        with open (label_path, 'r') as f:
            label = json.load(f)
        return label

    def raw_to_kitti_calib(pc):
        # points[:,2] = points[:,2] - 3
        np_x = (np.array(pc['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc['z'], dtype=np.float32)).astype(np.float32) + 2
        np_i = (np.array(pc['intensity'], dtype=np.float32)).astype(np.float32) / 256

        return np.nan_to_num(np.transpose(np.vstack((np_x, np_y, np_z, np_i))))
    
    @staticmethod  
    def create_kitti_tree(save_path,sensor_lib):
        all_sensor_path = dict()
        for i in sensor_lib:
            velodyne_path = os.path.join(save_path, i)

            if not os.path.exists(velodyne_path):
                os.makedirs(velodyne_path)
            all_sensor_path[i]=velodyne_path
        return all_sensor_path

    def image_reader(info_path,save_path,filename):
        def jpg_reader(image_path):
            im1 = Image.open(r'{}'.format(image_path))
            return im1
        
        jpg_path = info_path
        if not os.path.exists(jpg_path):
                raise ValueError("The frame image1 is misssing!")
        image_2 = jpg_reader(jpg_path)
        png_path = os.path.join(save_path,'{}.png'.format(filename))
        image_2.save(r'{}'.format(png_path))
        
    def json2txt(js,save_path):
        lis_st = []
        for lb in js['labels']:
                st = lb['type']+' 0'*7+' '+\
                    str(lb['size']['x'])+' '+\
                    str(lb['size']['y'])+' '+\
                    str(lb['size']['z'])+' '+\
                    str(lb['center']['x'])+' '+\
                    str(lb['center']['y'])+' '+\
                    str(lb['center']['z'])+' '+\
                    str(lb['rotation']['roll']+ np.pi / 2)+'\n'
                lis_st.append(st)
        f_w = open(save_path, 'w')
        f_w.writelines(lis_st)
        f_w.close()
        
    def json2txt_60(js,save_path):
        lis_st = []
        for lb in js['step_1']['result']:
                st = lb['attribute']+' 0'*7+' '+\
                    str(lb['depth'])+' '+\
                    str(lb['width'])+' '+\
                    str(lb['height'])+' '+\
                    str(lb['center']['x'])+' '+\
                    str(lb['center']['y'])+' '+\
                    str(lb['center']['z'])+' '+\
                    str(lb['rotation']+ np.pi / 2)+'\n'
                lis_st.append(st)
        f_w = open(save_path, 'w')
        f_w.writelines(lis_st)
        f_w.close()
    
    def json2txt_hesai(js,save_path):
        lis_st = []
        for lb in js['instances']:
                st = hesai_name_mapping[lb['category']]+' 0'*7+' '+\
                    str(lb['geometry']['size']['complete_size']['l'])+' '+\
                    str(lb['geometry']['size']['complete_size']['w'])+' '+\
                    str(lb['geometry']['size']['complete_size']['h'])+' '+\
                    str(lb['geometry']['position']['complete_center']['x'])+' '+\
                    str(lb['geometry']['position']['complete_center']['y'])+' '+\
                    str(lb['geometry']['position']['complete_center']['z']+2)+' '+\
                    str(lb['geometry']['heading']+ np.pi / 2)+'\n'
                lis_st.append(st)
        f_w = open(save_path, 'w')
        f_w.writelines(lis_st)
        f_w.close()


def reader_2_writer(index):
    if raw_data_type == 'data_hesai_6000':
            label_path = label_list[index]
            label = json_reader(label_path)
            pcd_paths = []
            try:
                for i in label['annotations']:
                    pcd_paths.append(pc_hash_table[label_path.split('/')[-2]][i['url'].split('/')[-2]])
            except:
                l_path = label_path.split('/')[-2]
                print(f'Pcd path {l_path} is not existing')
                return
            labels = label['annotations']
            global start_idx
    else:
            label_file = os.path.basename(label_list[index])
            time_stamp = os.path.splitext(os.path.basename(label_file))[0]
            filename = str(index).zfill(6)
            
            # 1.points reader
            if raw_data_type == 'data_6018':
                pcd_path = os.path.join(cfg.root_path,label_list[index].split('/')[-3],'{}.pcd'.format(time_stamp))
            elif raw_data_type == 'data_60':
                pcd_path = os.path.join(cfg.root_path,'{}.pcd'.format(time_stamp))
            elif raw_data_type == 'data_6187' or raw_data_type == 'data_6187_all_label':
                pcd_path = pc_hash_table[time_stamp]
            pcd_paths = [pcd_path]
            
            label_path = label_list[index]
            label = json_reader(label_path)
            labels = [label]
          
    for idx,pcd_path in enumerate(pcd_paths):  
        
        #filename idex 
        filename = str(start_idx + idx).zfill(6)
            
        #1.Label reader     
        label_save_path = os.path.join(all_sensor_path['label_2'], '{}.txt'.format(filename))
        if raw_data_type == 'data_60':
            RawDataReader.json2txt_60(labels[idx],label_save_path)
        elif raw_data_type == 'data_hesai_6000':
            if len(labels[idx]['instances']) == 0:
                start_idx -= 1
                continue
            RawDataReader.json2txt_hesai(labels[idx],label_save_path)
        else:
            RawDataReader.json2txt(labels[idx],label_save_path)
        
        #2.point reader   
        pcd_path = pcd_paths[idx]
        if not os.path.exists(pcd_path):
            raise ValueError(f"The frame {time_stamp} pointcloud is missing!")
            # print(f"The frame {time_stamp} pointcloud is missing!")
            return

        points = DataReader.pcd_reader(pcd_path=pcd_path)
        points= RawDataReader.raw_to_kitti_calib(points)
        
        points.tofile(os.path.join(all_sensor_path['velodyne'],'{}.bin'.format(filename)))

        # # 2.Image reader
        # jpg_path = os.path.join(cfg.root_path,label_list[index].split('/')[-3],'{}_cam2.jpg'.format(time_stamp))
        # image_reader(jpg_path,all_sensor_path['image_2'],filename)
        
        # jpg_path = os.path.join(cfg.root_path,label_list[index].split('/')[-3],'{}_cam1.jpg'.format(time_stamp))
        # image_reader(info_path = jpg_path,save_path = all_sensor_path['image_1'],filename = filename)
        
        # jpg_path = os.path.join(cfg.root_path,label_list[index].split('/')[-3],'{}_cam3.jpg'.format(time_stamp))
        # image_reader(jpg_path,all_sensor_path['image_3'],filename)
        
        # jpg_path = os.path.join(cfg.root_path,label_list[index].split('/')[-3],'{}_cam4.jpg'.format(time_stamp))
        # image_reader(jpg_path,all_sensor_path['image_4'],filename)
        

    # pcd_idx = [i for i in range(len(pcd_paths))]
    # def run(f, my_iter):
    #     with futures.ProcessPoolExecutor(4) as executor:
    #         list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
    # run(hesai_writer,pcd_idx)   
        
    start_idx = start_idx + len(pcd_paths)
    
    

from utils.config_load import cfg, cfg_from_yaml_file
config_path = '/media/jw11/jw13/a_project_/file_reader/config/data_hesai_6000.yaml'
cfg = cfg_from_yaml_file(config_path, cfg)


raw_data_type = cfg.type

DataReader=RawDataReader(raw_data_type=raw_data_type,
                    root_path=cfg.root_path,
                    save_path=cfg.save_path,
                    labe_path=cfg.raw_label_path,
                    label_file_tree = cfg.label_file_tree,
                    pcd_file_tree = cfg.pcd_file_tree,
                    sensor_lib = cfg.sensor_lib)
all_sensor_path = DataReader.all_sensor_path
label_list = DataReader.label_list
pc_hash_table = DataReader.pc_hash_table
def timestamp(x):
        return int(''.join([st for st in os.path.basename(x) if st.isdigit()]))
if raw_data_type == 'data_hesai_6000':
    label_list.sort(key=lambda x:int(''.join([i for i in x.split('/')[-2] if i.isdigit()])))
else:
    label_list.sort(key=lambda x:timestamp(x))
image_ids = [id for id in range(len(label_list))]
image_reader = RawDataReader.image_reader
json_reader = RawDataReader.json_reader


def main():
    
    ## test
    global start_idx
    start_idx = 0
    for i in tqdm(image_ids):
        # if i > 10:
        #     break
        reader_2_writer(i)
        
    num_workers = 4
        
    # def run(f, my_iter):
    #     with futures.ProcessPoolExecutor(num_workers) as executor:
    #         list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
    # run(reader_2_writer,image_ids)   

if __name__ == "__main__":
    main()  

