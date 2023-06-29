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
import concurrent.futures as futures
from tqdm.contrib.concurrent import process_map
import multiprocessing
from functools import partial
import re
import warnings
from easydict import EasyDict
import struct
import lzf


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

name_mapping_xj3_6M1 = {
    'Car':'vehicle',
    'Cyclist':'bicycle',
    'Truck':'big_vehicle',
    'Pedestrian':'pedestrian',
    'Other':'ignore'
}

class RawDataReader():
    def __init__(self,
                 raw_data_type = None,
                 root_path = '/media/jw11/jw11/data/6018_frames_data/raw_data/PCDx1_CAMx4_1',
                 save_path = '/media/jw11/jw11/data/converter_data',
                 labe_path = '/media/jw11/jw11/data/6018_frames_data/label_baidu_0610_6018/PCDx1_CAMx4_1',
                 label_file_tree = None,
                 pcd_file_tree = None,
                 calib_file_tree = None,
                 sensor_lib = None,
                 data_format = 'KITTI'):
        if not (os.path.exists(root_path) and os.path.exists(labe_path)):
            raise ValueError("Input path is Empty!!!")
        self.raw_data_type = raw_data_type
        self.root_path = root_path
        self.save_path = save_path
        self.label_path = labe_path
        self.all_sensor_path = self.create_kitti_tree(save_path,sensor_lib)
        self.label_list = self.label_reader_new(labe_path,label_file_tree)
        
        self.calib_list = [(Path(i).parent.parent / calib_file_tree).resolve() for i in self.label_list] if calib_file_tree \
            else None
        self.pc_hash_table = self.pc_table(root_path,pcd_file_tree)
        self.data_format = data_format
        
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
        return np.fromstring(buf, dtype=dtype)
    
    def parse_binary_compressed_pc_data(self,f, dtype, metadata):
        """ Parse lzf-compressed data.
        Format is undocumented but seems to be:
        - compressed size of data (uint32)
        - uncompressed size of data (uint32)
        - compressed data
        - junk
        """
        fmt = 'II'
        compressed_size, uncompressed_size =\
            struct.unpack(fmt, f.read(struct.calcsize(fmt)))
        compressed_data = f.read(compressed_size)
        # TODO what to use as second argument? if buf is None
        # (compressed > uncompressed)
        # should we read buf as raw binary?
        buf = lzf.decompress(compressed_data, uncompressed_size)
        if len(buf) != uncompressed_size:
            raise IOError('Error decompressing data')
        # the data is stored field-by-field
        pc_data = np.zeros(metadata['width'], dtype=dtype)
        ix = 0
        for dti in range(len(dtype)):
            dt = dtype[dti]
            bytes = dt.itemsize * metadata['width']
            column = np.frombuffer(buf[ix:(ix+bytes)], dt)
            pc_data[dtype.names[dti]] = column
            ix += bytes
        return pc_data
    
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
            elif metadata['data'] == 'binary_compressed':
                pc_data = self.parse_binary_compressed_pc_data(f, dtype, metadata)
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
    def label_reader_new(self,root_path, label_file_tree=None):

        return glob.glob(str(Path(root_path) / label_file_tree))
        
    
    
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
        np_z = (np.array(pc['z'], dtype=np.float32)).astype(np.float32)
        np_i = (np.array(pc['intensity'], dtype=np.float32)).astype(np.float32) / 256

        return np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
    
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
                raise ValueError("The frame image is misssing!")
        image_2 = jpg_reader(jpg_path)
        png_path = os.path.join(save_path,'{}.png'.format(filename))
        image_2.save(r'{}'.format(png_path))
        
    def image_reader_v2(info_path,save_path,filename):
        jpg_path = info_path
        png_path = os.path.join(save_path,'{}.jpg'.format(filename))
        if not os.path.exists(jpg_path):
                raise ValueError("The frame image is misssing!")

        with open(jpg_path,'rb') as fp1, open(png_path,'wb') as fp2:
            b1 = fp1.read()
            fp2.write(b1)
            

        
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
                st = name_mapping_xj3_6M1[lb['attribute']]+' 0'*7+' '+\
                    str(lb['width'])+' '+\
                    str(lb['height'])+' '+\
                    str(lb['depth'])+' '+\
                    str(lb['center']['x'])+' '+\
                    str(lb['center']['y'])+' '+\
                    str(lb['center']['z'])+' '+\
                    str(lb['rotation']+ np.pi / 2)+'\n'
                lis_st.append(st)
        f_w = open(save_path, 'w')
        f_w.writelines(lis_st)
        f_w.close()
        
    def json2txt_fusion(js,save_path):
        lis_st = []
        for lb in js['step_1']['result']:
            try:
                st = name_mapping_xj3_6M1[lb['attribute']]+' 0'*7+' '+\
                    str(lb['width'])+' '+\
                    str(lb['height'])+' '+\
                    str(lb['depth'])+' '+\
                    str(lb['center']['x'])+' '+\
                    str(lb['center']['y'])+' '+\
                    str(lb['center']['z'])+' '+\
                    str(lb['rotation']+ np.pi / 2)+'\n'
                lis_st.append(st)
            except:
                print(f'The wrong object is {lb}')
                continue
        f_w = open(save_path, 'w')
        f_w.writelines(lis_st)
        f_w.close()