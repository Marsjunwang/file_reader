from utils.config_load import cfg, cfg_from_yaml_file
from utils.dataConverter import RawDataReader
from utils.calib_extractor import CalibExtractor
from tqdm import tqdm
import os
import glob
import time
import concurrent.futures as futures

DEBUG = False

def image_path(root_path,image_path):
    jpg_path = os.path.join(root_path,image_path)
    jpg_path += '/' + os.listdir(jpg_path)[0]
    return jpg_path

def reader_2_writer(index):
    label_file = os.path.basename(label_list[index])
    time_stamp = os.path.splitext(label_file)[0]
    filename = str(index).zfill(6)
            
    # 1.points reader
    pcd_path = pc_hash_table[time_stamp]
    label_path = label_list[index]
    label = json_reader(label_path)
    if not os.path.exists(pcd_path):
        raise ValueError(f"The frame {time_stamp} pointcloud is missing!")
    if DEBUG:
        start_time = time.time()
    points = DataReader.pcd_reader(pcd_path=pcd_path)
    points= RawDataReader.raw_to_kitti_calib(points)
    points.tofile(os.path.join(all_sensor_path['velodyne'],'{}.bin'.format(filename)))
    if DEBUG:
        pcd_time = time.time()
        print(f'Pcd time is {pcd_time - start_time}')

    # 2.Image reader
    jpg_path = image_path(label_path.split('/point_cloud_3d')[0],'images/front-narrow')
    image_reader(jpg_path,all_sensor_path['image_front-narrow'],filename)
    
    jpg_path = image_path(label_path.split('/point_cloud_3d')[0],'images/front-wide')
    image_reader(jpg_path,all_sensor_path['image_front-wide'],filename)
    
    jpg_path = image_path(label_path.split('/point_cloud_3d')[0],'images/left-forward')
    image_reader(jpg_path,all_sensor_path['image_left-forward'],filename)
    
    jpg_path = image_path(label_path.split('/point_cloud_3d')[0],'images/left-backward')
    image_reader(jpg_path,all_sensor_path['image_left-backward'],filename)
    
    jpg_path = image_path(label_path.split('/point_cloud_3d')[0],'images/right-forward')
    image_reader(jpg_path,all_sensor_path['image_right-forward'],filename)
    
    jpg_path = image_path(label_path.split('/point_cloud_3d')[0],'images/right-backward')
    image_reader(jpg_path,all_sensor_path['image_right-backward'],filename)
    if DEBUG:
        img_time = time.time()
        print(f'Image time is {img_time - pcd_time}')
        
    # 3.Label reader
    label_save_path = os.path.join(all_sensor_path['label_2'], '{}.txt'.format(filename))
    RawDataReader.json2txt(label,label_save_path)
    if DEBUG:
        lab_time = time.time()
        print(f'Image time is {lab_time - img_time}')
        
    # 4.Calib reader
    calib_path = calib_list[index]
    calib_targ_pth = os.path.join(all_sensor_path['calib'])
    CalibExtractor(src_pth=calib_path,targ_dir=calib_targ_pth,filename='{}.txt'.format(filename),data_format=DataReader.cvt_type)
    if DEBUG:
        cal_time = time.time()
        print(f'Image time is {cal_time - lab_time}')

if __name__ == "__main__":
    # 1.Read config
    config_path = '/media/jw11/jw13/a_project_/file_reader-main/config/xj3_to_kitti_28000.yaml'
    cfg = cfg_from_yaml_file(config_path, cfg)
    converter_type = cfg.type
    root_path=cfg.root_path
    save_path=cfg.save_path
    label_file_tree = cfg.label_file_tree,
    pcd_file_tree = cfg.pcd_file_tree,
    calib_file_tree = cfg.calib_file_tree,
    sensor_lib = cfg.sensor_lib,

    # 2.File Tree reader
    DataReader = RawDataReader(converter_type,
                               root_path,
                               save_path,
                               label_file_tree = cfg.label_file_tree,
                               pcd_file_tree = cfg.pcd_file_tree,
                               calib_file_tree = cfg.calib_file_tree,
                               sensor_lib = cfg.sensor_lib)
    all_sensor_path = DataReader.all_sensor_path
    label_list = DataReader.label_list
    calib_list = DataReader.calib_list
    pc_hash_table = DataReader.pc_hash_table
    data_len = len(label_list)
    image_ids = [id for id in range(data_len)]
    image_reader = RawDataReader.image_reader_v2
    json_reader = RawDataReader.json_reader
    
    # 3.Data Converter
    for i in tqdm(image_ids):
        if i > 30:
            break
        reader_2_writer(i)
        
    # num_workers = 4
    # def run(f, my_iter):
    #     with futures.ProcessPoolExecutor(num_workers) as executor:
    #         list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
    # run(reader_2_writer,image_ids)     

