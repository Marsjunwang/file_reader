from utils.config_load import cfg, cfg_from_yaml_file
from utils.dataConverter import RawDataReader
from utils.calib_extractor import CalibExtractor
from tqdm import tqdm
import os

DEBUG = False

config_path = '/media/jw11/jw13/a_project_/file_reader-main/config/baidu_40000.yaml'

#1.Read config
cfg = cfg_from_yaml_file(config_path, cfg)
raw_data_type = cfg.type

DataReader=RawDataReader(raw_data_type=raw_data_type,
                    root_path=cfg.root_path,
                    save_path=cfg.save_path,
                    labe_path=cfg.raw_label_path,
                    label_file_tree = cfg.label_file_tree,
                    pcd_file_tree = cfg.pcd_file_tree,
                    calib_file_tree = cfg.calib_file_tree,
                    sensor_lib = cfg.sensor_lib,
                    data_format=cfg.data_format)
all_sensor_path = DataReader.all_sensor_path
label_list = DataReader.label_list
calib_list = DataReader.calib_list
data_format = cfg.data_format
    
pc_hash_table = DataReader.pc_hash_table
image_ids = [id for id in range(len(label_list))]
image_reader = RawDataReader.image_reader_v2
json_reader = RawDataReader.json_reader


def reader_2_writer(index):
            label_file = os.path.basename(label_list[index])
            time_stamp = os.path.splitext(os.path.basename(label_file))[0]
            filename = str(index).zfill(6)
            
            # 1.points reader
            if raw_data_type == 'data_6018':
                pcd_path = os.path.join(cfg.root_path,label_list[index].split('/')[-3],'{}.pcd'.format(time_stamp))
            elif raw_data_type == 'data_60':
                pcd_path = os.path.join(cfg.root_path,'{}.pcd'.format(time_stamp))
            else:
                pcd_path = pc_hash_table[time_stamp]
            
            label_path = label_list[index]
            label = json_reader(label_path)
    
            if not os.path.exists(pcd_path):
                raise ValueError(f"The frame {time_stamp} pointcloud is missing!")
                # print(f"The frame {time_stamp} pointcloud is missing!")
                # return
            if DEBUG:
                import time
                start_time = time.time()
            points = DataReader.pcd_reader(pcd_path=pcd_path)
            points= RawDataReader.raw_to_kitti_calib(points)
            points.tofile(os.path.join(all_sensor_path['velodyne'],'{}.bin'.format(filename)))
            if DEBUG:
                pcd_time = time.time()
                print(f'Pcd time is {pcd_time - start_time}')

            # 2.Image reader
            jpg_path = os.path.join(label_path.split('/point_cloud_3d')[0],'images/front-wide','{}.jpg'.format('*'))
            jpg_path = glob.glob(jpg_path)[0]
            image_reader(jpg_path,all_sensor_path['image_2'],filename)
            
            jpg_path = os.path.join(label_path.split('/point_cloud_3d')[0],'images/left-forward','{}.jpg'.format('*'))
            jpg_path = glob.glob(jpg_path)[0]
            image_reader(jpg_path,all_sensor_path['image_0'],filename)
            
            jpg_path = os.path.join(label_path.split('/point_cloud_3d')[0],'images/left-backward','{}.jpg'.format('*'))
            jpg_path = glob.glob(jpg_path)[0]
            image_reader(jpg_path,all_sensor_path['image_1'],filename)
            
            jpg_path = os.path.join(label_path.split('/point_cloud_3d')[0],'images/right-forward','{}.jpg'.format('*'))
            jpg_path = glob.glob(jpg_path)[0]
            image_reader(jpg_path,all_sensor_path['image_3'],filename)
            if DEBUG:
                img_time = time.time()
                print(f'Image time is {img_time - pcd_time}')
            # 3.Label reader

            label_save_path = os.path.join(all_sensor_path['label_2'], '{}.txt'.format(filename))
            if raw_data_type == 'data_60':
                RawDataReader.json2txt_60(label,label_save_path)
            else:
                RawDataReader.json2txt_fusion(label,label_save_path)
            if DEBUG:
                lab_time = time.time()
                print(f'Image time is {lab_time - img_time}')
            # 4.Calib reader
            calib_path = calib_list[index]
            calib_targ_pth = os.path.join(all_sensor_path['calib'])
            CalibExtractor(src_pth=calib_path,
                           targ_dir=calib_targ_pth,
                           filename='{}.txt'.format(filename))
            if DEBUG:
                cal_time = time.time()
                print(f'Image time is {cal_time - lab_time}')
def main():
    
    # test
    for i in tqdm(image_ids):
        if i > 30:
            break
        reader_2_writer(i)
        
    # num_workers = 4
    # # image_ids = image_ids[:20]
        
    # def run(f, my_iter):
    #     with futures.ProcessPoolExecutor(num_workers) as executor:
    #         list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
    # run(reader_2_writer,image_ids)   

if __name__ == "__main__":
    main()  

