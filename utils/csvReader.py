import json
import pandas as pd
import glob
import os
from pathlib import Path
from tqdm import tqdm


class CSV_IO():
    def __init__(self,
                 root_path='/media/jw11/jw11/data/csv_data',
                 save_path=None):
        
        self.data_list = self.data_reader(Path(root_path))
        if not save_path:
            self.save_path = root_path
        else:
            self.save_path = save_path
            
        file_label = {
                'labels':[]
                }
        
        self.file_label = self.csv_reader(self.data_list,
                                            file_label)
            
        
    def csv_reader(self,
                   file_list,
                   file_label):
        for i in file_list:
            csv_content = pd.read_csv(i, index_col=0)
            filename = os.path.splitext(os.path.basename(i))[0]
    
            for index in tqdm(set(csv_content.index)):
                content_per_frame = csv_content.loc[index]
                if content_per_frame.index.name == 'frame_num':
                    obejct_num = len(content_per_frame.index)
                else:
                    obejct_num = 1
                target = {
                    "center":{
                        "x":[],
                        "y":[],
                        "z":[]
                    },
                    "rotation":{
                        "pitch":[],
                        "roll":[],
                        "yaw":[]
                    },
                    "size":{
                        "x":[],
                        "y":[],
                        "z":[]
                    },
                    "tracker_id":[],
                    "type":[]
                    }
                object_per_frame_after = list()
                for v in range(obejct_num):
                    label_per_frame_before = content_per_frame.iloc[v]
                    if content_per_frame.index.name != 'frame_num':
                        label_per_frame_before = content_per_frame
                    target['center']['x'] = label_per_frame_before['center.x']
                    target['center']['y'] = label_per_frame_before['center.y']
                    target['center']['z'] = 0
                    
                    target['rotation']['pitch'] = label_per_frame_before['heading']
                    
                    target['size']['x'] = label_per_frame_before['length']
                    target['size']['y'] = label_per_frame_before['width']
                    target['size']['z'] = label_per_frame_before['height']
                    
                    target['tracker_id'] = label_per_frame_before['track_id']
                    target['type'] = label_per_frame_before['type']
                    
                    object_per_frame_after.append(target)
                file_label['labels'].append(object_per_frame_after)
                # break
            self.json_writer(self.save_path,
                             filename,
                             file_label)
            
            return file_label
            
    def json_writer(self,
                    save_path,
                    filename,
                    data):
        if not os.path.exists(save_path):
                os.mkdir(save_path)
        with open(os.path.join(save_path, '{}.json'.format(filename)),'w') as f:
                json.dump(data,f,indent=4)
     
    def data_reader(self,
                    root_path):
        ext = '.csv'
        data_list = glob.glob(str(root_path / '*{}'.format(ext))) if root_path.is_dir() else [root_path]
        return data_list

if __name__ == '__main__':
    CSV_IO()
    