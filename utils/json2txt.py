import glob
import os
import json
import numpy as np

'''
原始标注格式：
|--PCD*/
    |--1/
        |--**.json
        |--**.json
        |--**.json
    |--2/
    |--3/
    |--4/
'''

# 创建保存路径
dir_list = os.listdir('/media/jw11/jw11/data/6018_frames_data/label_baidu_0610_6018')
save_path = os.path.join('/media/jw11/jw11/data/converter_data')
save_list = os.listdir(save_path)
if 'baidu_new_txt' not in save_list:
    os.mkdir(os.path.join(save_path,'baidu_new_txt'))
save_path = os.path.join(save_path,'baidu_new_txt')
# 标注文件夹路径
pth='/media/jw11/jw11/data/6018_frames_data/label_baidu_0610_6018/PCD*'
# 标注次级文件夹
pths = glob.glob(pth)

for p in pths:
    n_p = glob.glob(os.path.join(p,'*/'))
    for n_ph in n_p:
        js_nm = glob.glob(os.path.join(n_ph,'*.json'))
        for nm in js_nm:
            f = open(nm,'r')
            js = json.load(f)
            num_lbs = len(js['labels'])
            f_w = open(save_path+'/'+nm.split('/')[-1][:-4]+'txt', 'w')
            lis_st = []
            for lb in js['labels']:
                st = lb['type']+' 0'*7+' '+\
                    str(lb['size']['x'])+' '+\
                    str(lb['size']['y'])+' '+\
                    str(lb['size']['z'])+' '+\
                    str(lb['center']['x'])+' '+\
                    str(lb['center']['y'])+' '+\
                    str(lb['center']['z'])+' '+\
                    str(lb['rotation']['yaw']+ np.pi / 2)+'\n'
                lis_st.append(st)
            f_w.writelines(lis_st)
            f_w.close()