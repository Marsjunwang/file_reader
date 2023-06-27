from turtle import distance
import numpy as np
import os
import json


file = 'E:/data/4D_radar/CUBTEK 4D RADAR 20220809_114929_Obj.log'
save_path = 'E:/a_project/file_reader/4d_radar'
with open(file,'r',encoding='UTF-8') as f:
    radar_raw = f.readlines()
dict_list = list()
for i in radar_raw:
    dict_list.append(json.loads(i))
dict_list = np.array(dict_list)

for id,frame in enumerate(dict_list):
    points = []
    time_stamp = frame['parsedSOMEIPData']['Time_Stamp_Second']
    for raw_point in frame['parsedSOMEIPData']['DetectionList']:
        distance_point = raw_point['Radial_Distance']
        Azimuth = raw_point['Azimuth']
        Elevation = raw_point['Elevation']
        # x front;y left; z up
        x = distance_point * np.math.cos(Elevation) * np.math.cos(Azimuth)
        y = distance_point * np.math.cos(Elevation) * np.math.sin(Azimuth)
        z = np.math.sin(Elevation) * distance_point
        Radar_Cross_Section = raw_point['Radar_Cross_Section']
        Relative_Velocity = raw_point['Relative_Velocity']
        Signal_To_Noise_Ratio = raw_point['Existence_Probability']
        point = np.array([x, y, z, Radar_Cross_Section, Relative_Velocity, Signal_To_Noise_Ratio])
        points.append(point)
    points = np.array(points,dtype=np.float32)
    points.tofile(os.path.join(save_path,'{}_{}.bin'.format(time_stamp,str(id).zfill(5))))
    print(time_stamp,id)




#dict_list[0]['parsedSOMEIPData']['DetectionList'][0].keys()
#dict_keys(['Azimuth', 'Classification_Type', 'Detection_ID', 'Detection_Valid', 'Elevation', 
# 'Existence_Probability', 'Multi_Target_Probability', 'Object_ID_Reference', 'Radar_Cross_Section', 
# 'Radial_Distance', 'Relative_Velocity', 'Reserved1', 'Signal_To_Noise_Ratio'])
#dict_list[0]['header']
#dict_keys(['ClientID', 'InterfaceVer', 'Length', 'MessageType', 'MethodID', 'ProtocolVer', 'ReturnCode', 'ServiceID', 'SessionID'])
