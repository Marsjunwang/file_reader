import yaml
import os.path as osp
import os
import math
import numpy as np
import json
import cv2


class CalibExtractor:
    def __init__(self,
                 src_pth,
                 targ_dir,
                 filename='undefineFrameId-1652876697789456000-0',
                 undistort_image=False,
                 data_format='KITTI'):
        self.source_pth = src_pth
        self.target_pth = targ_dir
        self.filename = filename
        self.undistort_image = undistort_image
        self.source_calib = CalibExtractor.txtInput(src_pth)
        sensor_names, RTs, amend_matrixs, calib_cam2imgs = self.calibDecode()
        self.target_calib = self.calibEncode(sensor_names, RTs, amend_matrixs, calib_cam2imgs) if not self.undistort_image \
                                else self.calibEncodeUndistort(sensor_names, RTs, amend_matrixs, calib_cam2imgs)
        
        if not osp.exists(targ_dir):
            os.makedirs(targ_dir)
        if data_format == 'xj3_to_kitti':
            self.xj3_to_kitti_writer()
        elif data_format == 'SenseTime':
            for i,name in enumerate(sensor_names):
                save_path = osp.join(targ_dir,name)
                if not osp.exists(save_path):
                    os.makedirs(save_path) 
                save_path = osp.join(save_path, filename + '.json')
                CalibExtractor.jsonOutput(self.target_calib[i], save_path)
            if __name__ == '__main__':
                print(self.target_calib)
        else:
            raise ValueError(f'Data format:{data_format} is not support. Only support KITTI and SenseTime')
        
    def xj3_to_kitti_writer(self):
        save_path = self.target_pth
        st = ''
        for calib in self.target_calib:
            R = calib['R']
            P = calib['P']
            T = calib['T']
            name = calib['calName']
            P = name + '_P' + ':' + ' '  + ' '.join(map(str, [item for sublist in P for item in sublist])) + '\n'
            R = name + '_R' + ':' + ' '  + ' '.join(map(str, [item for sublist in R for item in sublist])) + '\n'
            T = name + '_T' + ':' + ' '  + ' '.join(map(str, [item for sublist in T for item in sublist])) + '\n'
            st += P + R + T
            with open(save_path + '/' + self.filename,'w') as f:
                f.write(st)     
    @staticmethod    
    def txtInput(pth):
        if not osp.exists(pth):
            raise ValueError(f'The path {pth} is not existing!')
        with open(pth, 'r', encoding='utf-8') as f:
            return f.readlines()
        
    @staticmethod    
    def jsonOutput(val, pth):
        # if not osp.exists(pth):
        #     raise ValueError(f'The path {pth} is not existing!')
        with open(pth, 'w', encoding='utf-8') as f:
            json.dump(val, f, ensure_ascii=False, indent=4)
        
    def calibDecode(self):
        sensor_name = []
        RTs = []
        amend_matrixs = []
        calib_cam2imgs = []
        
        for i in self.source_calib:
            calib = i.strip().split()
            if calib[0] == '#':
                continue
            elif calib[0][:5] == 'calib':
                calib_sencor = calib[0].split('_')[1]
                calib_RT = [float(v) for v in calib[1:]]
                sensor_name.append(calib_sencor)
                RTs.append(calib_RT)
            elif calib[0] == calib_sencor+'_D:':
                amend_matrix = [float(v) for v in calib[1:]]
                amend_matrixs.append(amend_matrix)
            elif calib[0] == calib_sencor+'_K:':
                calib_cam2img = [float(v) for v in calib[1:]]
                calib_cam2imgs.append(calib_cam2img)
            else:
                pass
        return sensor_name, RTs, amend_matrixs, calib_cam2imgs
    
    
    def quat2RTHm(self, quat):
        #right hand
        px, py, pz, rx, ry ,rz, w = quat[0], quat[1], quat[2], \
            quat[3], quat[4], quat[5], quat[6]
        RT = np.array([[0, 0, 0, px],
                        [0, 0, 0, py],
                        [0, 0, 0, pz],
                        [0,0,0,1]])
        
        RT[0, 0] = 1 - 2 * pow(ry, 2) - 2 * pow(rz, 2)
        RT[0, 1] = 2 * (rx * ry - w * rz)
        RT[0, 2] = 2 * (rx * rz + w * ry)
    
        RT[1, 0] = 2 * (rx * ry + w * rz)
        RT[1, 1] = 1 - 2 * pow(rx, 2) - 2 * pow(rz, 2)
        RT[1, 2] = 2 * (ry * rz - w * rx)
        
        RT[2, 0] = 2 * (rx * rz - w * ry)
        RT[2, 1] = 2 * (ry * rz + w * rx)
        RT[2, 2] = 1 - 2 * pow(rx, 2) - 2 * pow(ry, 2)
        if __name__ == '__main__':
            print(RT)
        # cam==>velo  converter to velo ==> cam
        RT = np.linalg.inv(RT)[:3,:]
        RT = RT[:3,:]
        return RT.tolist()
            
    def quat2RTJPL(self, quat):
        #left hand
        px, py, pz, rx, ry ,rz, w = quat[0], quat[1], quat[2], \
            quat[3], quat[4], quat[5], quat[6]
        RT = np.array([[0, 0, 0, px],
                        [0, 0, 0, py],
                        [0, 0, 0, pz],
                        [0,0,0,1]])
        
        RT[0, 0] = 1 - 2 * pow(ry, 2) - 2 * pow(rz, 2)
        RT[0, 1] = 2 * (rx * ry + w * rz)
        RT[0, 2] = 2 * (rx * rz - w * ry)
    
        RT[1, 0] = 2 * (rx * ry - w * rz)
        RT[1, 1] = 1 - 2 * pow(rx, 2) - 2 * pow(rz, 2)
        RT[1, 2] = 2 * (ry * rz + w * rx)
        
        RT[2, 0] = 2 * (rx * rz + w * ry)
        RT[2, 1] = 2 * (ry * rz - w * rx)
        RT[2, 2] = 1 - 2 * pow(rx, 2) - 2 * pow(ry, 2)
        if __name__ == '__main__':
            print(RT)
        # cam==>velo  to velo ==> cam
        RT = np.linalg.inv(RT)[:3,:]
        return RT.tolist()
    
    def camInstrinc(self, P):
        P = np.array(P).reshape(-1,3)
        self.camera_matrix = P
        tail = np.zeros((3,1))
        P = np.concatenate((P, tail), axis=1)
        return P.tolist()
        
    def calibEncode(self,sensor_name, RTs, amend_matrixs, calib_cam2imgs):
        target_calib = []
        for i in range(len(sensor_name)):
            calib = {
                'calName':sensor_name[i],
                    'timestamp': 0,
                    'P':self.camInstrinc(calib_cam2imgs[i]),
                    'R': [[
                            1,
                            0,
                            0
                        ],
                        [
                        0,
                        1,
                        0
                        ],
                        [
                        0,
                        0,
                        1
                        ]],
                'T':self.quat2RTHm(RTs[i])
            }

            target_calib.append(calib)
        return target_calib
    
    def calibEncodeUndistort(self,sensor_name, RTs, amend_matrixs, calib_cam2imgs):
        target_calib = []
        
        
        for i in range(len(sensor_name)):
            filename = self.filename + '.jpg'
            sensor_path = osp.join(osp.dirname(self.target_pth), 'image', sensor_name[i], filename)
            sensor_path_targ = osp.join(osp.dirname(self.target_pth), 'image', sensor_name[i])
            if not osp.exists(sensor_path_targ):
                os.makedirs(sensor_path_targ)
            sensor_path_targ = sensor_path_targ + '/' + filename
            
            img = cv2.imread(sensor_path)
            self.camInstrinc(calib_cam2imgs[i])
            undistorted_img, new_camera_matrix = CalibExtractor.undistort_image(img, self.camera_matrix,np.array(amend_matrixs[i]))
            os.remove(sensor_path)
            cv2.imwrite(sensor_path_targ, undistorted_img)
            
            calib = {
                'calName':sensor_name[i],
                    'timestamp': 0,
                    'P':new_camera_matrix,
                    'R': [[
                            1,
                            0,
                            0
                        ],
                        [
                        0,
                        1,
                        0
                        ],
                        [
                        0,
                        0,
                        1
                        ]],
                'T':self.quat2RTHm(RTs[i])
            }

            target_calib.append(calib)
        return target_calib
    
    
    @staticmethod
    def undistort_image(img, camera_matrix, dist_coeffs):
        h, w = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        x, y, w, h = roi
        undistorted_img = undistorted_img[y:y+h, x:x+w]
        tail = np.zeros((3,1))
        new_camera_matrix = np.concatenate((new_camera_matrix, tail), axis=1)
        
        # h, w = img.shape[:2]
        # # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        # undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
        # # undistorted_img = undistorted_img[y:y+h, x:x+w]
        # tail = np.zeros((3,1))
        # new_camera_matrix = np.concatenate((camera_matrix, tail), axis=1)
        return undistorted_img, new_camera_matrix.tolist()
        

if __name__ == '__main__':
   source_path = '/media/jw11/jw13/a_project_/FileConvert/subset-131-837-92/original_data/params/params-to-sensetime.txt'
   target_path = './test'
   extractor = CalibExtractor(src_pth=source_path,
                              targ_pth=target_path)