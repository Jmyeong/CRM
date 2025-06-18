import os
import cv2
import yaml
import math
from tqdm import tqdm
import numpy as np
import argparse
from PIL import Image
import glob

class CameraFisheye:
    def __init__(self, calibration_file):
        self.calibration_file = calibration_file

        with open(self.calibration_file) as f:
            calibration = yaml.load(f, Loader=yaml.FullLoader)
            
        self.camera_id = calibration[0]['camera']
        
        roll = calibration[5]['extrinsic_parameters'][0]['roll']
        print(f"roll : {roll}")
        R_x = [[1, 0, 0],[0, math.cos(roll), -math.sin(roll)],[0, math.sin(roll), math.cos(roll)]]
        pitch = calibration[5]['extrinsic_parameters'][1]['pitch']
        print(f"pitch : {pitch}")
        R_y = [[math.cos(pitch), 0, math.sin(pitch)],[0, 1, 0],[-math.sin(pitch), 0, math.cos(pitch)]]
        yaw = calibration[5]['extrinsic_parameters'][2]['yaw']
        print(f"yaw : {yaw}")
        R_z = [[math.cos(yaw), -math.sin(yaw), 0],[math.sin(yaw), math.cos(yaw), 0],[0, 0, 1]]
        R = np.matmul(np.array(R_z),np.matmul(np.array(R_y),np.array(R_x)))
        Trans = np.array([calibration[5]['extrinsic_parameters'][3]['x'], calibration[5]['extrinsic_parameters'][4]['y'], calibration[5]['extrinsic_parameters'][5]['z']])
        Trans_inv = -np.matmul(np.linalg.inv(R), Trans.T)
        #print(R)
        #print(Trans)
        self.TrLidarToCam = np.column_stack((np.linalg.inv(R), Trans_inv.T))
        self.TrLidarToCam = np.row_stack((self.TrLidarToCam, [0.0, 0.0, 0.0, 1.0]))
        #print(self.TrLidarToCam)
        #Trans_inv_test = -np.matmul(np.linalg.inv(self.TrLidarToCam[:3,:3]), self.TrLidarToCam[:3,3].T)
        #self.TrCamToLidar = np.column_stack((np.linalg.inv(self.TrLidarToCam[:3,:3]), Trans_inv_test.T))
        #self.TrCamToLidar = np.row_stack((self.TrCamToLidar, [0.0, 0.0, 0.0, 1.0]))
        #print(self.TrCamToLidar)
        
        self.width, self.height = calibration[1]['image_width'], calibration[2]['image_height']
        
        self.fi = {'fx':float(calibration[4]['intrinsic_parameters'][0]['fx']), 'fy':float(calibration[4]['intrinsic_parameters'][1]['fy']), 'cx':float(calibration[4]['intrinsic_parameters'][2]['cx']),'cy':float(calibration[4]['intrinsic_parameters'][3]['cy']),'k1':float(calibration[3]['distortion_parameters'][0]['k1']),'k2':float(calibration[3]['distortion_parameters'][1]['k2']),'k3':float(calibration[3]['distortion_parameters'][2]['t1']),'t1':float(calibration[3]['distortion_parameters'][3]['t2']),'t2':float(calibration[3]['distortion_parameters'][4]['k3'])}
        
    def cam2image(self, points):
        ''' camera coordinate to image plane '''
        points = points.T
        #norm = np.linalg.norm(points, axis=1)
        
        #x = points[:,0] / norm
        #y = points[:,1] / norm
        #z = points[:,2] / norm
        
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        
        x /= z
        y /= z

        k1, k2, k3, t1, t2 = self.fi['k1'], self.fi['k2'], self.fi['k3'], self.fi['t1'], self.fi['t2']
        fx, fy, cx, cy = self.fi['fx'], self.fi['fy'], self.fi['cx'], self.fi['cy']

        ro2 = x*x + y*y
        
        #x = (1 + k1*ro2 + k2*ro2*ro2 + k3*ro2*ro2*ro2)*x + 2*t1*x*y + t2*(ro2 + 2*x*x)
        #y = (1 + k1*ro2 + k2*ro2*ro2 + k3*ro2*ro2*ro2)*y + t1*(ro2 + 2*y*y) + 2*t2*x*y
        
        #x = (1 + k1*ro2 + k2*ro2*ro2 + k3*ro2*ro2*ro2)*x
        #y = (1 + k1*ro2 + k2*ro2*ro2 + k3*ro2*ro2*ro2)*y 
        
        #x = (1 + k1*ro2 + k2*ro2*ro2)*x
        #y = (1 + k1*ro2 + k2*ro2*ro2)*y 
        
        x = (1 + k1*ro2)*x
        y = (1 + k1*ro2)*y 
        
        u = fx*x + cx
        v = fy*y + cy

        #return u, v, norm * points[:,2] / np.abs(points[:,2])
        return u, v, z

def loadpointcloud(bin_path):
    pcdFile = os.path.join(bin_path)
    if not os.path.isfile(pcdFile):
        raise RuntimeError('%s does not exist!' % pcdFile)
    pcd = np.fromfile(pcdFile, dtype=np.float32)
    # intensity = np.random.rand(50253)   # intensity 값 (각 포인트에 대한 강도)
    # # intensity = intensity[:, np.newaxis]
    # print(pcd.shape, intensity.shape)
    # pcd = np.hstack((pcd, intensity))

    # pcd = np.hstack((pcd, np.ones(pcd.shape[0], 1)))  # 4번째 열을 추가하여 1을 채워 넣음
    # print(pcd.shape)
    pcd = np.reshape(pcd,[-1,3])
    return pcd
    
def generate_depthmap(args, camera, points, idx, custom_vis=True):

    threshold = 900
    """
    #print(points.shape) #(32768, 4)
    criteria = points[:,0] > 0
    mask = np.full_like(criteria, 1e-7)
    points[:,0] = np.where(criteria, points[:,0], mask)
    points[:,1] = np.where(criteria, points[:,1], mask)
    points[:,2] = np.where(criteria, points[:,2], mask)
    """
    
    pointsCam = np.matmul(camera.TrLidarToCam, points.T).T
    pointsCam = pointsCam[:,:3]
    
    
    u,v,depth= camera.cam2image(pointsCam.T)
    u = u.astype(np.int64)
    v = v.astype(np.int64)
    # print(f"u : {u.shape}, v :{v.shape}")
    # prepare depth map for visualization
    depthMap = np.zeros((camera.height, camera.width))
    depthImage = np.zeros((camera.height, camera.width, 3))
    mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<camera.width), v>=0), v<camera.height)

    # visualize points within 80 meters
    # OS0 = 50m, OS1 = 120m, OS2 = 240m
    # OS0 = 35m(range at 10%)
    #mask = np.logical_and(np.logical_and(mask, depth>0), depth<80)
    mask = np.logical_and(np.logical_and(mask, depth>0), depth<40)
    
    # @@@ projection boundary by threshold @@@
    # mask = np.logical_and(mask, ((u-camera.fi['cx'])**2+(v-camera.fi['cy'])**2)**(1/2) < threshold)
    depthMap[v[mask],u[mask]] = depth[mask]
    # print(depthMap.shape)
    # @@@ custom visualization @@@
    if custom_vis:
        camera_name = (camera.camera_id).replace('zed2i','camera')
        imagePath = os.path.join(args.input_path,f'image_03/%010d.png' % int(idx))
        
        if not os.path.isfile(imagePath):
            raise RuntimeError('Image file %s does not exist!' % imagePath)
    
        colorImage = np.array(Image.open(imagePath)) / 255.
    
        visualImage = np.full_like(colorImage, 1)
        visualImage[depthMap>0] = depthImage[depthMap>0]
        # Kernel
        '''
        kernel = np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]])
        '''
        #'''
        kernel = np.array([[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]])
        #'''
 
        visualImage = cv2.filter2D(visualImage, -1, kernel)
        print(np.unique(visualImage))
        visualImage_norm = (visualImage - visualImage.min()) / (visualImage.max() - visualImage.min())
        print(np.unique(visualImage_norm))
        visualImage_norm = (visualImage_norm*255).astype(np.uint8)
        print(np.unique(visualImage_norm))
        visualImage_gray = cv2.cvtColor(visualImage_norm, cv2.COLOR_BGR2GRAY)
        visualImage_color = cv2.applyColorMap(visualImage_norm, cv2.COLORMAP_JET)
        colorImage[visualImage<250] = visualImage_color[visualImage<250]
        
        colorImage = np.float32(colorImage)
        colorImage = cv2.cvtColor(colorImage, cv2.COLOR_RGB2BGR)
        cv2.imshow('colorImage', colorImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # assert False
    else :
        return depthMap

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    이미지를 0~255 범위로 정규화하는 함수.
    
    Args:
    - image (np.ndarray): 정규화할 이미지 (0~1 범위로 가정)
    
    Returns:
    - np.ndarray: 0~255 범위로 정규화된 이미지
    """
    # 이미지가 0~1 사이의 값으로 가정하고 정규화
    normalized_image = (image - image.min()) / (image.max() - image.min())  # 0~1 정규화
    normalized_image = (normalized_image * 255).astype(np.uint8)  # 0~255 범위로 변환 후 uint8로 변환
    return normalized_image

def rotation_matrix_y(angle):
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return np.array([
        [cos_angle, 0, sin_angle],
        [0, 1, 0],
        [-sin_angle, 0, cos_angle]
    ])

def rotation_matrix_x(angle):
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, cos_angle, -sin_angle],
        [0, sin_angle, cos_angle]
    ])

def rotation_matrix_z(theta):
    """
    Z축을 기준으로 회전하는 3x3 회전 행렬을 생성합니다.
    
    :param theta: 회전 각도 (라디안 단위)
    :return: Z축 회전 행렬
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
# 포인트클라우드를 회전시키는 함수
def rotate_point_cloud(points, x_angle, y_angle, z_angle):
    R = rotation_matrix_x(x_angle)
    points = np.dot(points, R.T)  # 점들의 좌표에 회전 행렬 적용
    R = rotation_matrix_y(y_angle)  # Y축을 기준으로 회전
    points = np.dot(points, R.T)  # 점들의 좌표에 회전 행렬 적용
    R = rotation_matrix_y(z_angle)  # Y축을 기준으로 회전
    rotated_points = np.dot(points, R.T)  # 점들의 좌표에 회전 행렬 적용

    return rotated_points

def main():
    parser = argparse.ArgumentParser(description="data setup from pairing file.")
    parser.add_argument("--input_path", help="input path", type=str, required=True)
    parser.add_argument("--custom_vis", action='store_true')
    
    args = parser.parse_args()
    i = 0
    for dirpath, dirnames, filenames in os.walk(args.input_path):
        # print(dirnames)
        if i < 2:
            for dirname in dirnames:
                print(dirname)
                if dirname == "image_02" or dirname == "image_03":
                    # realsense_images = sorted(os.listdir(os.path.join(args.input_path,dirname,"data")))
                    realsense_images = sorted(glob.glob(os.path.join(args.input_path,dirname,"data/" + "*.png")))
                    # print(realsense_images)
                    print(f"len images : {len(realsense_images)}")
                    bins = sorted(os.listdir(os.path.join(args.input_path,'ouster_bins/data')))
                    calibration = os.path.join(args.input_path.replace(args.input_path.split('/')[-2],''), 'calibration')
                    print(calibration)          
                    # os.makedirs(realsense_depthmap_path, exist_ok=True)
                    # print(realsense_depthmap_path)
                    if dirname == "image_02":
                        realsense_cam = CameraFisheye(os.path.join(calibration,'zed2i_left.yaml'))
                    elif dirname == "image_03":
                        realsense_cam = CameraFisheye(os.path.join(calibration,'zed2i_right.yaml'))

                    
                    file_num = len(realsense_images)
                    print(f'Total file number is {file_num} for each camera')
                    
                    for idx in tqdm(range(file_num)):
                        x_angle = math.radians(5) # 5
                        y_angle = math.radians(-10) # -10
                        z_angle = math.radians(320) # 320
                        points = loadpointcloud(os.path.join(args.input_path,'ouster_bins/data',bins[idx]))
                        points = rotate_point_cloud(points, x_angle, y_angle, z_angle)
                        points = np.hstack((points, np.ones((points.shape[0], 1))))  # 새로운 열을 추가

                        points[:,3] = 1
                        realsense_depthmap = generate_depthmap(args, realsense_cam, points, idx, custom_vis=args.custom_vis)
                        realsense_depthmap = realsense_depthmap.astype(np.float32)
                        
                        realsense_depthmap_path = os.path.join(args.input_path, f'proj_depth/groundtruth/{dirname}',"%010i.npy" % idx)
                        # print(np.unique(realsense_depthmap))
                        # print(realsense_depthmap_path)
                        # cv2.imwrite(realsense_depthmap_path, realsense_depthmap, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        np.save(realsense_depthmap_path, realsense_depthmap)
        i += 1
    
if __name__ == '__main__':
    main()
