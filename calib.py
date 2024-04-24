import cv2
import numpy as np
import os
from tqdm import trange
from scipy.spatial.transform import Rotation as R

def calib():
    data_path = "input/"
    param_txt_path = "input/param.txt"
    output="output/"

    # 用于从点云生成标定数据，并保存cam2tcp
    cam2tcp_rmtx,cam2tcp_tvec=calibrateHandEye_using_files(data_path, param_txt_path)

    camera_mtx, camera_dist = Loading_Params_From_Txt(param_txt_path)
    New_Generate_Pointclouds(data_path,output, cam2tcp_rmtx, cam2tcp_tvec, camera_mtx, camera_dist)


def calibrateHandEye_using_files(data_path,param_txt_path):
    camera_mtx, camera_dist = Loading_Params_From_Txt(param_txt_path)
    print("Camera Mtx:", camera_mtx)
    print("Camera Dist:", camera_dist)

    tcp2base_rmtxs, tcp2base_rvecs, tcp2base_tvecs = Get_TCP2Base_Transform_From_Json(data_path)
    board2cam_rmtxs, board2cam_tvecs = Get_Board2Cam_Transform(data_path, camera_mtx, camera_dist,use_2D=True)
    # Not robust for computing pose for Camera 2 Board
    Transform_BoardCenters_FromCameraToBoard(data_path, camera_mtx, camera_dist, use_2D=True)

    cam2tcp_rmtx, cam2tcp_tvec = cv2.calibrateHandEye(tcp2base_rmtxs, tcp2base_tvecs, board2cam_rmtxs, board2cam_tvecs,
                                                      method=cv2.CALIB_HAND_EYE_PARK)
    cam2tcp_rmtx = np.matrix(cam2tcp_rmtx)
    cam2tcp_tvec = np.matrix(cam2tcp_tvec)

    cam2tcp_rvec = get_rvec(cam2tcp_rmtx)
    print('Cam2Tcp的数据如下(R、T)：')
    print("Cam2Tcp R:", cam2tcp_rmtx)
    print("Cam2Tcp T:", cam2tcp_tvec)

    data=[]
    for i in range(3):
        for j in range(3):
            data.append(round(cam2tcp_rmtx[i, j],8))
    for g in range(3):
            data.append(round(cam2tcp_tvec[g, 0],8))

    file_path = os.path.join(os.getcwd(), 'output', 'result.txt')
    with open(file_path, 'w') as file:
        for item in data:
            file.write(str(item) + '\n')

    return cam2tcp_rmtx, cam2tcp_tvec
def get_rmtx(rvec):
    check_rvec(rvec)
    rmtx, _ = cv2.Rodrigues(rvec)
    rmtx = np.matrix(rmtx)
    check_rmtx(rmtx)
    return rmtx



def get_rvec(rmtx):
    check_rmtx(rmtx)
    rvec, _ = cv2.Rodrigues(rmtx)
    rvec = np.matrix(rvec)
    check_rvec(rvec)
    return rvec

def InvTransformRT(R, T):
    check_rmtx(R)
    check_tvec(T)

    R_inv = R.T
    T_inv = -R_inv * T

    return R_inv, T_inv

def check_tvec(tvec):
    assert tvec.__class__.__name__ == 'matrix'
    assert tvec.ndim == 2
    assert tvec.shape == (3, 1)
    assert tvec.dtype == np.float32 or tvec.dtype == np.float64

def check_rvec(rvec):
    assert rvec.__class__.__name__ == 'matrix'
    assert rvec.ndim == 2
    assert rvec.shape == (3, 1)
    assert rvec.dtype == np.float32 or rvec.dtype == np.float64

def check_rmtx(rmtx):
    assert rmtx.__class__.__name__ == 'matrix'
    assert rmtx.ndim == 2
    assert rmtx.shape == (3, 3)
    assert rmtx.dtype == np.float32 or rmtx.dtype == np.float64

def get_rvec_Yaskawa(rx, ry, rz):
    euler_angle = [rx, ry, rz]
    rmtx = R.from_euler('xyz', euler_angle, degrees=True).as_matrix()
    rmtx = np.matrix(rmtx, dtype=np.float32)
    rvec, _ = cv2.Rodrigues(rmtx)
    rvec = np.matrix(rvec)

    check_rvec(rvec)
    return rvec

def Loading_Depth_From_Tiff(depth_file):
    depth = cv2.imread(depth_file, -1)
    depth = np.float32(np.array(depth))

    return depth

def interp2d(image, point):
    x = point[0]
    y = point[1]

    x0 = int(x)
    x1 = int(x + 1)
    y0 = int(y)
    y1 = int(y + 1)

    x0_weight = x - x0
    x1_weight = x1 - x
    y0_weight = y - y0
    y1_weight = y1 - y

    value = image[y0, x0] * y1_weight * x1_weight + \
            image[y1, x0] * y0_weight * x1_weight + \
            image[y1, x1] * y0_weight * x0_weight + \
            image[y0, x1] * y1_weight * x0_weight
    return value


# get Transform from pc2 to pc1 R * pc2 + t = pc1
# numpy version
#   pc1: np.array  [N, 3]
#   pc2: np.array  [N, 3]
def svdICP(pc1, pc2):
    pc1Center = pc1.mean(axis=0)
    pc2Center = pc2.mean(axis=0)
    pc1Centered = pc1 - pc1Center
    pc2Centered = pc2 - pc2Center
    # batch matrix multiplication using broadcasting
    print("svd测试!!!", pc2Centered[:, :, None])
    print("svd测试!!!", pc1Centered[:, None, :])
    w = pc2Centered[:, :, None] * pc1Centered[:, None, :]
    w = w.sum(axis=0)
    # print(w)
    u, sigma, vT = np.linalg.svd(w)
    # relativeR = u @ vT
    relativeR = vT.T @ u.T
    detR = np.linalg.det(relativeR)
    if detR < 0:
        vT[2, :] = -vT[2, :]
        relativeR = vT.T @ u.T
    relativeT = pc1Center - (relativeR @ pc2Center[:, None]).reshape(-1)

    return relativeR, relativeT

def Get_Board2Cam_Transform_2D(color_img_path, camera_mtx, camera_dist):
    # print("Using 2D-3D to compute the Transform Matrix between Board and Camera")
    objectpoints = CircleObjectPointsGenerate(11, 7, 20, 20)
    color_img = cv2.imread(color_img_path, 0)
    color_img = 255 - color_img
    ret, centers = cv2.findCirclesGrid(color_img, (7, 11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    _, rvec, tvec = cv2.solvePnP(objectpoints, centers, camera_mtx, camera_dist)
    rmtx = get_rmtx(np.matrix(rvec))
    tvec = np.matrix(tvec)
    check_rmtx(rmtx)
    check_tvec(tvec)

    return rmtx, tvec

def Get_Board2Cam_Transform_3D(color_img_path, depth_img_path, camera_mtx, camera_dist):
    # print("Using 3D-3D to compute the Transform Matrix between Board and Camera")
    objectpoints = CircleObjectPointsGenerate(11, 7, 20, 20)
    color_img = cv2.imread(color_img_path, 0)
    color_img = 255 - color_img
    _, centers = cv2.findCirclesGrid(color_img, (7, 11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    depth = Loading_Depth_From_Tiff(depth_img_path)
    # depth = Depth_Middle_Fliter(depth)
    # Filter is needed for depth?
    centers_undistort = cv2.undistortPoints(centers, camera_mtx, camera_dist)
    centers_3d = []
    for i in range(len(centers)):
        points = centers[i, 0, :]
        z = interp2d(depth, points)
        x = centers_undistort[i, 0, 0] * z
        y = centers_undistort[i, 0, 1] * z
        pts = np.array([x, y, z])
        centers_3d.append(pts)
    centers_3d = np.array(centers_3d)
    rmtx, tvec = svdICP(centers_3d, objectpoints)
    rmtx = np.matrix(rmtx)
    tvec = np.matrix(tvec.reshape(3, 1))
    check_tvec(tvec)
    check_rmtx(rmtx)

    return rmtx, tvec

def Loading_Params_From_Txt(params_file):
    params = np.loadtxt(params_file)
    camera_mtx = np.zeros((3, 3))
    camera_mtx[0, 0] = params[0]
    camera_mtx[0, 2] = params[2]
    camera_mtx[1, 1] = params[4]
    camera_mtx[1, 2] = params[5]
    camera_mtx[2, 2] = 1
    camera_dist = params[9:14]

    return camera_mtx, camera_dist

def Get_TCP2Base_Transform_From_Json(data_path):

    tcp2base_rmtxs = []
    tcp2base_rvecs = []
    tcp2base_tvecs = []
    tex = []

    for i in range(8):
        tex.append(f'input/pos{i}.txt')

    # for root, dirs, files in os.walk(data_path):
    #     for file in files:
    #         if file.startswith("pos") and file.endswith(".txt"):
    #             pos_num = int(file[3:4])
    #             if pos_num >= 0 and pos_num <= 7:
    #                 tex.append(os.path.join(root, file))
    # tex = sorted(tex, key=lambda x: int(x.split("/")[-1][3:4]))
    print('txt')
    print(tex)
    for i in range(8):
        data = tex[i]
        #print(data)

        f = open(data, "r")
        lines = f.readlines()  # 读取全部内容
        list1 = []
        for line in lines:
            list1.append(line.strip().split('\t'))
        rvec = get_rvec_Yaskawa(list1[0][0], list1[1][0], list1[2][0])
        tvec = np.matrix([list1[3][0], list1[4][0], list1[5][0]], dtype=np.float32).T
        rmtx = get_rmtx(rvec)
        tcp2base_rvecs.append(rvec)
        tcp2base_rmtxs.append(rmtx)
        tcp2base_tvecs.append(tvec)

    return tcp2base_rmtxs, tcp2base_rvecs, tcp2base_tvecs

def Get_Board2Cam_Transform(data_path, camera_mtx, camera_dist, use_2D=False, use_3D=False):
    board2cam_rmtxs = []
    board2cam_tvecs = []


    file_list = []

    for i in range(8):
        file_list.append(f'input/pos{i}.bmp')

    # for file in os.listdir(data_path):
    # file_path = os.path.join(data_path, file)
    # if os.path.isfile(file_path) and file.endswith('.bmp'):
    # file_list.append(file_path)
    # file_list = sorted(file_list, key=lambda x: int(x.split("/")[-1][3:4]))
    print("bmp")
    print(file_list)

    for i in range(8):
        color_img_path = file_list[i]
        depth_img_path = file_list[i]
        if use_2D:
            rmtx, tvec = Get_Board2Cam_Transform_2D(color_img_path, camera_mtx, camera_dist)
        elif use_3D:
            rmtx, tvec = Get_Board2Cam_Transform_3D(color_img_path, depth_img_path, camera_mtx, camera_dist)
        else:
            raise NotImplementedError("Not implented yet")

        board2cam_rmtxs.append(rmtx)
        board2cam_tvecs.append(tvec)

    return board2cam_rmtxs, board2cam_tvecs

def Transform_BoardCenters_FromCameraToBoard(data_path, camera_mtx, camera_dist, use_2D=False,
                                             use_3D=False):
    objectpoints = CircleObjectPointsGenerate(11, 7, 20, 20)
    total_error_list = []


    file_list = []
    file_list = []
    for i in range(8):
        file_list.append(f'input/pos{i}.bmp')
    # for file in os.listdir(data_path):
    #     file_path = os.path.join(data_path, file)
    #     if os.path.isfile(file_path) and file.endswith('.bmp'):
    #         file_list.append(file_path)
    # file_list = sorted(file_list, key=lambda x: int(x.split("/")[-1][3:4]))
    print("mean")
    print(file_list)
    for i in range(8):
        color_img_path = file_list[i]
        depth_img_path = file_list[i]
        if use_2D:
            rmtx, tvec = Get_Board2Cam_Transform_2D(color_img_path, camera_mtx, camera_dist)
        elif use_3D:
            rmtx, tvec = Get_Board2Cam_Transform_3D(color_img_path, depth_img_path, camera_mtx, camera_dist)
        else:
            raise NotImplementedError("Not implented yet")
        # print("Cam2Board Rmtx:", rmtx[0,0])
        # print("Cam2Board Tvec:", tvec[0,0])
        centers_3d = Get_Board_Centers_3D(color_img_path, depth_img_path, camera_mtx, camera_dist)
        cam2board_rmtx, cam2board_tvec = InvTransformRT(rmtx, tvec)
        centers_3d = np.matrix(centers_3d).T
        centers_3d = cam2board_rmtx * centers_3d + cam2board_tvec
        centers_3d = np.array(centers_3d).T
        # print("Centers_3d:", centers_3d[0])
        # print("Objectpoints:", objectpoints[0])
        # print(np.linalg.norm(centers_3d-objectpoints, axis=1, keepdims=True))
        error_list = np.linalg.norm(centers_3d - objectpoints, axis=1, keepdims=True)
        #print("Current error when transform centers from Camera to Board:", np.mean(error_list))
        total_error_list.append(error_list)
        # np.savetxt(os.path.join(data_path, files[key]+'_center_board_3D.xyz'), centers_3d)

    total_error_list = np.array(total_error_list)
    # print(total_error_list.shape)
    avg_error = np.mean(total_error_list)
    #print("Average error when transform centers from Camera to Board:", avg_error)

    return 0

def CircleObjectPointsGenerate(xNum, yNum, xSpace, ySpace):
    objectpoints = []
    for j in range(xNum):
        for i in range(yNum):
            if j % 2 == 0:
                x = i * xSpace
            else:
                x = (i + 0.5) * xSpace
            y = j * ySpace * 0.5
            z = 0
            objectpoint = np.array([x, y, z])
            objectpoints.append(objectpoint)
    objectpoints = np.array(objectpoints)

    return objectpoints

def Get_Board_Centers_3D(color_img_path, depth_img_path, camera_mtx, camera_dist):
    color_img = cv2.imread(color_img_path, 0)
    color_img = 255 - color_img
    _, centers = cv2.findCirclesGrid(color_img, (7, 11), flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    depth = Loading_Depth_From_Tiff(depth_img_path)
    centers_undistort = cv2.undistortPoints(centers, camera_mtx, camera_dist)
    centers_3d = []
    for i in range(len(centers)):
        points = centers[i, 0, :]
        z = interp2d(depth, points)
        x = centers_undistort[i, 0, 0] * z
        y = centers_undistort[i, 0, 1] * z
        pts = np.array([x, y, z])
        centers_3d.append(pts)
    centers_3d = np.array(centers_3d)

    return centers_3d



# 这个函数可以生成保存没有畸变的、基于base坐标系的点云图，在拍照阶段可以调用
def New_Generate_Pointclouds(data_path,output, cam2tcp_rmtx, cam2tcp_tvec, camera_mtx,camera_dist):

    file_list = []
    file_list1 = []
    for i in range(8):
        file_list.append(f'input/pos{i}.bmp')
    for i in range(8):
        file_list1.append(f'input/pos{i}.tiff')
    # for root, dirs, files in os.walk(data_path):
    #
    #     for file in files:
    #         if file.endswith("bmp") :
    #             file_list.append(os.path.join(root, file))
    #     for file in files:
    #         if file.endswith("tiff"):
    #             file_list1.append(os.path.join(root, file))
    # file_list = sorted(file_list, key=lambda x: int(x.split("/")[-1][3:4]))
    # file_list1 = sorted(file_list1, key=lambda x: int(x.split("/")[-1][3:4]))
    print("生成点云")
    print(file_list)
    print(file_list1)
    tex = []
    for i in range(8):
        tex.append(f'input/pos{i}.txt')
    for i in range(8):
        color_img_path = file_list[i]
        depth_img_path = file_list1[i]
        color = cv2.imread(color_img_path)
        depth = Loading_Depth_From_Tiff(depth_img_path)
        pointcloud = Generate_Pointcloud_From_Depth_undistort(depth, color, camera_mtx, camera_dist)


        # for root, dirs, files in os.walk(data_path):
        #     for file in files:
        #         if file.startswith("pos") and file.endswith(".txt"):
        #             pos_num = int(file[3:4])
        #             if pos_num >= 0 and pos_num <= 7:
        #                 tex.append(os.path.join(root, file))
        #tex = sorted(tex, key=lambda x: int(x.split("/")[-1][3:4]))

        data = tex[i]
        print(data)

        f = open(data, "r")
        lines = f.readlines()  # 读取全部内容
        list1 = []
        for line in lines:
            list1.append(line.strip().split('\t'))
        tcp2base_rvec = get_rvec_Yaskawa(list1[0][0], list1[1][0], list1[2][0])
        tcp2base_tvec = np.matrix([list1[3][0], list1[4][0], list1[5][0]], dtype=np.float32).T
        tcp2base_rmtx = get_rmtx(tcp2base_rvec)
        # 通过计算得到需要的参数
        pointcloud_base = np.zeros_like(pointcloud)
        pts = pointcloud[:, :3]
        color = pointcloud[:, 3:]
        cam_pos_rmtx, cam_pos_tvec = Get_Camera_Position(cam2tcp_rmtx, cam2tcp_tvec, tcp2base_rmtx, tcp2base_tvec)

        # 计算得到需要的点云
        pts = np.matrix(pts).T
        pts = cam_pos_rmtx * pts + cam_pos_tvec
        pts = np.array(pts).T
        pointcloud_base[:, :3] = pts
        pointcloud_base[:, 3:] = color
        list=['pos0','pos1','pos2','pos3','pos4','pos5','pos6','pos7',]
        np.savetxt(os.path.join(output, list[i] + '_base_new.xyz'), pointcloud_base)

    return 0

def Generate_Pointcloud_From_Depth_undistort(depth, color, camera_mtx, camera_dist):
    pointcloud = []

    for iy in trange(1200):
        for ix in range(1920):
            # 先对图片去畸变
            ixiy = cv2.undistortPoints(np.float32(np.array([ix, iy])), camera_mtx, camera_dist)
            z = depth[iy, ix]
            r, g, b = color[iy, ix]
            if z > 0:
                x = ixiy[0, 0, 0] * z
                y = ixiy[0, 0, 1] * z
                pts = np.array([x, y, z, r, g, b])
                pointcloud.append(pts)

    pointcloud = np.array(pointcloud)
    return pointcloud

def Loading_Depth_From_Tiff(depth_file):
    depth = cv2.imread(depth_file, -1)
    depth = np.float32(np.array(depth))
    return depth

def Get_Camera_Position(cam2tcp_rmtx, cam2tcp_tvec, tcp2base_rmtx, tcp2base_tvec):
    cam_pos_rmtx = tcp2base_rmtx * cam2tcp_rmtx
    cam_pos_tvec = tcp2base_rmtx * cam2tcp_tvec + tcp2base_tvec
    return cam_pos_rmtx, cam_pos_tvec


if __name__ == '__main__':
    calib()

