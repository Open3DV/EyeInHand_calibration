import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import json

# 重要提示：
# rvec = rotation vector 旋转向量。程序中必须是3x1的np.matrix形式
# tvec = transformation vector 平移向量。程序中必须是3x1的np.matrix形式
# rmtx = rotation matrix 旋转矩阵。程序中必须是3x3的np.matrix形式
# htmtx = homogeneous transformation matrix 齐次转换矩阵。程序中必须是4x4的np.matrix形式
# 每次使用上述变量时，都要调用check_xxx函数加以验证

def check_rmtx(rmtx):
    assert rmtx.__class__.__name__ == 'matrix'
    assert rmtx.ndim == 2
    assert rmtx.shape == (3, 3)
    assert rmtx.dtype == np.float32 or rmtx.dtype == np.float64
    
def check_rvec(rvec):
    assert rvec.__class__.__name__ == 'matrix'
    assert rvec.ndim == 2
    assert rvec.shape == (3, 1)
    assert rvec.dtype == np.float32 or rvec.dtype == np.float64

    
def check_tvec(tvec):
    assert tvec.__class__.__name__ == 'matrix'
    assert tvec.ndim == 2
    assert tvec.shape == (3, 1)
    assert tvec.dtype == np.float32 or tvec.dtype == np.float64
    
def check_htmtx(htmtx):
    assert htmtx.__class__.__name__ == 'matrix'
    assert htmtx.ndim == 2
    assert htmtx.shape == (4, 4)
    assert htmtx.dtype == np.float32 or htmtx.dtype == np.float64

def get_rvec_Yaskawa(rx, ry, rz):
    euler_angle = [rx, ry, rz]
    rmtx = R.from_euler('xyz', euler_angle, degrees=True).as_matrix()
    rmtx = np.matrix(rmtx, dtype=np.float32)
    rvec, _ = cv2.Rodrigues(rmtx)
    rvec = np.matrix(rvec)
    
    check_rvec(rvec)
    return rvec
    
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
    
def invert_transform(rvec, tvec):
    check_rvec(rvec)
    check_tvec(tvec)
    
    rmtx, _ = cv2.Rodrigues(rvec)
    dst_tvec = - rmtx.T * tvec
    dst_rmtx = rmtx.T
    dst_rvec, _ = cv2.Rodrigues(dst_rmtx)
    dst_rvec = np.matrix(dst_rvec)
    
    check_rvec(dst_rvec)
    check_tvec(dst_tvec)
    return dst_rvec, dst_tvec
    
def multiply_transform(left_rvec, left_tvec, right_rvec, right_tvec):
    check_rvec(left_rvec)
    check_tvec(left_tvec)
    check_rvec(right_rvec)
    check_tvec(right_tvec)
    
    left_rmtx = get_rmtx(left_rvec)
    right_rmtx = get_rmtx(right_rvec)
    
    rmtx = left_rmtx * right_rmtx
    rvec = get_rvec(rmtx)
    tvec = left_rmtx * right_tvec + left_tvec
    
    check_rvec(rvec)
    check_tvec(tvec)
    return rvec, tvec
    
def point_transform(rvec, tvec, points):
    check_rvec(rvec)
    check_tvec(tvec)

    points = np.matrix(points).T
    assert points.ndim == 2
    assert points.shape[0] == 3
    
    rmtx = get_rmtx(rvec)
    dst_points = rmtx * points + tvec

    dst_points = np.array(dst_points).T
    
    return dst_points
    
def save_transform(json_path, rvec, tvec):
    check_rvec(rvec)
    check_tvec(tvec)

    transform = { 'rvec_x' : rvec [0,0],
                  'rvec_y' : rvec [1,0],
                  'rvec_z' : rvec [2,0],
                  'tvec_x' : tvec [0,0],
                  'tvec_y' : tvec [1,0],
                  'tvec_z' : tvec [2,0]}
    with open(json_path, 'w') as fp:
        json.dump(transform, fp)
    
