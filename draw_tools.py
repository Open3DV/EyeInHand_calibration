import numpy as np
import cv2
from spatial_transform import get_rvec_Yaskawa, invert_transform, get_rmtx, get_rvec, check_rvec, check_tvec, point_transform

def create_board_points():
    points = []
    for i in range(11):
        if i%2==0:
            x0 = 15.0
        else:
            x0 = 25.0
        y = 15.0+i*10.0
        for j in range(7):
            points.append([x0+j*20.0, y, 0.0])
    return points
    
def draw_board(ax, rvec, tvec, color, nx=10, ny=7, grid_width=20):
    check_rvec(rvec)
    check_tvec(tvec)
    
    board_points = create_board_points()
    board_points = point_transform(rvec, tvec, board_points)
    ax.scatter3D(board_points[:,0], board_points[:,1], board_points[:,2], color=color, alpha=0.1)    
        
        
def draw_camera(ax, camera_mtx=np.array([[1500, 0, 540],
                                         [0, 1500, 360],
                                         [0,    0,   1]]), 
                    h_size=1920, 
                    rvec=np.array([0.0,0.0,0.0]), 
                    tvec=np.matrix([0.0,0.0,0.0]).T, 
                    color='b'):
    
    check_rvec(rvec)
    check_tvec(tvec)
    
    fx = camera_mtx[0,0] / h_size * 100
    fy = camera_mtx[1,1] / h_size * 100
    cx = camera_mtx[0,2] / h_size * 100
    cy = camera_mtx[1,2] / h_size * 100
    
    rmtx = get_rmtx(rvec)
    
    p0 = np.matrix([0,0,0], dtype=np.float32).T
    p1 = np.matrix([cx,cy,fx], dtype=np.float32).T
    p2 = np.matrix([-cx,cy,fx], dtype=np.float32).T
    p3 = np.matrix([-cx,-cy,fx], dtype=np.float32).T
    p4 = np.matrix([cx,-cy,fx], dtype=np.float32).T
    
    p0 = rmtx * p0 + tvec
    p1 = rmtx * p1 + tvec
    p2 = rmtx * p2 + tvec
    p3 = rmtx * p3 + tvec
    p4 = rmtx * p4 + tvec
    
    ax.plot([p0[0,0],p1[0,0]], [p0[1,0],p1[1,0]], [p0[2,0],p1[2,0]], color)
    ax.plot([p0[0,0],p2[0,0]], [p0[1,0],p2[1,0]], [p0[2,0],p2[2,0]], color)
    ax.plot([p0[0,0],p3[0,0]], [p0[1,0],p3[1,0]], [p0[2,0],p3[2,0]], color)
    ax.plot([p0[0,0],p4[0,0]], [p0[1,0],p4[1,0]], [p0[2,0],p4[2,0]], color)
    
    ax.plot([p1[0,0],p2[0,0],p3[0,0],p4[0,0],p1[0,0]], [p1[1,0],p2[1,0],p3[1,0],p4[1,0],p1[1,0]], [p1[2,0],p2[2,0],p3[2,0],p4[2,0],p1[2,0]], color)
    
def draw_camera_from_inverted_transform(ax, camera_mtx=np.array([[1500, 0, 540],
                                         [0, 1500, 360],
                                         [0,    0,   1]]), 
                    h_size=1920, 
                    inverted_rvec=np.array([0.0,0.0,0.0]), 
                    inverted_tvec=np.matrix([0.0,0.0,0.0]).T, 
                    color='b'):
    
    rvec, tvec = invert_transform(inverted_rvec, inverted_tvec)
    
    draw_camera(ax, camera_mtx, 
                    h_size, 
                    rvec, 
                    tvec, 
                    color)

    
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
   
def draw_axis(ax, tvec, rvec, index=-1):
    check_rvec(rvec)
    check_tvec(tvec)
    rmtx = get_rmtx(rvec)
    # 旋转矩阵每一列，就是一个维度的单位向量
    ax.plot3D([tvec[0,0],tvec[0,0]+rmtx[0,0]*100], [tvec[1,0],tvec[1,0]+rmtx[1,0]*100], [tvec[2,0],tvec[2,0]+rmtx[2,0]*100], 'r')
    ax.plot3D([tvec[0,0],tvec[0,0]+rmtx[0,1]*100], [tvec[1,0],tvec[1,0]+rmtx[1,1]*100], [tvec[2,0],tvec[2,0]+rmtx[2,1]*100], 'g')
    ax.plot3D([tvec[0,0],tvec[0,0]+rmtx[0,2]*100], [tvec[1,0],tvec[1,0]+rmtx[1,2]*100], [tvec[2,0],tvec[2,0]+rmtx[2,2]*100], 'b')

    if index >= 0:
        ax.text(tvec[0,0], tvec[1,0], tvec[2,0], index, color='black')
