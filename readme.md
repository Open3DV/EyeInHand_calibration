操作步骤：

1、导入相机参数param.txt到input文件夹中。
2、采集8张不同位姿的bmp以及tiff到input文件夹中。
3、导入8张不同位姿下的机械臂pos（顺序为Rx、Ry、Rz、X、Y、Z）。
4、默认使用圆心距20mm标定板，否则自行修改代码168行。
5、运行calib.py。
6、结果检查：将output文件夹生成的.xyz文件导入点云查看软件，检验圆心重合度。或利用result.txt结果走点校验。



