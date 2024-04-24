
Operation steps:

1. Import camera parameters from param.txt into the input folder.
2. Capture 8 BMP and TIFF images with different poses into the input folder.
Import the positions of the robot arm under 8 different poses (order: Rx, Ry, Rz, X, Y, Z).
3. By default, use a calibration board with a 20mm center distance; otherwise, modify line 168 of the code accordingly.
4. Run calib.py. 
5. Check the "output/result.txt" (order: rotation_matrix[0:9], traslation_vector[0:3])
5. Result verification: open .xyz files generated in the "output/" folder into CloudCompare to check the overlap of the centers. 

