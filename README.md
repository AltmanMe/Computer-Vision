# Stereo Calibration using OpenCV-Python

All codes had been tested in Manjaro Linux

## Features
* ### calibrate single camera
* ### calibrate binocular camera
* ### rectify images
* ### compute depth map

## Requirements
* ### cv2
* ### numpy

## Guiding
* ### Before running any above files, you can use '-h' to get help.
```python
python draw_lines.py -h
```
### Then it will give you tips like:
```
optional arguments:
  -h, -- help            show this help message and exit
  -l LEFT_IMG, -- left_img LEFT_IMG path to the left images directory                        
  -r RIGHT_IMG, -- right_img RIGHT_IMG path to the right images directory
```


## Example
* ### calibrate single camera and undistort images
```
python single_calibrate.py -i /path/image_path -s False
False means that do not save the undistoted images.
If True, it will make a directory called `undistoted_images` in current dir.
```
* ### calibrate binouclar camera system
```
python camera_calibrate.py -h
Assumption here is that given path contains two directories of chessboard images named left and right.
```
* ### rectify images
```
python rectify.py -h
```
* ### visulize depth 
```
python compute_depth.py
Assumption here is that givn path contains two directories of left and right rectified images.
```
* ### find corresponding feature points and draw epipolar lines
```
python epipolar_lines.py -h
```





















