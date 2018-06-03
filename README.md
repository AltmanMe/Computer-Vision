Computer Vision Basics
========
#Stereo Calibration using OpenCV-Python

## Features
* ### calibrate single camera
* ### calibrate binocular camera
* ### rectify images
* ### compute depth map

## Requirements
* ### cv2
* ### numpy

## Guiding
* ### Assumption here is that given path contains two directories of chessboard images named `left` and `right`.
* ### Before running any above files, you can use '-h' to get help.
```python
python draw_lines.py -h
```
### Then it will give you tips like:
```
optional arguments:
  -h, -- help            show this help message and exit
  -l LEFT_IMG, -- left_img LEFT_IMG path to the left                        
  -r RIGHT_IMG, -- right_img RIGHT_IMG path to the right
```


## Example
* ### calibrate camera and undistort images
```
python single_calibrate.py -i /path/image_path -s False
False means that do not save the undistoted images.
if you want to save them, you can ignore this parameter.
```





















