ROI tool
========

Description
-----------
The ROI tool will run through a directory of .ppm, .jpg, .jpeg, .png, .bmp images and allow you to mark the regions of interest, saving the regions in the file gt.txt.

Dependencies
------------
sudo apt-get install python-imaging-tk

Instructions
------------
./roi_tool.py -i /path/to/source/directory

The ROI tool will give the first image found in the source directory. Select the regions of interest by dragging the mouse over them and and hit Enter. The first region of interest will be shown in red, write the class and hit Enter. The already classified regions of interest will be blue and the not yet classified remain green.

To exit, hit Escape
