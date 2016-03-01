#!/usr/bin/env python
from gui import ROIApp


image_fullname = "/home/gunsno/Wallpapers/ubuntu.jpg"

if __name__ == "__main__":

    win = ROIApp()
    win.set_image(image_fullname)

    win.root.mainloop()
