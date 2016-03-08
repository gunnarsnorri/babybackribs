#!/usr/bin/env python
import tkFileDialog
import gtk.gdk


if __name__ == "__main__":
    w = gtk.gdk.get_default_root_window()
    sz = w.get_size()
    pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, sz[0], sz[1])
    pb = pb.get_from_drawable(w, w.get_colormap(), 0, 0, 0, 0, sz[0], sz[1])

    filename = tkFileDialog.asksaveasfilename(initialdir="~/Pictures")
    f_ex = filename.split(".")
    if len(f_ex) <= 1:
        extension = "png"  # Default extension
        filename = "%s.%s" % (filename, extension)
    else:
        extension = f_ex[-1]

    if (pb is not None):
        x = 1650
        y = 200
        width = 1800
        height = 800
        cropped = pb.subpixbuf(x, y, width, height)
        cropped.save(filename, extension)
