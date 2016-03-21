#!/usr/bin/env python
import argparse

from gui import ROIApp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--indir", required=True,
                    help="Path to the source directory")
    args = vars(ap.parse_args())

    win = ROIApp(args["indir"])

    try:
        win.root.mainloop()
    except KeyboardInterrupt:
        win.quit()
        raise
