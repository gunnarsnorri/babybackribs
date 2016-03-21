import os
import sys
import csv
from Tkinter import Tk, Entry, Label, Frame
from PIL import Image, ImageTk, ImageDraw


class ROIApp(object):

    def __init__(self, source_dir):
        self.state = "Selecting"

        self.image_types = ["ppm", "jpg", "jpeg", "png", "bmp"]  # Add more
        self.source_dir = source_dir
        self.gt_filename = os.path.join(self.source_dir, "gt.txt")
        images = self.get_images()
        self.image_fullpaths = (
            os.path.join(self.source_dir, image_filename)
            for image_filename in images)
        self.current_image_fullpath = None
        self.gt_file = open(self.gt_filename, 'a')
        self.gt_writer = csv.writer(self.gt_file, delimiter=";")

        self.root = Tk()
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.bind("<KP_Enter>", self.on_kp_enter)
        self.root.bind("<Return>", self.on_kp_enter)
        self.root.wm_title("ROI Tool")

        # Frame for the input field and buttons
        self.input_frame = Frame(self.root)
        self.input_frame.pack(side="top")

        self.label = Label(self.input_frame, text="Class:")
        self.label.pack(side="left")
        self.entry_field = Entry(self.input_frame)
        self.entry_field.pack(side="left")

        self.image_label = Label(self.root)
        self.image_label.pack(side="bottom")
        self.image_label.bind("<ButtonPress-1>", self.on_button_press)
        self.image_label.bind("<ButtonRelease-1>", self.on_button_release)

        self.image = None
        self.imagetk = None
        self.draw = None

        self.ref_pts = []
        self.ref_start = None
        self.ref_pts_iter = None
        self.current_ref_pt = None
        self.gt_rows = []

        try:
            self.next_image()
        except StopIteration:
            print("No images were found (extensions %s) "
                  "or all the images found were already parsed, check file "
                  "%s" % (", ".join(self.image_types), self.gt_filename))
            self.quit()
            sys.exit(1)
        self.show_frame()

    def quit(self):
        self.gt_file.close()
        self.root.quit()

    def ask_for_next_category(self):
        """Ask for next category if another ref point exists,
        else move onto next image"""
        try:
            self.current_ref_pt = self.ref_pts_iter.next()
            self.draw.rectangle(self.current_ref_pt, outline="red")
            self.entry_field.select_range(0, "end")
            self.entry_field.focus()
        except StopIteration:
            self.state = "Selecting"
            try:
                self.next_image()
            except StopIteration:
                print(
                    "Done, regions of interest written in %s" %
                    self.gt_filename)
                self.quit()

    def on_kp_enter(self, event):
        if self.state == "Categorizing":
            category = self.entry_field.get()
            image_path = self.current_image_fullpath.split("/")[-1]
            data = ((image_path,) + self.current_ref_pt[0] +
                    self.current_ref_pt[1] + (category,))
            self.gt_rows.append(data)
            self.draw.rectangle(self.current_ref_pt, outline="blue")
            self.ask_for_next_category()
        else:
            self.state = "Categorizing"
            self.ref_pts_iter = iter(self.ref_pts)
            self.ref_pts = []
            self.ask_for_next_category()

    def get_images(self):
        try:
            gt_file = open(self.gt_filename, 'r')
            reader = csv.reader(gt_file, delimiter=";")
            already_parsed = [row[0] for row in reader]
        except IOError:
            already_parsed = []
        return [filename for filename in os.listdir(self.source_dir) if
                (filename.split(".")[-1] in self.image_types and
                 filename not in already_parsed)]

    def next_image(self):
        self.gt_writer.writerows(self.gt_rows)
        self.gt_rows = []
        self.current_image_fullpath = self.image_fullpaths.next()
        self.image = Image.open(self.current_image_fullpath)
        self.draw = ImageDraw.Draw(self.image)

    def show_frame(self):
        self.imagetk = ImageTk.PhotoImage(image=self.image)
        self.image_label.configure(image=self.imagetk)
        self.image_label.after(10, self.show_frame)

    def on_button_press(self, event):
        if self.state == "Selecting":
            self.ref_start = (event.x, event.y)

    def on_button_release(self, event):
        if self.state == "Selecting":
            # Make sure ROI doesn't exceed pixture coordinates and that
            # the corners go from top left to bottom right
            ref_end = (event.x, event.y)
            ref_pt = self.top_left_to_bottom_right(ref_end)
            self.ref_pts.append(ref_pt)

            # Draw rectangle around ROI
            self.draw.rectangle(
                self.ref_pts[-1],
                outline="green")

    def top_left_to_bottom_right(self, ref_end):
        """Returns the tuple:
            (top_left, bottom_right) where top_left and bottom_right
            are coordinate tuples"""
        x1 = max(0, min(self.ref_start[0], ref_end[0]))
        x2 = min(max(self.ref_start[0], ref_end[0]), self.image.size[0])
        y1 = max(0, min(self.ref_start[1], ref_end[1]))
        y2 = min(max(self.ref_start[1], ref_end[1]), self.image.size[1])
        return ((x1, y1), (x2, y2))
