from Tkinter import Tk, Entry, Label, Frame
from PIL import Image, ImageTk, ImageDraw


class ROIApp(object):

    def __init__(self):
        self.root = Tk()
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.wm_title("ROI Tool")

        # Frame for the input field and buttons
        self.input_frame = Frame(self.root)
        self.input_frame.pack(side="top")

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
        self.cropping = False

    def set_image(self, path):
        self.image = Image.open(path)
        self.imagetk = ImageTk.PhotoImage(image=self.image)
        self.draw = ImageDraw.Draw(self.image)
        self.show_frame()

    def show_frame(self):
        self.imagetk = ImageTk.PhotoImage(image=self.image)
        self.image_label.configure(image=self.imagetk)
        self.image_label.after(10, self.show_frame)

    def on_button_press(self, event):
        self.cropping = True
        self.ref_start = (event.x, event.y)

    def on_button_release(self, event):
        self.cropping = False
        ref_end = (event.x, event.y)
        self.ref_pts.append((self.ref_start, ref_end))
        print("releasing")

        # Draw rectangle around ROI
        self.draw.rectangle(
            self.ref_pts[-1],
            outline="green")
