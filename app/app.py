import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model import Model
model = Model()

params = np.load('../model/parameters.npz')
model.dense1.weights = params['w1']
model.dense1.biases = params['b1']
model.dense2.weights = params['w2']
model.dense2.biases = params['b2']
model.dense3.weights = params['w3']
model.dense3.biases = params['b3']


import tkinter as tk
from tkinter import Canvas, Frame, Label, Button, ttk
from PIL import Image, ImageDraw

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Digit Classifier")

        # Left: Drawing canvas
        self.canvas_size = 280
        self.canvas = Canvas(root, height=self.canvas_size, width=self.canvas_size, bg='white')
        self.canvas.grid(row=0, column=0, padx=5, pady=5, sticky='n')

        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 255)

        self.canvas.bind('<B1-Motion>', self.draw_on_canvas)

        # Right: Controls + progress bars
        self.right_frame = Frame(root)
        self.right_frame.grid(row=0, column=1, padx=(0, 5), pady=(5, 0), sticky='n')

        # Clear button on top
        self.clear_btn = Button(self.right_frame, text='Clear', command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 2))

        # Progress bars with labels
        self.progress_bars = []

        for i in range(10):
            label = Label(self.right_frame, text=f'{i}:', anchor='e')
            label.grid(row=i+1, column=0, sticky='e', pady=1)

            progress = ttk.Progressbar(self.right_frame, orient='horizontal', length=100, mode='determinate', maximum=100)
            progress.grid(row=i+1, column=1, sticky='w', pady=2)

            self.progress_bars.append(progress)

        # Make column 1 expand (progress bars)
        self.right_frame.columnconfigure(1, weight=1)

        self.update_progress_bars()

    def draw_on_canvas(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        
        draw = ImageDraw.Draw(self.image)
        draw.ellipse((x - r, y - r, x + r, y + r), fill='black')

        self.predict_digit()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 255)
    
    def preprocess_image(self):
        img = self.image.convert('L')  # Convert to grayscale (already in grayscale mode)
        img = img.resize((28, 28))  # Resize the image to 28x28 (standard for MNIST)
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = img_array.reshape(1, 784)
        img_array = 1 - img_array
        return img_array

    def predict_digit(self):
        # Preprocess the image from the canvas
        X_input = self.preprocess_image()
        output = model.forward(X_input, y=None, type='test')
        # digit = np.argmax(output)
        self.update_progress_bars(output)

    def update_progress_bars(self, output=None):
        if output is None:
            output = np.zeros(10)
        else:
            output = output[0]

        for i in range(10):
            self.progress_bars[i]["value"] = output[i] * 100


root = tk.Tk()
app = App(root)
# root.after(2000, root.destroy)
root.mainloop()