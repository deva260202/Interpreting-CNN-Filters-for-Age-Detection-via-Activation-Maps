import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
from numpy import expand_dims
import math
from tensorflow.keras.applications.vgg16 import preprocess_input

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(48, 48)):
    image = load_img(image_path, target_size=target_size)  # Resize image to target size
    image = img_to_array(image)
    image = expand_dims(image, axis=0)
    image = preprocess_input(image)  # Preprocess the image
    return image

# Function to visualize feature maps
def visualize_feature_maps(model, image_path, layer_indices):
    image = load_and_preprocess_image(image_path)
    outputs = [model.layers[i].output for i in layer_indices]
    model_partial = Model(inputs=model.inputs, outputs=outputs)
    feature_maps = model_partial.predict(image)
    
    for block_index, fmap in zip(layer_indices, feature_maps):
        num_features = fmap.shape[3]
        grid_size = math.ceil(math.sqrt(num_features))
        
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(f"BLOCK_{block_index}", fontsize=20)
        for i in range(1, num_features + 1):
            plt.subplot(grid_size, grid_size, i)
            plt.imshow(fmap[0, :, :, i - 1], cmap='gray')
            plt.axis('off')
        
        plt.show()

# Load the custom pre-trained model
model = load_model('C:/Users/admin/Desktop/Python Prog/Age gender detector/Age_Sex Detection.keras')

# Function to handle the image file selection and feature map visualization
def on_select_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        visualize_feature_maps(model, image_path, layer_indices=[2, 5, 9, 13])  # Example layer indices

# Initialize the GUI
root = tk.Tk()
root.title("Feature Map Visualizer")
root.geometry("1000x700") 

heading = tk.Label(root, text="Age and Gender Feature Map Visualizer", pady=20, font=("Arial", 24, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

# Create and place a button to select the image file
select_button = tk.Button(root, text="Select Image", command=on_select_image)
select_button.pack(pady=20)

# Run the GUI loop
root.mainloop()
