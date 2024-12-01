import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Define class names
class_names = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

def preprocess_image(img_path):
    """Preprocess the image for model prediction."""
    img = image.load_img(img_path, target_size=(32, 32), color_mode='grayscale')  # Load as grayscale
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

def classify_image():
    """Classify the uploaded image and display the result."""
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Load and preprocess the image
            img_array = preprocess_image(file_path)
            
            # Make a prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = class_names.get(predicted_class, "Unknown Sign")
            confidence = np.max(prediction)  # Get the confidence score

            # Display the image
            img = Image.open(file_path)
            img.thumbnail((200, 200), Image.LANCZOS)  # Maintain aspect ratio using LANCZOS filter
            img = ImageTk.PhotoImage(img)
            
            img_label.configure(image=img)
            img_label.image = img
            result_label.config(text=f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}")

        except Exception as e:
            result_label.config(text=f"Error processing image: {str(e)}")

# Set up the GUI
root = tk.Tk()
root.title("Traffic Sign Classifier")
root.geometry("400x400")

# Label to show the image
img_label = Label(root)
img_label.pack()

# Label to display classification result
result_label = Label(root, text="Upload an image of a traffic sign", font=("Helvetica", 16))
result_label.pack(pady=20)

# Button to upload and classify image
upload_button = tk.Button(root, text="Upload Image", command=classify_image)
upload_button.pack()

root.mainloop()
