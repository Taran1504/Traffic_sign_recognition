# Traffic_sign_recognition
## Introduction

Traffic Sign Detection is an essential part of computer vision applications in autonomous vehicles. This project uses CNNs to train a model capable of identifying different traffic signs. The key steps involve importing traffic sign datasets, splitting them into training, validation, and testing sets, defining a CNN model, and evaluating its performance.

---

## Project Structure

### Key Components:
1. **Importing Libraries**: Necessary libraries for data manipulation, visualization, and modeling.
2. **Declaring Variables**: Initializing key variables for the project.
3. **Data Processing**:
   - Importing and preparing traffic sign data.
   - Splitting the data into training, validation, and testing sets.
4. **Modeling**:
   - Designing a CNN model with multiple layers.
   - Training the model with the processed dataset.
5. **Evaluation**:
   - Plotting graphs to visualize training progress.
   - Calculating accuracy and score of the model.
6. **Saving the Model**: Exporting the trained model for future use.

---

## Setup and Requirements

### Prerequisites
- Python (>=3.7)
- Required Libraries:
  - TensorFlow/Keras
  - NumPy
  - Matplotlib
  - OpenCV

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/traffic-sign-detection.git
   cd traffic-sign-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Prepare your dataset of traffic sign images (with corresponding XML annotations if using OpenCV).
2. Run the notebook or script to preprocess the data, train the model, and evaluate it.
3. Save the trained model for deployment.

---

## Results

The trained model achieves high accuracy in classifying traffic signs. Graphs illustrating the training and validation loss/accuracy are plotted for analysis.

---

## Acknowledgements

Special thanks to public datasets and libraries used in this project, including TensorFlow and OpenCV.
