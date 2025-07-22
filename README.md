# Multi-Analysis Image Inspector

A full-stack web application designed for comprehensive image analysis. This tool allows users to upload images and receive instant feedback on three key metrics: the presence of QR codes, the presence of barcodes, and whether the image contains a printed sticker.

The application features a modern, drag-and-drop interface that processes multiple files concurrently and displays the results—including a preview image with detected object bounding boxes—in real-time.

*(Replace this placeholder with an actual screenshot of your application)*

## Features

- **Dual YOLOv8 Models**: Utilizes two separate YOLOv8 models for high-accuracy QR code and barcode detection.
- **Sticker vs. No-Sticker Classification**: Employs an OpenCV-based algorithm to classify whether an image contains a sticker by analyzing edge detail variance.
- **Modern Frontend**: A clean, responsive, and user-friendly interface built with Tailwind CSS.
- **Drag & Drop Upload**: Supports easy file uploads by dragging them directly into the browser window.
- **Concurrent Processing**: The frontend makes simultaneous API calls for each analysis type, providing faster results.
- **Real-time Results**: Uploaded files are processed immediately, and results are displayed dynamically for each file.
- **Visual Feedback**: For QR/barcode detection, the application returns and displays the original image with bounding boxes drawn around detected objects.
- **Automatic Server Cleanup**: Processed images are automatically deleted from the server after the results are sent to the client to conserve storage space.

## How It Works

The project is composed of a Flask backend that handles the heavy lifting of model inference and a pure HTML, CSS, and JavaScript frontend for the user interface.

### Backend (Flask)

The backend is built with Flask and exposes three main API endpoints:

**`/predict` (POST):**
- Accepts an image file.
- Loads two specialized YOLOv8 models (`best_qr_YOLOv8n.pt` and `best_barcode_YOLOv8s.pt`).
- Runs inference on the image to detect QR codes and barcodes.
- Uses the `.plot()` method from the ultralytics library to draw bounding boxes on the result.
- Saves this plotted image temporarily and returns its URL along with the detection status.

**`/detect_sticker` (POST):**
- Accepts an image file.
- Uses OpenCV to analyze the image's texture and detail.
- It calculates the variance of the Laplacian on a central region of interest. A high variance indicates numerous sharp edges (like text and graphics on a sticker), while a low variance suggests a smooth, uniform surface.
- Compares the variance score against a configurable `LAPLACIAN_THRESHOLD` to classify the image as "Sticker Detected" or "No Sticker Detected".

**`/delete_prediction` (POST):**
- Accepts a filename.
- This endpoint is called by the frontend after a result image has been successfully loaded in the user's browser, allowing the server to safely delete the temporary file.

### Frontend (HTML, JS, Tailwind CSS)

The frontend is a single `index.html` file that provides a rich user experience without requiring a complex framework.

- When a user uploads one or more files, the JavaScript logic initiates two fetch requests concurrently for each file using `Promise.all()`.
- One request hits the `/predict` endpoint, and the other hits `/detect_sticker`.
- Once both responses are received, the JavaScript combines the results into a single data object.
- A dynamic result card is created for the file, displaying the status of all three analyses (QR, Barcode, Sticker) and the resulting image with bounding boxes.

## Project Structure

```
.
├── models/
│   ├── best_barcode_YOLOv8s.pt   # Barcode detection model
│   └── best_qr_YOLOv8n.pt        # QR code detection model
├── static/
│   └── predictions/              # Temporarily stores result images
├── templates/
│   └── index.html                # The main frontend file
├── uploads/                      # Temporarily stores uploaded files
├── app.py                        # The Flask backend server
└── requirements.txt              # Python dependencies
```

## Setup and Installation

Follow these steps to run the project locally.

### Prerequisites

- Python 3.9 or newer
- pip (Python package installer)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/multi-analysis-inspector.git
cd multi-analysis-inspector
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required Python packages using the requirements.txt file.

```bash
pip install -r requirements.txt
```

### 4. Download YOLOv8 Models

You need to have the YOLOv8 model files (.pt) for QR code and barcode detection.

1. Create a directory named `models` in the project root.
2. Place your trained model files inside this directory:
   - `models/best_qr_YOLOv8n.pt`
   - `models/best_barcode_YOLOv8s.pt`

**Note:** The paths are hardcoded in `app.py`. Ensure the filenames match exactly.

### 5. Run the Application

Once the setup is complete, start the Flask server.

```bash
python app.py
```

The server will start, typically on `http://127.0.0.1:5000`.

## Usage

1. Open your web browser and navigate to `http://127.0.0.1:5000`.
2. Drag and drop one or more image files (.jpg, .png) onto the upload area, or click the "Browse Files" button to select them from your computer.
3. The application will automatically process each file and display a result card in the right-hand panel.
4. Each card will show:
   - A preview of the image with detected QR codes/barcodes boxed.
   - The detection status for QR codes and barcodes.
   - The classification result for sticker presence.
   - The calculated "Detail Score" from the sticker detection algorithm.

## Configuration

### Sticker Detection Threshold

The sensitivity of the sticker detection can be adjusted by changing the `LAPLACIAN_THRESHOLD` constant at the top of the `app.py` file.

```python
# app.py

# ...
# Threshold from the Streamlit app, can be adjusted here.
LAPLACIAN_THRESHOLD = 150.0 
# ...
```

- Increase the value to make the detection stricter (requiring more detail to be classified as a sticker).
- Decrease the value to make it more sensitive (classifying images with less detail as stickers).

## Dependencies

- **Flask**: Web server framework.
- **ultralytics**: For YOLOv8 model inference.
- **Pillow**: Image manipulation library.
- **opencv-python**: For the sticker detection image processing logic.
- **numpy**: Numerical operations, required by OpenCV.