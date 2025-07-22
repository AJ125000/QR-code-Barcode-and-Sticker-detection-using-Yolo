import os
import uuid
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
import logging
import cv2
import numpy as np

# --- Setup ---
app = Flask(__name__, static_folder='static', template_folder='templates')
logging.basicConfig(level=logging.INFO)

# --- Constants ---
UPLOADS_DIR = 'uploads'
PREDICTIONS_DIR = os.path.join('static', 'predictions')
# Threshold from the Streamlit app, can be adjusted here.
LAPLACIAN_THRESHOLD = 150.0 

# --- Create Directories ---
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# --- Load Models ---
try:
    logging.info("Loading YOLOv8 models...")
    qr_model = YOLO('models\\best_qr_YOLOv8n.pt')
    barcode_model = YOLO('models\\best_barcode_YOLOv8s.pt')
    logging.info("Models loaded successfully!")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    qr_model = None
    barcode_model = None

# --- Core Sticker Detection Logic (from Streamlit) ---
def calculate_laplacian_variance(image_array):
    """
    Calculates the Laplacian variance for the central region of an image.
    This score represents the amount of detail/edges.

    Args:
        image_array (np.ndarray): The image in BGR format.

    Returns:
        float: The calculated Laplacian variance.
    """
    # Define the Region of Interest (ROI) as the central 70% width and 40% height.
    h, w, _ = image_array.shape
    roi_w_start = int(w * 0.15)
    roi_w_end = int(w * 0.85)
    roi_h_start = int(h * 0.30)
    roi_h_end = int(h * 0.70)
    
    roi = image_array[roi_h_start:roi_h_end, roi_w_start:roi_w_end]

    # Convert the ROI to grayscale for edge detection.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply the Laplacian operator to get the edges.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Calculate the variance of the Laplacian.
    variance = laplacian.var()
    
    return variance

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and qr_model and barcode_model:
        unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOADS_DIR, unique_filename)
        file.save(filepath)
        logging.info(f"File saved to {filepath}")

        try:
            # Run Inference
            qr_results = qr_model.predict(filepath, verbose=False)
            qr_present = len(qr_results[0].boxes) > 0
            qr_status = "QR code present" if qr_present else "QR code absent"

            barcode_results = barcode_model.predict(filepath, verbose=False)
            barcode_present = len(barcode_results[0].boxes) > 0
            barcode_status = "Barcode present" if barcode_present else "Barcode absent"

            # Save the Visual Result
            result_plot = qr_results[0].plot()  # Returns a NumPy array (BGR)
            result_plot_rgb = Image.fromarray(result_plot[..., ::-1])
            output_image_path = os.path.join(PREDICTIONS_DIR, unique_filename)
            result_plot_rgb.save(output_image_path)
            
            image_url = f'/static/predictions/{unique_filename}'

            # Clean up and Respond
            os.remove(filepath)

            return jsonify({
                'qr_result': qr_status,
                'barcode_result': barcode_status,
                'image_url': image_url
            })

        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Failed to process the image.'}), 500

    return jsonify({'error': 'Server error: Models not loaded'}), 500

# --- NEW: Sticker Detection Endpoint ---
@app.route('/detect_sticker', methods=['POST'])
def detect_sticker():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filepath = None  # Initialize for the finally block
    try:
        # Save file temporarily
        filepath = os.path.join(UPLOADS_DIR, file.filename)
        file.save(filepath)

        # Open image with PIL and convert to NumPy array
        image = Image.open(filepath)
        image_array_pil = np.array(image)

        # Ensure the image is in BGR format for OpenCV
        # This handles both RGB and RGBA images from PIL
        if len(image_array_pil.shape) == 2: # Grayscale image
            image_array_bgr = cv2.cvtColor(image_array_pil, cv2.COLOR_GRAY2BGR)
        elif image_array_pil.shape[2] == 4: # RGBA
            image_array_bgr = cv2.cvtColor(image_array_pil, cv2.COLOR_RGBA2BGR)
        else: # RGB
            image_array_bgr = cv2.cvtColor(image_array_pil, cv2.COLOR_RGB2BGR)

        # Perform the classification
        variance = calculate_laplacian_variance(image_array_bgr)
        
        # Apply the classification logic
        if variance > LAPLACIAN_THRESHOLD:
            status = "Sticker Detected"
        else:
            status = "No Sticker Detected"

        # Return the result
        return jsonify({
            'sticker_status': status,
            'detail_score': f"{variance:.2f}"
        })

    except Exception as e:
        logging.error(f"An error occurred during sticker detection: {e}")
        return jsonify({'error': 'Failed to process the image for sticker detection.'}), 500
    
    finally:
        # Clean up the uploaded file
        if filepath and os.path.exists(filepath):
            os.remove(filepath)


@app.route('/delete_prediction', methods=['POST'])
def delete_prediction():
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    file_path = os.path.join(PREDICTIONS_DIR, filename)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'status': 'deleted'})
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logging.error(f"Error deleting file {file_path}: {e}")
        return jsonify({'error': 'Failed to delete file'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)