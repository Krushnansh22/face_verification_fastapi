from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from datetime import datetime
from typing import List

# Initialize the FastAPI app
app = FastAPI()

# Allow CORS for testing purposes (adjust as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global list for debug logs
debug_logs = []

def add_log(message: str):
    """Adds a log entry with a timestamp."""
    entry = f"{datetime.now().isoformat()}: {message}"
    debug_logs.append(entry)
    if len(debug_logs) > 50:
        debug_logs.pop(0)  # Limit log size
    print(entry)

def augment_image(img: np.ndarray) -> List[np.ndarray]:
    """
    Apply several transformations to an image to generate additional samples.
    Transformations include:
      1. Horizontal flip
      2. Zoom in by cropping a centered region
      3. Increase brightness
      4. Add Gaussian noise
    
    Returns a list including the original image and its augmented versions.
    """
    augmented_images = [img]  # Include the original image

    # 1. Horizontal Flip
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped)

    # 2. Zoom In (crop center region and resize back to original size)
    h, w = img.shape[:2]
    start_row, start_col = int(0.1 * h), int(0.1 * w)
    end_row, end_col = int(0.9 * h), int(0.9 * w)
    cropped = img[start_row:end_row, start_col:end_col]
    zoomed = cv2.resize(cropped, (w, h))
    augmented_images.append(zoomed)

    # 3. Increase Brightness
    bright = cv2.convertScaleAbs(img, alpha=1.0, beta=50)  # beta adds brightness
    augmented_images.append(bright)

    # 4. Add Gaussian Noise
    noise = np.random.normal(0, 10, img.shape)  # mean=0, std=10
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    augmented_images.append(noisy)

    return augmented_images

def train_model(training_files: List[UploadFile]):
    """
    Train an LBPH face recognizer using cropped face regions extracted from uploaded images 
    and their augmented versions. Faces are detected using Haar cascades, resized to 100x100, 
    and augmented. All images are assigned the label "UserFace" (label 0).
    
    Returns the trained recognizer and the expected label, or None if training fails.
    """
    images = []
    labels = []
    expected_label = 0  # Label for "UserFace"

    add_log(f"Received {len(training_files)} training files.")
    
    # Initialize Haar cascade for face detection
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if detector.empty():
        add_log("Error: Failed to load Haar cascade file.")
        raise HTTPException(status_code=500, detail="Face detection model not found.")
    
    for file in training_files:
        file_bytes = file.file.read()  # Read file bytes
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            add_log(f"Warning: Could not decode training file {file.filename}. Skipping.")
            continue
        
        # Detect faces in the image
        faces = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            add_log(f"No face detected in image {file.filename}. Skipping.")
            continue

        # Process each detected face
        for (x, y, w, h) in faces:
            cropped = img[y:y + h, x:x + w]
            try:
                cropped_resized = cv2.resize(cropped, (100, 100))
            except Exception as e:
                add_log(f"Error resizing face in file {file.filename}: {e}")
                continue

            # Augment the cropped face
            augmented_images = augment_image(cropped_resized)
            for aug_img in augmented_images:
                images.append(aug_img)
                labels.append(expected_label)
                
        file.file.seek(0)  # Reset file pointer if needed

    if not images:
        add_log("No valid face images found after detection and augmentation.")
        return None, None

    images_np = np.array(images, dtype="uint8")
    labels_np = np.array(labels)
    add_log(f"Total augmented faces used for training: {len(images)}")
    
    # Train the recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images_np, labels_np)
    
    add_log("Training completed successfully using cropped faces.")
    return recognizer, expected_label

def predict_image(recognizer, test_file: UploadFile, expected_label=0, threshold=99):
    """
    Predict the label of the test image using the trained recognizer.
    The test image is decoded to grayscale, and faces are detected using Haar cascades.
    Each detected face is cropped, resized to 100x100, and predicted.
    
    Returns True if any face matches the expected label with confidence below the threshold.
    """
    file_bytes = test_file.file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        add_log("Error: Could not decode test image.")
        return False

    # Initialize Haar cascade for face detection
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if detector.empty():
        add_log("Error: Failed to load Haar cascade file.")
        raise HTTPException(status_code=500, detail="Face detection model not found.")

    # Detect faces in the image
    faces = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        add_log("No face detected in test image.")
        return False

    # Process each detected face
    for (x, y, w, h) in faces:
        cropped = img[y:y+h, x:x+w]
        try:
            face_resized = cv2.resize(cropped, (100, 100))
        except Exception as e:
            add_log(f"Error resizing face: {e}")
            continue
        try:
            label, confidence = recognizer.predict(face_resized)
        except Exception as e:
            add_log(f"Prediction error: {e}")
            continue

        add_log(f"Predicted label: {label}, Confidence: {confidence}")
        if (label == expected_label) and (confidence < threshold):
            return True  # Match found

    return False

@app.post("/verify")
async def verify_endpoint(
    training_images: List[UploadFile] = File(...),
    test_image: UploadFile = File(...)
):
    """Endpoint to verify a test image against a set of training images."""
    add_log("Verification request received.")
    if not training_images or test_image is None:
        add_log("Error: Missing training images or test image.")
        raise HTTPException(status_code=400, detail="Missing training images or test image.")

    recognizer, expected_label = train_model(training_images)
    if recognizer is None:
        raise HTTPException(status_code=400, detail="Training failed. No valid images.")
    verified = predict_image(recognizer, test_image, expected_label)
    add_log(f"Verification result: {'Verified' if verified else 'Not Verified'}.")
    return {"verified": verified}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Debug page displaying the latest log entries.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <title>Face Verification Debug</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 20px; }}
          h1 {{ color: #333; }}
          .log-entry {{ margin-bottom: 5px; padding: 5px; border-bottom: 1px solid #ccc; }}
        </style>
      </head>
      <body>
        <h1>Debug Logs</h1>
        {'<br>'.join(f"<div class='log-entry'>{log}</div>" for log in debug_logs)}
        <hr />
        <p>POST your files to <code>/verify</code> to see logs here.</p>
      </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
