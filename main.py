from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from datetime import datetime
from typing import List
import tempfile
import traceback
import os

app = FastAPI()

# Allow CORS for testing (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug logs
debug_logs = []

def add_log(message: str):
    """Add a log entry with a timestamp."""
    entry = f"{datetime.now().isoformat()}: {message}"
    debug_logs.append(entry)
    if len(debug_logs) > 50:
        debug_logs.pop(0)
    print(entry)

def train_model(training_files: List[UploadFile]) -> tuple:
    """Train an LBPH face recognizer with the original training images only."""
    try:
        # Load Haar Cascade from environment variable
        xml_content = os.getenv("HARIHAR")
        if not xml_content:
            add_log("Error: Haar Cascade XML content not found in environment variable 'HARIHAR'.")
            raise ValueError("Haar Cascade XML content not found.")

        with tempfile.NamedTemporaryFile(delete=True, suffix='.xml') as temp_file:
            temp_file.write(xml_content.encode('utf-8'))
            temp_file.flush()
            detector = cv2.CascadeClassifier(temp_file.name)
            if detector.empty():
                add_log("Error: Failed to load Haar Cascade classifier from environment variable.")
                raise ValueError("Invalid Haar Cascade XML content.")

        images = []
        labels = []
        expected_label = 0  # All images are "UserFace"
        add_log(f"Received {len(training_files)} training files.")
        for file in training_files:
            file_bytes = file.file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                add_log(f"Warning: Could not decode training file {file.filename}. Skipping.")
                continue
            faces = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                add_log(f"No face detected in training image {file.filename}. Skipping.")
                continue
            # Use the first detected face
            for (x, y, w, h) in faces:
                cropped = img[y:y+h, x:x+w]
                break  # Take the first face
            img_resized = cv2.resize(cropped, (100, 100))
            images.append(img_resized)
            labels.append(expected_label)
            file.file.seek(0)
        if not images:
            add_log("No valid training images with detectable faces found.")
            return None, None
        images_np = np.array(images, dtype="uint8")
        labels_np = np.array(labels)
        add_log(f"Training with {len(images)} original images.")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(images_np, labels_np)
        add_log("Training completed successfully.")
        return recognizer, expected_label
    except Exception as e:
        add_log(f"Error in train_model: {str(e)}")
        raise

def predict_image(recognizer, test_file: UploadFile, expected_label=0, threshold=100) -> bool:
    """Predict the label of the test image using the trained recognizer."""
    try:
        # Load Haar Cascade from environment variable
        xml_content = os.getenv("HARIHAR")
        if not xml_content:
            add_log("Error: Haar Cascade XML content not found in environment variable 'HARIHAR'.")
            raise ValueError("Haar Cascade XML content not found.")

        with tempfile.NamedTemporaryFile(delete=True, suffix='.xml') as temp_file:
            temp_file.write(xml_content.encode('utf-8'))
            temp_file.flush()
            detector = cv2.CascadeClassifier(temp_file.name)
            if detector.empty():
                add_log("Error: Failed to load Haar Cascade classifier from environment variable.")
                raise ValueError("Invalid Haar Cascade XML content.")

        # Read and decode the test image
        file_bytes = test_file.file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            add_log("Error: Could not decode test image.")
            return False

        # Detect faces
        faces = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            add_log("No face detected in test image.")
            return False

        # Use the first detected face
        for (x, y, w, h) in faces:
            cropped = img[y:y+h, x:x+w]
            break  # Take the first face
        img_resized = cv2.resize(cropped, (100, 100))

        label, confidence = recognizer.predict(img_resized)
        add_log(f"Predicted label: {label}, Confidence: {confidence}")
        return (label == expected_label) and (confidence < threshold)
    except Exception as e:
        add_log(f"Error in predict_image: {str(e)}")
        raise

@app.post("/verify")
async def verify_endpoint(
    training_images: List[UploadFile] = File(...),
    test_image: UploadFile = File(...)
):
    """Verify the test image against the training images."""
    add_log("Verification request received.")
    try:
        if not training_images or test_image is None:
            add_log("Error: Missing training images or test image.")
            raise HTTPException(status_code=400, detail="Missing required files.")

        recognizer, expected_label = train_model(training_images)
        if recognizer is None:
            add_log("Training failed due to no valid images.")
            raise HTTPException(status_code=400, detail="Training failed. No valid images.")

        verified = predict_image(recognizer, test_image, expected_label)
        add_log(f"Verification result: {'Verified' if verified else 'Not Verified'}.")
        return {"verified": verified}
    except Exception as e:
        add_log(f"Server error: {str(e)}\nStack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/logs")
async def get_logs():
    """Return debug logs for troubleshooting."""
    return {"logs": debug_logs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
