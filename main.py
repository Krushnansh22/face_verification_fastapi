from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
from datetime import datetime
from typing import List

app = FastAPI()

# Allow CORS for production (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider restricting origins in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global list for debug logs.
debug_logs = []


def add_log(message: str):
    """Adds a log entry with a timestamp."""
    entry = f"{datetime.now().isoformat()}: {message}"
    debug_logs.append(entry)
    if len(debug_logs) > 50:
        debug_logs.pop(0)
    print(entry)


# Load Haarcascade for face detection.
haarcascade_path = os.path.join("data", "haarcascade_frontalface_default.xml")
if not os.path.exists(haarcascade_path):
    raise FileNotFoundError(f"Haarcascade file not found at {haarcascade_path}")
face_cascade = cv2.CascadeClassifier(haarcascade_path)


def augment_image(img: np.ndarray) -> List[np.ndarray]:
    """
    Applies several transformations to an image to generate additional samples.
    Transformations include:
      1. Horizontal flip.
      2. Zoom in by cropping a centered region.
      3. Increase brightness.
      4. Add Gaussian noise.
    Returns a list containing the original image plus augmented versions.
    """
    augmented_images = [img]  # Include the original image

    # 1. Horizontal Flip
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped)

    # 2. Zoom In (crop center region then resize back to original size)
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
    Trains an LBPH face recognizer using only the face regions cropped from training images.
    For each training file, a Haarcascade detects a face. The face region is cropped, resized to 100x100,
    and then augmented to improve variation. The recognizer is trained on these augmented face images.
    Returns the trained recognizer and the expected label (always 0).
    """
    images = []
    labels = []
    expected_label = 0  # All images belong to one class ("UserFace")

    add_log(f"Received {len(training_files)} training files.")
    for file in training_files:
        file_bytes = file.file.read()  # Read file bytes
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            add_log(f"Warning: Could not decode training file {file.filename}. Skipping.")
            continue

        # Detect faces using Haarcascade.
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            add_log(f"No face detected in file {file.filename}. Skipping.")
            continue

        # Use the first detected face (you could iterate over multiple faces if desired).
        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]  # Crop the face region.
        face_resized = cv2.resize(face_img, (100, 100))  # Resize to standard size.

        # Augment the cropped face.
        augmented_images = augment_image(face_resized)
        for aug_img in augmented_images:
            images.append(aug_img)
            labels.append(expected_label)

        file.file.seek(0)  # Reset pointer if needed

    if not images:
        add_log("No valid training images found after face detection and augmentation.")
        return None, None

    images_np = np.array(images, dtype="uint8")
    labels_np = np.array(labels)
    add_log(f"Total augmented images: {len(images)}")
    
    # Create and train the LBPH face recognizer.
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images_np, labels_np)
    add_log("Training completed successfully with cropped and augmented faces.")
    return recognizer, expected_label


def predict_image(recognizer, test_file: UploadFile, expected_label=0, threshold=100):
    """
    Predicts the label of a test image using the trained recognizer.
    The test image is processed by detecting the face region via Haarcascade,
    cropping that region, resizing it to 100x100, and making a prediction.
    Returns True if the predicted label equals expected_label and the confidence is below the threshold.
    """
    file_bytes = test_file.file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        add_log("Error: Could not decode test image.")
        return False

    # Detect faces in the test image.
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        add_log("No face detected in test image. Prediction aborted.")
        return False

    # Use the first detected face for prediction.
    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]  # Crop the face region.
    face_resized = cv2.resize(face_img, (100, 100))  # Resize to standard size.
    
    try:
        label, confidence = recognizer.predict(face_resized)
    except Exception as e:
        add_log(f"Prediction error: {e}")
        return False

    add_log(f"Predicted label: {label}, Confidence: {confidence}")
    # The recognizer may not return an exact match but a similar face as long as confidence is below the threshold.
    return (label == expected_label) and (confidence < threshold)


@app.post("/verify")
async def verify_endpoint(
    training_images: List[UploadFile] = File(...),
    test_image: UploadFile = File(...)
):
    try:
        add_log("Verification request received.")
        if not training_images or test_image is None:
            add_log("Error: Missing training images or test image.")
            raise HTTPException(status_code=400, detail="Missing training images or test image.")

        recognizer, expected_label = train_model(training_images)
        if recognizer is None:
            raise HTTPException(status_code=400, detail="Training failed. No valid images found.")
        verified = predict_image(recognizer, test_image, expected_label)
        add_log(f"Verification result: {'Verified' if verified else 'Not Verified'}.")
        return JSONResponse(content={"verified": verified})
    except Exception as e:
        add_log(f"Unexpected error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


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
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import os
    import uvicorn
    # In production, Render (or similar services) passes the port via the PORT environment variable.
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
