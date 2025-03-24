from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from datetime import datetime
from typing import List
import tempfile
import traceback
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

debug_logs = []

def add_log(message: str):
    entry = f"{datetime.now().isoformat()}: {message}"
    debug_logs.append(entry)
    if len(debug_logs) > 50:
        debug_logs.pop(0)
    print(entry)

def augment_image(img: np.ndarray) -> List[np.ndarray]:
    try:
        augmented_images = [img]
        flipped = cv2.flip(img, 1)
        augmented_images.append(flipped)
        h, w = img.shape[:2]
        start_row, start_col = int(0.1 * h), int(0.1 * w)
        end_row, end_col = int(0.9 * h), int(0.9 * w)
        cropped = img[start_row:end_row, start_col:end_col]
        zoomed = cv2.resize(cropped, (w, h))
        augmented_images.append(zoomed)
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=50)
        augmented_images.append(bright)
        noise = np.random.normal(0, 10, img.shape)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        augmented_images.append(noisy)
        return augmented_images
    except Exception as e:
        add_log(f"Error in augment_image: {str(e)}")
        raise

async def train_model(xml_bytes: bytes, training_files: List[UploadFile]) -> tuple:
    try:
        add_log(f"Received Haar Cascade XML with size {len(xml_bytes)} bytes.")
        with tempfile.NamedTemporaryFile(delete=True, suffix='.xml') as temp_file:
            temp_file.write(xml_bytes)
            temp_file.flush()
            detector = cv2.CascadeClassifier(temp_file.name)
            if detector.empty():
                add_log("Error: Failed to load Haar Cascade classifier from uploaded XML.")
                raise ValueError("Invalid Haar Cascade XML file.")

        images = []
        labels = []
        expected_label = 0
        add_log(f"Received {len(training_files)} training files.")
        for file in training_files:
            file_bytes = await file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                add_log(f"Warning: Could not decode training file {file.filename}. Skipping.")
                continue
            faces = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                add_log(f"No face detected in training image {file.filename}. Skipping.")
                continue
            for (x, y, w, h) in faces:
                cropped = img[y:y+h, x:x+w]
                break
            img_resized = cv2.resize(cropped, (100, 100))
            augmented_images = augment_image(img_resized)
            for aug_img in augmented_images:
                images.append(aug_img)
                labels.append(expected_label)
            await asyncio.sleep(0.01)  # Yield control to prevent blocking
        if not images:
            add_log("No valid training images with detectable faces found.")
            return None, None
        images_np = np.array(images, dtype="uint8")
        labels_np = np.array(labels)
        add_log(f"Training with {len(images)} total images (including augmentations).")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(images_np, labels_np)
        add_log("Training completed successfully.")
        return recognizer, expected_label
    except Exception as e:
        add_log(f"Error in train_model: {str(e)}")
        raise

async def predict_image(recognizer, test_file: UploadFile, xml_bytes: bytes, expected_label=0, threshold=100) -> bool:
    try:
        add_log(f"Received Haar Cascade XML with size {len(xml_bytes)} bytes.")
        with tempfile.NamedTemporaryFile(delete=True, suffix='.xml') as temp_file:
            temp_file.write(xml_bytes)
            temp_file.flush()
            detector = cv2.CascadeClassifier(temp_file.name)
            if detector.empty():
                add_log("Error: Failed to load Haar Cascade classifier from uploaded XML.")
                raise ValueError("Invalid Haar Cascade XML file.")

        file_bytes = await test_file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            add_log("Error: Could not decode test image.")
            return False

        faces = detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            add_log("No face detected in test image.")
            return False

        for (x, y, w, h) in faces:
            cropped = img[y:y+h, x:x+w]
            break
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
    test_image: UploadFile = File(...),
    haarcascade_xml: UploadFile = File(...)
):
    add_log("Verification request received.")
    try:
        xml_bytes = await haarcascade_xml.read()

        recognizer, expected_label = await asyncio.wait_for(
            train_model(xml_bytes, training_images), timeout=30.0
        )
        if recognizer is None:
            add_log("Training failed due to no valid images.")
            raise HTTPException(status_code=400, detail="Training failed. No valid images.")

        verified = await asyncio.wait_for(
            predict_image(recognizer, test_image, xml_bytes, expected_label), timeout=30.0
        )
        add_log(f"Verification result: {'Verified' if verified else 'Not Verified'}.")
        return {"verified": verified}
    except asyncio.TimeoutError:
        add_log("Request timed out.")
        raise HTTPException(status_code=504, detail="Request processing timed out.")
    except Exception as e:
        add_log(f"Server error: {str(e)}\nStack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/logs")
async def get_logs():
    return {"logs": debug_logs}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
