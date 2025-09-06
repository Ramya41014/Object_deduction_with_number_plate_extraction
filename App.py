import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr
import io

st.title("Vehicle and Number Plate Detection with OCR")

st.sidebar.header("Options")
use_webcam = st.sidebar.checkbox("Use Webcam")

if not use_webcam:
    st.write("Upload an image below to detect vehicles and number plates and extract text.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
else:
    st.write("Real-time detection using webcam feed.")

trained_model_path = 'C:/Users/Admin/Guvi_Project/Final_project/OBJECT_DEDUCTION/OBJECT_DEDUCTION/Object_Deduction/yolov8n_GPU/weights/best.pt'


@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(trained_model_path)


@st.cache_resource
def load_ocr_reader():
    try:
        reader = easyocr.Reader(['en'])
        return reader
    except Exception as e:
        st.error(f"Error initializing OCR reader: {e}")
        return None

reader = load_ocr_reader()

if model:
    st.sidebar.success("YOLO model loaded successfully!")
else:
    st.sidebar.error("Failed to load YOLO model.")

if reader:
    st.sidebar.success("OCR reader loaded successfully!")
else:
    st.sidebar.error("Failed to load OCR reader.")

def extract_text_from_plate(plate_img):
    if reader is None:
        return ""
    try:
        # Use the raw cropped plate image, no preprocessing
        result = reader.readtext(
            plate_img,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            contrast_ths=0.05,
            adjust_contrast=0.7
        )
        plate_text = " ".join([text[1] for text in result])
        return plate_text
    except Exception as e:
        st.warning(f"Error during OCR processing: {e}")
        return ""

def draw_plate_text_on_image(image, plates_info):
    annotated_img = image.copy()
    for plate in plates_info:
        x1, y1, x2, y2 = plate['bbox']
        text = plate['text']
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put text under the rectangle
        cv2.putText(
            annotated_img, text, (x1, y2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA
        )
    return annotated_img

def process_frame(frame, model, reader):
    extracted_plates_info = []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    annotated_frame = results[0].plot()
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            if class_id < len(model.names) and model.names[class_id] == 'Number plate':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                height, width = frame.shape[:2]
                y1 = max(0, y1)
                y2 = min(height, y2)
                x1 = max(0, x1)
                x2 = min(width, x2)
                number_plate_img = frame[y1:y2, x1:x2]
                if number_plate_img.shape[0] > 0 and number_plate_img.shape[1] > 0:
                    if reader:
                        plate_text = extract_text_from_plate(number_plate_img)
                        extracted_plates_info.append({
                            "bbox": (x1, y1, x2, y2),
                            "text": plate_text,
                            "confidence": confidence
                        })
                    else:
                        st.warning("OCR reader not loaded. Skipping text extraction for number plates.")
    return annotated_frame, extracted_plates_info

if use_webcam and model and reader:
    st.subheader("Webcam Feed")
    frame_placeholder = st.empty()
    text_placeholder = st.empty()
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            while use_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to capture frame from webcam.")
                    break
                annotated_frame_rgb, extracted_plates_info = process_frame(frame, model, reader)
                # Draw plate text on image
                annotated_with_text = draw_plate_text_on_image(
                    cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR),
                    extracted_plates_info
                )
                annotated_with_text_rgb = cv2.cvtColor(annotated_with_text, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(annotated_with_text_rgb, channels="RGB", use_container_width=True)
                if extracted_plates_info:
                    plate_text_display = "**Extracted Number Plates:**\n"
                    for i, plate_info in enumerate(extracted_plates_info):
                        plate_text_display += f"- Plate {i+1}: {plate_info['text']} (Confidence: {plate_info['confidence']:.2f})\n"
                    text_placeholder.markdown(plate_text_display)
                else:
                    text_placeholder.write("No number plates detected.")
    except Exception as e:
        st.error(f"An error occurred during webcam processing: {e}")
    finally:
        if cap is not None:
            cap.release()
            print("Webcam released.")

elif uploaded_file is not None and model and reader:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    submit = st.button("Submit")
    if submit:
        st.subheader("Detection Results")
        # Run YOLO detection
        annotated_img_rgb, extracted_plates_info = process_frame(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR), model, reader)
        # Draw number plate text on YOLO-annotated image (keep YOLO boxes and add text)
        annotated_img_bgr = cv2.cvtColor(annotated_img_rgb, cv2.COLOR_RGB2BGR)
        def draw_plate_text_on_image_yellow(image, plates_info):
            annotated_img = image.copy()
            for plate in plates_info:
                x1, y1, x2, y2 = plate['bbox']
                text = plate['text']
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put text under the rectangle in yellow
                cv2.putText(
                    annotated_img, text, (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA
                )
            return annotated_img

        annotated_with_text = draw_plate_text_on_image_yellow(
            annotated_img_bgr,
            extracted_plates_info
        )
        annotated_with_text_rgb = cv2.cvtColor(annotated_with_text, cv2.COLOR_BGR2RGB)
        st.image(annotated_with_text_rgb, caption='Detection Results', use_container_width=True)

        # Download button for annotated image (after display)
        is_success, buffer = cv2.imencode(".png", annotated_with_text)
        if is_success:
            img_bytes = io.BytesIO(buffer).getvalue()
            st.download_button(
                label="Download Annotated Image",
                data=img_bytes,
                file_name="annotated_image.png",
                mime="image/png"
            )

        # Display extracted text below the image
        st.markdown("### Extracted Number Plates")
        if extracted_plates_info:
            for i, plate_info in enumerate(extracted_plates_info):
                st.write(f"**Plate {i+1}:** {plate_info['text']} (Confidence: {plate_info['confidence']:.2f})")
            extracted_text_string = "Extracted Number Plate Information:\n\n"
            for i, plate_info in enumerate(extracted_plates_info):
                extracted_text_string += f"Number Plate {i+1}:\n"
                extracted_text_string += f"  Text: {plate_info['text']}\n"
                extracted_text_string += f"  Confidence: {plate_info['confidence']:.2f}\n\n"
        else:
            st.write("No number plates detected.")
            extracted_text_string = "No number plates detected or extracted.\n"

        st.download_button(
            label="Download Extracted Text",
            data=extracted_text_string.encode('utf-8'),
            file_name="extracted_plates.txt",
            mime="text/plain"
        )

        # Show other detections
        st.subheader("Other Detections")
        other_detections_found = False
        results = model(img_array)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                if class_name != 'Number plate':
                    st.write(f"Detected {class_name} (Confidence: {confidence:.2f})")
                    other_detections_found = True
        if not other_detections_found:
            st.write("No other vehicles or persons detected.")

elif (use_webcam or uploaded_file is not None) and (not model or not reader):
    if use_webcam:
        if not model:
            st.error("YOLO model not loaded. Cannot start webcam detection.")
        if not reader:
            st.error("OCR reader not loaded. Cannot perform OCR on webcam feed.")
    elif uploaded_file is not None:
        if not model:
            st.error("YOLO model not loaded. Cannot perform object detection on uploaded image.")
        if not reader:
            st.error("OCR reader not loaded. Cannot perform OCR on uploaded image.")

else:
    st.write("Please select an option from the sidebar or upload an image to get started.")
