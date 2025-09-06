

🚗 Vehicle Number Plate Detection System
An system for automatic detection and recognition of vehicle number plates using computer vision and deep learning. This project leverages YOLO for object detection and OCR (Tesseract/EasyOCR) for text extraction, deployed via a Streamlit web application for real-time use.

📌 Problem Statement
Manual identification of vehicle plates is time-consuming, error-prone, and inefficient. This project aims to automate the process, enhancing accuracy and reliability across domains like law enforcement, toll collection, parking management, and traffic analytics.

🎯 Objectives
- Detect vehicles, people, and number plates in diverse environments and lighting conditions.
- Accurately extract number plate regions despite occlusions, background clutter, and format variations.
- Apply OCR to extract alphanumeric text from detected plates.
- Deploy the system as a user-friendly web application for real-time processing.

Business Use Cases:
1. Traffic Management:

Automated vehicle identification can help track vehicles, enforce traffic
regulations, and issue fines for violations such as speeding and red-light
jumping.
2. Law Enforcement:

Helps police and security agencies track stolen or unauthorized vehicles,
aiding investigations.

3. Toll Collection:

Enables seamless electronic toll collection without requiring vehicles to
stop, reducing congestion at toll plazas.

4. Parking Systems:

Automates access control in parking lots, allowing only authorized
vehicles and generating logs of entry and exit times.

5. Fleet Management:

Logistics and transportation companies can use automated number plate
recognition to monitor fleet movements and enhance tracking accuracy.

6. Security Monitoring:

Enhances security at gated communities, offices, and industrial sites by
ensuring only registered vehicles are allowed entry.

7. Urban Planning & Analytics:

Governments and municipalities can analyze traffic patterns to make
informed decisions on infrastructure development.



🧠 Approach
1. Data Collection
- Diverse vehicle images from multiple angles, lighting, and weather conditions.
- Annotated datasets in YOLO format.
2. Preprocessing
- Image resizing and normalization.
- Augmentation: rotation, brightness, noise, flipping, cropping.
- Filtering low-quality images.
3. Model Training
- YOLO for real-time detection of vehicles, people, and plates.
- Hyperparameter tuning (learning rate, batch size, anchor boxes).
- GPU-accelerated training.
4. Post-Processing
- Plate region extraction and OCR via Tesseract/EasyOCR.
- Binarization and contrast enhancement for improved OCR accuracy.
5. Deployment
- Streamlit-based web app with image upload and webcam support.
- Real-time detection and OCR output.
- Exportable detection logs.

🖥️ Streamlit Web App Features
- 📸 Image Upload: Upload vehicle images for detection.
- 🎥 Webcam Support: Real-time detection via live feed.
- 🧾 Detection Results: Bounding boxes and confidence scores.
- 🔡 OCR Output: Extracted plate text displayed.
- 📥 Download Logs: Export detection results for future use.

🚀 Getting Started
Installation
git clone https://github.com/Ramya41014/Object_deduction_with_number_plate_extraction.git
cd Object_deduction_with_number_plate_extraction
pip install -r requirements.txt


Run the App
streamlit run app.py

📈 Future Enhancements
- Expand dataset for better generalization.
- Optimize model for edge deployment.
- Integrate database for vehicle tracking.



Let me know if you'd like help customizing the README for a specific GitHub repo structure or adding visuals like architecture diagrams or sample outputs.
