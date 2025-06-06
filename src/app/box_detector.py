import cv2
import numpy as np
import time
from .models import load_models, preprocess_for_classification, get_traffic_sign_name


class Detector:
    def __init__(self):
        print("Initializing Detector...")
        self.detection_model, self.classification_model = load_models()

        if self.detection_model is None or self.classification_model is None:
            raise Exception("Failed to load models")

        # Thresholds
        self.detection_confidence = 0.6
        self.classification_confidence = 0.7

        # Statistics
        self.total_detections = 0
        self.processing_times = []

        print("Detector initialized successfully")

    def process_frame(self, frame):
        start_time = time.time()

        # Detect traffic signs
        detections = self.detect_traffic_signs(frame)

        # Process detections
        processed_frame = frame.copy()
        detections_info = []

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            detection_conf = detection['confidence']

            # Crop and classify
            sign_crop = frame[y1:y2, x1:x2]
            if sign_crop.size > 0:
                classification = self.classify_traffic_sign(sign_crop)

                if classification:
                    class_name = classification['class_name']
                    class_conf = classification['confidence']
                    combined_conf = (detection_conf * class_conf) / 100

                    # Draw detection
                    processed_frame = self.draw_detection(
                        processed_frame, (x1, y1, x2, y2), class_name, combined_conf
                    )

                    # Add to results
                    detections_info.append({
                        'class_name': class_name,
                        'confidence': combined_conf * 100,
                        'bbox': (x1, y1, x2, y2)
                    })

        # Update statistics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)

        self.total_detections += len(detections_info)

        return processed_frame, detections_info

    def detect_traffic_signs(self, frame):
        try:
            results = self.detection_model(frame, conf=self.detection_confidence, verbose=False)
            detections = []

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())

                        if x2 > x1 and y2 > y1:
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'confidence': confidence
                            })

            return detections
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []

    def classify_traffic_sign(self, sign_crop):
        try:
            preprocessed = preprocess_for_classification(sign_crop)
            predictions = self.classification_model.predict(preprocessed, verbose=0)

            class_id = np.argmax(predictions[0])
            confidence = float(predictions[0][class_id]) * 100

            if confidence >= self.classification_confidence:
                return {
                    'class_name': get_traffic_sign_name(class_id),
                    'confidence': confidence
                }
            return None
        except Exception as e:
            print(f"Classification error: {e}")
            return None

    def draw_detection(self, frame, bbox, class_name, confidence):
        x1, y1, x2, y2 = bbox

        # Color based on confidence
        if confidence > 0.8:
            color = (0, 255, 0)  # Green
        elif confidence > 0.6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label = f"{class_name}: {confidence:.1%}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def get_statistics(self):
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            'total_detections': self.total_detections,
            'avg_processing_time': avg_time,
            'fps': fps
        }