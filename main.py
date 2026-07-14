import cv2
from ultralytics import YOLO
from deepface import DeepFace

# -------------------------------
# Load YOLO Face Detection Model
# -------------------------------
model = YOLO("models/best.pt")

# -------------------------------
# Open Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam!")
    exit()

print("Press 'q' to quit.")

# -------------------------------
# Main Loop
# -------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO Face Detection
    results = model.predict(
        source=frame,
        conf=0.5,
        verbose=False
    )

    # Process each detected face
    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Prevent invalid coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            age = "N/A"
            gender = "N/A"
            emotion = "N/A"

            try:

                analysis = DeepFace.analyze(
                    img_path=face,
                    actions=["age", "gender", "emotion"],
                    enforce_detection=False,
                    detector_backend="skip",
                    silent=True
                )[0]

                age = analysis["age"]

                gender = max(
                    analysis["gender"],
                    key=analysis["gender"].get
                )

                emotion = analysis["dominant_emotion"]

            except Exception:
                pass

            # Draw Rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw Information
            cv2.putText(
                frame,
                f"Age: {age}",
                (x1, y1 - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Gender: {gender}",
                (x1, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Emotion: {emotion}",
                (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Real-Time Face Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("1")

import cv2
print("2")

from ultralytics import YOLO
print("3")

from deepface import DeepFace
print("4")

model = YOLO("models/best.pt")
print("5")

cap = cv2.VideoCapture(0)
print("6")

if not cap.isOpened():
    print("Camera failed")
    exit()

print("7")