import cv2
from ultralytics import YOLO

# -------------------------------
# Load YOLO Model
# -------------------------------
model = YOLO("models/best.pt")

# ---------------------
# ----------
# Open Webcam
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam!")
    exit()

print("Press 'q' to quit.")

# -------------------------------
# Real-Time Detection
# -------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Run YOLO
    results = model.predict(
        source=frame,
        conf=0.4,
        verbose=False
    )

    # Draw detections
    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            confidence = float(box.conf[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Face {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("YOLO Face Detection", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()