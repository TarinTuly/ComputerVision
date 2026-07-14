from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")


def detect(frame):

    results = model.predict(
        source=frame,
        conf=0.5,
        verbose=False
    )

    detections = []

    for result in results:

        for box in result.boxes:

            coords = box.xyxy[0].cpu().numpy().astype(int)

            x1, y1, x2, y2 = coords

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face = frame[y1:y2, x1:x2]

            detections.append({
                "box": (x1, y1, x2, y2),
                "face": face
            })

    return detections