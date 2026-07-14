from deepface import DeepFace

def analyze_face(face):

    try:
        result = DeepFace.analyze(
            img_path=face,
            actions=["age", "gender", "emotion"],
            detector_backend="skip",
            enforce_detection=False,
            silent=True
        )[0]

        age = result["age"]

        gender_data = result["gender"]

        if isinstance(gender_data, dict):
            gender = max(gender_data, key=gender_data.get)
        else:
            gender = gender_data

        emotion = result["dominant_emotion"]

        return age, gender, emotion

    except Exception:
        return None, None, None