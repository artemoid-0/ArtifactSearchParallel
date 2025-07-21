import cv2
import mediapipe as mp

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Load image
dataset_path = r"D:\DATASETS\artifact_dataset\test\\"
image_path = dataset_path + "image_00041_0.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process image
with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,  # enables pupil landmarks
        min_detection_confidence=0.5
) as face_mesh:
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print("Face not found.")
    else:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks on the original image
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

            # Example: get pupil coordinates
            left_pupil = face_landmarks.landmark[468]  # left pupil
            right_pupil = face_landmarks.landmark[473]  # right pupil

            h, w, _ = image.shape
            lx, ly = int(left_pupil.x * w), int(left_pupil.y * h)
            rx, ry = int(right_pupil.x * w), int(right_pupil.y * h)
            cv2.circle(annotated_image, (lx, ly), 3, (255, 0, 0), -1)
            cv2.circle(annotated_image, (rx, ry), 3, (255, 0, 0), -1)

            # Example: print coordinates
            print(f"Left pupil: ({lx}, {ly})")
            print(f"Right pupil: ({rx}, {ry})")

        # Save result
        cv2.imwrite('results/output_with_landmarks.jpg', annotated_image)
        print("Image saved: output_with_landmarks.jpg")
