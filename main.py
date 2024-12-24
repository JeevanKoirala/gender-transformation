import cv2
import numpy as np
from PIL import Image
from gfpgan import GFPGANer

print("Loading the GFPGAN model. This might take a while...")
gfpgan = GFPGANer(
    model_path='GFPGANv1.3.pth',
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    device='cpu'
)
print("Model loaded successfully!")

def gender_transform_live(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    np_image = np.array(pil_image)
    cropped_faces, restored_faces, _ = gfpgan.enhance(np_image, has_aligned=False, only_center_face=False)
    if len(restored_faces) == 0:
        print("[INFO] No faces detected in the frame.")
        return frame
    processed_face = restored_faces[0]
    processed_face_resized = cv2.resize(processed_face, (frame.shape[1], frame.shape[0]))
    return processed_face_resized

def start_live_comparison():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to access the camera.")
        return
    print("[INFO] Camera accessed successfully. Starting live feed...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame from camera.")
            break
        try:
            transformed_frame = gender_transform_live(frame)
        except Exception as e:
            print(f"[ERROR] Transformation failed: {e}")
            transformed_frame = frame
        comparison_frame = np.hstack((frame, cv2.cvtColor(transformed_frame, cv2.COLOR_RGB2BGR)))
        cv2.imshow('Live Gender Transformation Comparison', comparison_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting the live feed.")
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting the gender transformation live feed. Press 'q' to quit.")
    start_live_comparison()
