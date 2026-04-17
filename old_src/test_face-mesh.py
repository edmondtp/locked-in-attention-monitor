import cv2
from face_features import FaceFeatureExtractor

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    extractor = FaceFeatureExtractor()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        face_data = extractor.process(frame)
        frame = extractor.draw_debug(frame, face_data)

        cv2.imshow("Face Mesh Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()