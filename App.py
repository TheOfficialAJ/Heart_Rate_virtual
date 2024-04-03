import cv2
from region_extractor import FaceExtractor
from heart_rate import HeartRate
# import line_profiler
from matplotlib import pyplot as plt

# @profile
def main():
    face_extractor = FaceExtractor("./weights/yolov8n-face.pt", "./weights/facemesh.pth")
    hr_estimator = HeartRate()
    hr_estimator.set_method("POS")
    feed = cv2.VideoCapture("../../P1_Virtual_Dataset/extracted_cases/case13_10sec.mp4")
    # feed = cv2.VideoCapture("../../P1_Virtual_Dataset/casualty_videos/uncompressed/Route_01/UAV_P01_1m_F_RGB.mp4")
    i = 1
    while True:
        ret, frame = feed.read()
        if not ret:
            break
        if i % 50 == 0:
            face_extractor.process_images()
            hr_estimator.process_signal(face_extractor)
            hr_estimator.update_image(face_extractor)
            face_extractor.clear_images()
        if i % 100 == 0:
            hr_estimator.estimateHR()
        face_extractor.add_image(frame)

        i += 1

    plt.pause(10)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
