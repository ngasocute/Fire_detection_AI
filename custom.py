import os
import cv2
from imageai.Detection.Custom import (
    CustomVideoObjectDetection,
    CustomObjectDetection
)


execution_path = os.getcwd()


class RealtimeVideoObjectDetection(CustomVideoObjectDetection):
    def __init__(self, model_path, json_path):
        self.__detector = CustomObjectDetection()
        self.__detector.setModelTypeAsYOLOv3()
        self.__detector.setModelPath(detection_model_path=model_path)
        self.__detector.setJsonPath(configuration_json=json_path)
        self.__detector.loadModel()

    def detect_from_camera(self, frame_detection_interval=1, minimum_percentage_probability=50,
                           display_percentage_probability=True, display_object_name=True):
        """
        Detect fire with input from webcam
        """
        input_video = cv2.VideoCapture(0)
        counting = 0

        video_frames_count = 0

        while (input_video.isOpened()):
            ret, frame = input_video.read()
            if (ret == True):
                detected_frame = frame.copy()
                video_frames_count += 1
                counting += 1

                check_frame_interval = counting % frame_detection_interval

                if (counting == 1 or check_frame_interval == 0):
                    try:
                        detected_frame, _ = self.__detector.detectObjectsFromImage(
                            input_image=frame, input_type="array", output_type="array",
                            minimum_percentage_probability=minimum_percentage_probability,
                            display_percentage_probability=display_percentage_probability,
                            display_object_name=display_object_name)
                    except Exception as e:
                        print(e)
                        None
                cv2.imshow("frame", detected_frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        input_video.release()


def fire_detection():
    model_path = os.path.join(
        execution_path, "detection_model-ex-33--loss-4.97.h5")
    json_path = os.path.join(execution_path, "detection_config.json")
    detector = RealtimeVideoObjectDetection(
        model_path=model_path, json_path=json_path)
    detector.detect_from_camera()


if __name__ == "__main__":
    fire_detection()
