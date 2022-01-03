from imageai.Detection.Custom import CustomVideoObjectDetection
import os
import cv2

execution_path = os.getcwd()


class RealtimeVideoObjectDetection(CustomVideoObjectDetection):
    def __init__(self):
        super().__init__()

    def detectObjectsFromVideo(self,
                               minimum_percentage_probability=50,
                               display_percentage_probability=True,
                               display_object_name=True
                               ):

        input_video = cv2.VideoCapture(0)

        video_frames_count = 0

        while (True):
            _, frame = input_video.read()
            print(frame)
            detected_frame = frame.copy()
            video_frames_count += 1
            try:
                detected_frame, _ = self.__detector.detectObjectsFromImage(
                    input_image=frame, input_type="array", output_type="array",
                    minimum_percentage_probability=minimum_percentage_probability,
                    display_percentage_probability=display_percentage_probability,
                    display_object_name=display_object_name)
            except:
                None
            cv2.imshow('frame', detected_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            input_video.release()


def detect_from_camera():
    detector = RealtimeVideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=os.path.join(
        execution_path, "detection_model-ex-33--loss-4.97.h5"))
    detector.setJsonPath(configuration_json=os.path.join(
        execution_path, "detection_config.json"))
    detector.loadModel()
    detector.detectObjectsFromVideo()


if __name__ == "__main__":
    detect_from_camera()
