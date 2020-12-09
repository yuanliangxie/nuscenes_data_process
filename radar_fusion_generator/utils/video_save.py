import cv2
import os
def video_save_from_capture(video_name, video_imgsize):
    video_path = os.path.join('.', video_name)
    video_save = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"XVID"), 5.0, video_imgsize)
    return video_save