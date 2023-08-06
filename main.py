import Pose_detection
import Restrictions
import os


#Pose Detection on Images
pose_detector = PoseDetection()
image_path = "path/to/your/image.jpg"
landmarks = pose_detector.detect_pose(image_path, display=True)


#Live Pose Detection

play=Pose_detection.PoseDetection()
play.detect_pose_live(filename = "deadlift.json")



