import Pose_detection
import Restrictions
import os


# rest = Restriction(
    # filename = "Right_hand_dumbbell_lift.json" , 
    # landmarks = PoseDetection().detect_pose(image_path="training-with-focus.jpg" , display=False)
    # )
#
play=Pose_detection.PoseDetection()

play.detect_pose_live(filename = "deadlift.json")

#  rest = Restriction( filename , image_landmarks_results)
#   title = rest.get_restrictions()
#   print(list(rest.get_restrictions()['angle']))

