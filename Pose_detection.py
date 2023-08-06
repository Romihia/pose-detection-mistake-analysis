import os
import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
from Restrictions import Restriction
from cvzone.PoseModule import PoseDetector


class PoseDetection:

    def __init__(self, csv_url: str = "./pose_train.csv") -> None:
        # Initializing mediapipe pose class.
        self.mp_pose = mp.solutions.pose
        # Setting up the Pose function.
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=0)
        # Initializing mediapipe drawing class, useful for annotation.
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmarks = {
            "NOSE": 0,
            "LEFT_EYE_INNER": 1,
            "LEFT_EYE": 2,
            "LEFT_EYE_OUTER": 3,
            "RIGHT_EYE_INNER": 4,
            "RIGHT_EYE": 5,
            "RIGHT_EYE_OUTER": 6,
            "LEFT_EAR": 7,
            "RIGHT_EAR": 8,
            "MOUTH_LEFT": 9,
            "MOUTH_RIGHT": 10,
            "LEFT_SHOULDER": 11,
            "RIGHT_SHOULDER": 12,
            "LEFT_ELBOW": 13,
            "RIGHT_ELBOW": 14,
            "LEFT_WRIST": 15,
            "RIGHT_WRIST": 16,
            "LEFT_PINKY": 17,
            "RIGHT_PINKY": 18,
            "LEFT_INDEX": 19,
            "RIGHT_INDEX": 20,
            "LEFT_THUMB": 21,
            "RIGHT_THUMB": 22,
            "LEFT_HIP": 23,
            "RIGHT_HIP": 24,
            "LEFT_KNEE": 25,
            "RIGHT_KNEE": 26,
            "LEFT_ANKLE": 27,
            "RIGHT_ANKLE": 28,
            "LEFT_HEEL": 29,
            "RIGHT_HEEL": 30,
            "LEFT_FOOT_INDEX": 31,
            "RIGHT_FOOT_INDEX": 32,
        }
        self.csv_url = csv_url
        self.csv_dataframe = self.read_pose_info()

    def add_row(self, landmark_info_from_result, target: str, image_path: str):

        temp = {}
        for key, result in zip(list(self.landmarks.keys()), landmark_info_from_result.landmark):
            temp[f"{key}_x"] = [result.x]
            temp[f"{key}_y"] = [result.y]
            temp[f"{key}_z"] = [result.z]
            temp[f"{key}_v"] = [result.visibility]

        temp["target"] = [target]
        temp["image_path"] = [image_path]

        df = pd.DataFrame.from_dict(temp)

        self.csv_dataframe = pd.concat([self.csv_dataframe, df])

        self.write_pose_info()

        return df

    def read_pose_info(self):
        if os.path.exists(self.csv_url):
            df = pd.read_csv(self.csv_url)

            return df
        else:

            temp = {}
            for key in self.landmarks.keys():
                temp[f"{key}_x"] = []
                temp[f"{key}_y"] = []
                temp[f"{key}_z"] = []
                temp[f"{key}_v"] = []

            temp["target"] = []
            temp["image_path"] = []

            df = pd.DataFrame.from_dict(temp)

            return df

    def write_pose_info(self):

        self.csv_dataframe.to_csv(self.csv_url, index=False)

    def detect_pose(self, image_path: str, tag: str = None, display: bool = True):

        '''
        This function performs pose detection on an image.
        Args:
            image: The input image with a prominent person whose pose landmarks needs to be detected.
            pose: The pose setup function required to perform the pose detection.
            display: A boolean value that is if set to true the function displays the original input image, the resultant image,
                    and the pose landmarks in 3D plot and returns nothing.
        Returns:
            output_image: The input image with the detected pose landmarks drawn.
            landmarks: A list of detected landmarks converted into their original scale.
        '''

        image = cv2.imread(image_path)
        # Create a copy of the input image.
        output_image = image.copy()
        # Convert the image from BGR into RGB format.
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Perform the Pose Detection.
        results = self.pose.process(imageRGB)
        # Retrieve the height and width of the input image.
        height, width, _ = image.shape

        # Initialize a list to store the detected landmarks.
        landmarks = []

        # Check if any landmarks are detected.
        if results.pose_landmarks:

            # Draw Pose landmarks on the output image.
            self.mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                           connections=self.mp_pose.POSE_CONNECTIONS)

            # Iterate over the detected landmarks.
            for landmark in results.pose_landmarks.landmark:
                # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))

        if display:
            # Display the original input image and the resultant image.
            plt.figure(figsize=[22, 22])
            plt.subplot(121)
            plt.imshow(image[:, :, ::-1])
            plt.title("Original Image")
            plt.axis('off')
            plt.subplot(122)
            plt.imshow(output_image[:, :, ::-1])
            plt.title("Output Image")
            plt.axis('off')

            # Also Plot the Pose landmarks in 3D.
            self.mp_drawing.plot_landmarks(results.pose_world_landmarks, self.mp_pose.POSE_CONNECTIONS)

        if tag is not None:
            target = tag
        else:
            # image naming convention - deadlift-legsplit.1.png
            target = os.path.basename(image_path).split(".")[0]

        self.add_row(results.pose_world_landmarks, target, image_path=image_path)

        image_landmarks_results = {}
        for landmark, result in zip(self.landmarks.keys(), list(results.pose_world_landmarks.landmark)):
            image_landmarks_results[landmark] = result

        return image_landmarks_results

        # Return the output image and the found landmarks.
        return output_image, results.pose_world_landmarks

    def detect_pose_live(self, tag: str = None, filename: str = None, display: bool = True):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pd = PoseDetector(trackCon=0.5, detectionCon=0.5)

        cap = cv2.VideoCapture(0)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Render detections lines and dots on the body
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(224, 224, 224), thickness=3, circle_radius=2)
                                          )

                try:
                    image_landmarks_results = {}
                    for landmark, result in zip(self.landmarks.keys(), list(results.pose_world_landmarks.landmark)):
                        # print(self.landmarks.keys() , list(results.pose_world_landmarks.landmark))
                        image_landmarks_results[landmark] = result
                except:
                    pass

                rest = Restriction(filename, image_landmarks_results)
                title = rest.get_restrictions()

                pd.findPose(image, draw=0)
                lmlist, bbox = pd.findPosition(image, draw=0, bboxWithHands=0)

                try:
                    j = 40
                    for key in title[1]:
                        cv2.putText(image, str(rest.get_mistakes()['angle'][key]['message'])
                                    , (90, j), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 2, cv2.LINE_AA)
                        point = lmlist[self.landmarks[key]]
                        cx, cy = point[1], point[2]
                        cv2.circle(image, (cx, cy), 10, (0, 0, 255), 5)
                        cv2.circle(image, (cx, cy), 15, (0, 0, 255), 5)
                        j = j + 25





                except:
                    cv2.putText(image, 'Excellent position'
                                , (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 250, 0), 2, cv2.LINE_AA)

                cv2.imshow('Mediapipe Feed', image)

                if (cv2.waitKey(10) & 0xFF == ord('q')) or (cv2.waitKey(10) & 0xFF == ord('Q')):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def get_landmarks(self, landmark: dict = None):
        return landmark

    def get_Current_restriction(self, rest: Restriction = None):
        return rest