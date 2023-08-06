import json
import numpy as np


class Restriction():

    def __init__(self, filename: str, landmarks: dict):
        self.filename = filename
        self.landmarks = landmarks
        self.rest, self.angle_rest, self.distance_rest, self.movement_rest = self.get_restrictions()

    def get_restrictions(self):
        with open(self.filename, "r") as f:
            data = f.read()

        data = json.loads(data)

        try:
            angle_rest = data["angle"]
        except:
            pass
        try:
            distance_rest = data["distance"]
        except:
            distance_rest = {}
        try:
            movement_rest = data["movement"]
        except:
            movement_rest = {}

        return data, angle_rest, distance_rest, movement_rest

    def clac_angle(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        dot_product = np.dot(v1, v2)
        magnitude_v1 = np.linalg.norm(v1)
        magnitude_v2 = np.linalg.norm(v2)
        angle = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
        return np.degrees(angle)

    def clac_length(self, v):
        v = np.array(v)
        length = np.linalg.norm(v)
        return length

    def clac_movement(self):
        pass

    def calc_vector(self, point1, point2):

        x1 = point1[0]
        y1 = point1[1]
        z1 = point1[2]
        x2 = point2[0]
        y2 = point2[1]
        z2 = point2[2]

        return [x2 - x1, y2 - y1, z2 - z1]

    def get_point(self, tag: str):
        x = self.landmarks[tag].x
        y = self.landmarks[tag].y
        z = self.landmarks[tag].z

        return [x, y, z]

    def calc_mistakes(self, mistake_type: str):
        report = {}

        if mistake_type == "angle":

            for rest in self.angle_rest.keys():

                cur_min = self.angle_rest[rest]["range"][0]
                cur_max = self.angle_rest[rest]["range"][1]

                lm_1 = self.angle_rest[rest]["points"][0]
                center_tag = self.angle_rest[rest]["points"][1]
                lm_2 = self.angle_rest[rest]["points"][2]

                v1 = self.calc_vector(
                    self.get_point(lm_1),
                    self.get_point(center_tag)
                )
                v2 = self.calc_vector(
                    self.get_point(lm_2),
                    self.get_point(center_tag)
                )

                c_angle = self.clac_angle(v1, v2)
                if c_angle < cur_min or c_angle > cur_max:
                    report[rest] = {
                        "points": self.angle_rest[rest]["points"],
                        "message": "Fix angle"
                    }

        elif mistake_type == "distance":
            for rest in self.distance_rest.keys():

                cur_min = self.distance_rest[rest]["range"][0]
                cur_max = self.distance_rest[rest]["range"][1]

                lm_1 = self.distance_rest[rest]["points"][0]
                lm_2 = self.distance_rest[rest]["points"][1]

                v1 = self.calc_vector(
                    self.get_point(lm_1),
                    self.get_point(lm_2)
                )

                c_length = self.clac_length(v1)
                if c_length < cur_min or c_length > cur_max:
                    report[rest] = {
                        "points": self.distance_rest[rest]["points"],
                        "message": "Fix distance"
                    }

        return report

    def get_mistakes(self):
        """
        This method building a mistake report
        """
        report = {
            "angle": self.calc_mistakes(mistake_type="angle"),
            "distance": self.calc_mistakes(mistake_type="distance"),
            # "movement" : self.calc_mistakes(mistake_type = "movement")
        }
        return report