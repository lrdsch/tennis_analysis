from ultralytics import YOLO 
import cv2
import pickle
import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        # Gruppi di keypoint: ogni keypoint è una coppia (x, y)
        group_1_indices = [0, 1, 4, 6, 8, 12, 9]
        group_2_indices = [2, 3, 5, 7, 10, 11, 13]

        def get_keypoint_coords(indices):
            return [(court_keypoints[i * 2], court_keypoints[i * 2 + 1]) for i in indices]

        group_1_coords = get_keypoint_coords(group_1_indices)
        group_2_coords = get_keypoint_coords(group_2_indices)

        distances_to_group_1 = []
        distances_to_group_2 = []

        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            # distanza media verso ciascun gruppo
            d1 = min([measure_distance(player_center, kp) for kp in group_1_coords])
            d2 = min([measure_distance(player_center, kp) for kp in group_2_coords])

            distances_to_group_1.append((track_id, d1))
            distances_to_group_2.append((track_id, d2))

        # Ordina per distanza crescente
        distances_to_group_1.sort(key=lambda x: x[1])
        distances_to_group_2.sort(key=lambda x: x[1])

        player_1 = distances_to_group_1[0][0]
        # Evita di assegnare lo stesso player due volte
        for track_id, _ in distances_to_group_2:
            if track_id != player_1:
                player_2 = track_id
                break

        return [player_1, player_2]

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    def detect_frame(self,frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    