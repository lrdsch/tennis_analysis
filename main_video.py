from utils import (read_video, 
                   save_video,
                   )
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

import cv2
import pandas as pd
from copy import deepcopy


def main():
    suffix = "video_2"
    # Read Video
    input_video_path = "input_videos/input_" + suffix + ".mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path = 'models/yolo5_last.pt')
    
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub = False, 
                                                     stub_path = 'tracker_stubs/player_detections.pkl')
    
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=False,
                                                 stub_path='tracker_stubs/ball_tracker.pkl')
    
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # Court Line Detector model
    court_model_path = 'models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints_per_frame = []
    for frame in video_frames:
        keypoints = court_line_detector.predict(frame)
        court_keypoints_per_frame.append(keypoints)
    
    # choose players 
    player_detections = player_tracker.choose_and_filter_players(court_keypoints_per_frame[0], player_detections)
    
    # Draw output
   
    ## Draw Player Bounding Boxes
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    
    ## Draw court keypoints
    output_video_frames = [
    court_line_detector.draw_keypoints(frame, keypoints)
    for frame, keypoints in zip(output_video_frames, court_keypoints_per_frame)
]
    
    # Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 255), 2)
    
    save_video(output_video_frames, "output_videos/output_" + suffix + ".avi")

if __name__ == "__main__":
    main()
    print('All good')