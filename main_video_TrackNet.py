from utils import (read_video, 
                   save_video,
                   )
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

import cv2
import pandas as pd
from copy import deepcopy

from TrackNet.model import BallTrackerNet
from TrackNet.general import postprocess
import torch
from TrackNet import infer_on_video
from TrackNet import kalmar


def main():
    suffix = "video_2_rescaled"
    # Read Video
    input_video_path = "input_videos/input_" + suffix + ".mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path = 'models/yolo5_last.pt')
    ball_TrackNet = BallTrackerNet()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ball_TrackNet.load_state_dict(torch.load('TrackNet/model_best.pt', map_location=device))
    ball_TrackNet = ball_TrackNet.to(device)
    ball_TrackNet.eval()
    
    # player detection
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub = False, 
                                                     stub_path = 'tracker_stubs/player_detections.pkl')
    
    # ball detection with YOLO
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=False,
                                                 stub_path='tracker_stubs/ball_tracker.pkl')
    
    # eliminate the interpolation on the yolo model for the ball
    # ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # ball detection with TrackNet
    frames, fps = infer_on_video.read_video(input_video_path)
    ball_track, dists = infer_on_video.infer_model(frames, ball_TrackNet, device = device) 
    ball_track = infer_on_video.remove_outliers(ball_track, dists)    
    
    # Applica filtro di Kalman alla traccia della palla di balltrackernet
    kf = kalmar.KalmanFilter2D()
    filtered_ball_track = []

    for pt in ball_track:
        if pt[0] is not None and pt[1] is not None:
            # correzione se punto valido
            kf.correct(pt[0], pt[1])
            pred_x, pred_y = kf.predict()
            filtered_ball_track.append([pred_x, pred_y])
        else:
            # predizione se punto mancante
            pred_x, pred_y = kf.predict()
            filtered_ball_track.append([pred_x, pred_y])
    ball_track = filtered_ball_track
    
    # eliminate the interpolation on the TrackNet model for the ball
    #if True: # extrapolation (interpolation)
        #subtracks = infer_on_video.split_track(ball_track)
        #for r in subtracks:
            #ball_subtrack = ball_track[r[0]:r[1]]
            #ball_subtrack = infer_on_video.interpolation(ball_subtrack)
            #ball_track[r[0]:r[1]] = ball_subtrack
    
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
    
    # draw BallTracker results  
    # Disegna la traccia della palla su output_video_frames
    trace = 1  # oppure un valore maggiore se vuoi una scia
    for num in range(len(output_video_frames)):
        for i in range(trace):
            if (num-i > 0) and ball_track[num-i][0]:
                x = int(ball_track[num-i][0])
                y = int(ball_track[num-i][1])
                cv2.circle(output_video_frames[num], (x,y), radius=0, color=(0, 0, 255), thickness=10-i)

    save_video(output_video_frames, "output_videos/output_" + suffix + ".avi")

if __name__ == "__main__":
    main()
    print('All good')