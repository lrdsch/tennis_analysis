import cv2
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from utils import save_video  # opzionale se vuoi salvare comunque in formato video
import json
import numpy as np

def convert_ndarray_to_list(obj):
    if isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def main():
    # Read single image
    input_image_path = "input_videos/image_last_frame.png"
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError(f"Immagine non trovata in: {input_image_path}")
    
    video_frames = [image]  # simuliamo una lista di un solo frame

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
    
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=False, 
        stub_path='tracker_stubs/player_detections.pkl'
    )
    
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stub=False,
        stub_path='tracker_stubs/ball_tracker.pkl'
    )
    
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # Court Line Detector
    court_model_path = 'models/keypoints_model.pth'
    court_line_detector = CourtLineDetector(court_model_path)
    
    keypoints = court_line_detector.predict(image)
    court_keypoints_per_frame = [keypoints]
    
    # ✅ Salva i keypoints su file JSON
    keypoints_output_path = "output_videos/keypoints_coordinates.json"
    keypoints_serializable = convert_ndarray_to_list(keypoints)
    with open(keypoints_output_path, "w") as f:
        json.dump(keypoints_serializable, f, indent=4)
    print(f"Keypoints salvati in: {keypoints_output_path}")

    # Choose and filter players
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints_per_frame[0],
        player_detections
    )
    
    # Draw outputs
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = [
        court_line_detector.draw_keypoints(frame, keypoints)
        for frame, keypoints in zip(output_video_frames, court_keypoints_per_frame)
    ]

    # Draw frame number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 255), 2)
    
    # Save single frame as image
    output_image_path = "output_videos/output_image.png"
    cv2.imwrite(output_image_path, output_video_frames[0])
    print(f"Output salvato in: {output_image_path}")

    # (Opzionale) salva come video di 1 frame
    # save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
    print("All good")
