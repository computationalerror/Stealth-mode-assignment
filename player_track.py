from ultralytics import YOLO

# loading pre-trained model for player tracking
model = YOLO("ultralytics_YOLO_model.pt")

# 2. Path to your video file
video_path = "15sec_input_720p.mp4"

# changing hyper parameters of "botsort" tracker for efficient tracking and re-identification
tracker_config_path = "new_botsort.yaml" 

# Dictionary to store history of object IDs
# Key: Track ID, Value: List of frames it appeared in
object_id_history = {} 

# Dictionary to store last known position for visualization or further logic
# Key: Track ID, Value: (x_center, y_center, frame_number)
last_known_positions = {}

# Run tracking with stream=True for frame-by-frame processing
results_generator = model.track(
    source=video_path,
    conf=0.3,
    iou=0.5,
    tracker=tracker_config_path,
    persist=True,
    show=True,
    save=False,
    stream=True
)

frame_number = 0
for results_per_frame in results_generator:
    frame_number += 1
    
    current_results = results_per_frame

    boxes = current_results.boxes

    if boxes is not None and boxes.id is not None:
        track_ids = boxes.id.int().tolist() 
        
        # Get confidence for each track
        confs = boxes.conf.cpu().numpy()

        print(f"\n--- Frame {frame_number} ---")
        for i, track_id in enumerate(track_ids):
            conf = confs[i]

            print(f"  Track ID: {track_id}, Conf: {conf:.2f}")

            # Store history of object IDs
            if track_id not in object_id_history:
                object_id_history[track_id] = []
            object_id_history[track_id].append(frame_number)

    else:
        print(f"\n--- Frame {frame_number} (No tracked objects) ---")

print("\n--- Tracking Summary ---")
for track_id, frames_appeared in object_id_history.items():
    
    # checking gaps in frame numbers for a given ID if it is larger then 20 frames, then re-identification
    # is successful.
    major_gap = [] 
    for i in range(1, len(frames_appeared)):
        if frames_appeared[i] - frames_appeared[i-1] > 20:
            major_gap.append((frames_appeared[i-1], frames_appeared[i]))

    if major_gap:
        print(f"major gaps in tracking ID {track_id} (disappeared/reappeared): {major_gap}")