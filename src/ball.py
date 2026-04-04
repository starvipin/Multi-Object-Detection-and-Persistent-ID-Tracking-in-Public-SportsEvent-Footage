import cv2
from ultralytics import YOLO

# YOLOv8 ka pre-trained nano model load kar rahe hain (Speed ke liye)
# Accuracy badhani ho toh 'yolov8m.pt' ya 'yolov8l.pt' use kar sakte hain
model = YOLO('yolov8n.pt') 

input_video_path = '../videos/fottable.mp4'  # Yahan apni video ka naam dalein
output_video_path = '../process_videos/fottable_ball_track.mp4'

# Video capture setup
cap = cv2.VideoCapture(input_video_path)

# Output video writer setup
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print("Video processing start ho rahi hai... Kripya wait karein.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLOv8 ka built-in tracker use kar rahe hain
    # classes=[32] ka matlab hai hum sirf 'sports ball' ko target kar rahe hain
    # persist=True ensures ki ball ko frames ke across same ID mile
    results = model.track(frame, persist=True, classes=[32], tracker="botsort.yaml", verbose=False)

    # Frame par bounding box aur label draw karna
    annotated_frame = results[0].plot()

    # Processed frame ko output video mein write karna
    out.write(annotated_frame)

    # Agar processing ke dauran output dekhna hai (Optional)
    # cv2.imshow("Football Tracker", annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

# Resources release karna
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete! Output video yahan save ho gayi hai: {output_video_path}")