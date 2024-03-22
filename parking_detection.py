import cv2

def detect_occupancy(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bg_subtractor.apply(frame)

        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        occupied_pixels = cv2.countNonZero(thresh)

        occupancy_threshold = 5000  
        if occupied_pixels > occupancy_threshold:
            status = "Occupied"
        else:
            status = "Empty"

        cv2.putText(frame, f"Parking Status: {status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Parking Lot', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the detection on a sample video
detect_occupancy('')
