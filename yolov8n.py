from ultralytics import YOLO
import cv2
import math
import speech_recognition as sr
import pyttsx3
import threading
import os

# Initialize YOLO model
model = YOLO(os.path.join("object_detection", "model", "yolov8n.pt"))

# Object data and focal length
data = {
    "cell phone": 0.15,
    "person": 1.8,
    "bed": 0.8,
    "laptop": 0.25,
    "chair": 1.0
}
FOCAL_LENGTH = 800

# Global variables
selected_object = None
tracker = None 
selected_object_bbox = None

# Text-to-speech for user feedback
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Voice command input
def get_voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak_text("Please say the object you want to detect.")
        print("Listening for object name...")
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"Recognized command: {command}")
            return command
        except sr.UnknownValueError:
            speak_text("Sorry, I didn't understand that.")
        except sr.RequestError:
            speak_text("There was an issue with the speech recognition service.")
        except sr.WaitTimeoutError:
            speak_text("Listening timed out. Please try again.")
    return None

# Distance estimation function
def estimate_distance(real_height, focal_length, object_height_in_pixels):
    if object_height_in_pixels == 0:
        return None
    return (real_height * focal_length) / object_height_in_pixels

# Real-world distance calculation
def calculate_real_world_distance(center1, distance1, center2, distance2):
    pixel_distance = abs(center1[0] - center2[0])
    horizontal_distance = (pixel_distance * distance1) / FOCAL_LENGTH
    return math.sqrt(horizontal_distance*2 + (distance1 - distance2)*2)

# Calculate direction to turn
def calculate_direction(person_center, target_center):
    # Compare x-coordinates for left/right direction
    if target_center[0] < person_center[0] - 20:  # Add tolerance for small movements
        return "left"
    elif target_center[0] > person_center[0] + 20:
        return "right"
    else:
        return "straight"

# Process video frames with object tracking, distance, and direction monitoring
def process_video():
    global selected_object, tracker, selected_object_bbox
    cap = cv2.VideoCapture("WhatsApp Video 2024-11-20 at 20.16.32_0dadccef.mp4")
    frame_count = 0
    object_data = []
    person_data = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Reset object data for the current frame
        object_data = []
        results = model(frame, conf=0.6)

        # Initialize or update tracker
        if selected_object_bbox and tracker:
            success, bbox = tracker.update(frame)
            if success:
                x1, y1, w, h = map(int, bbox)
                selected_object_bbox = (x1, y1, x1 + w, y1 + h)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                conf = box.conf[0]

                if class_id >= len(model.names):
                    continue

                object_name = model.names[class_id]
                object_height_in_pixels = y2 - y1
                bbox = (x1, y1, x2 - x1, y2 - y1)

                if object_name in data:
                    real_height = data[object_name]
                    distance = estimate_distance(real_height, FOCAL_LENGTH, object_height_in_pixels)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    if object_name == "person":
                        person_data = (center, distance)
                    elif object_name == selected_object:
                        object_data.append((object_name, center, distance))
                        if selected_object_bbox is None:
                            tracker = cv2.TrackerCSRT_create()
                            tracker.init(frame, bbox)
                            selected_object_bbox = bbox

                    if distance is not None:
                        cv2.putText(frame, f"Distance: {distance:.2f} m", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    label = f"{object_name} {int(conf * 100)}%"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if person_data and object_data:
            for obj_name, obj_center, obj_distance in object_data:
                person_center, person_distance = person_data
                real_distance = calculate_real_world_distance(person_center, person_distance, obj_center, obj_distance)
                direction = calculate_direction(person_center, obj_center)

                cv2.line(frame, person_center, obj_center, (0, 0, 255), 2)
                cv2.putText(frame, f"{obj_name}: {real_distance:.2f} m, {direction}",
                            (obj_center[0] + 10, obj_center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Threaded voice feedback for real-time synchronization
                threading.Thread(target=speak_text, args=(f"The {obj_name} is {real_distance:.2f} meters away. Move {direction}.",)).start()

        # Display instructions on the video feed
        cv2.putText(frame, "Press 'q' to quit, 's' to save frame", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("YOLO Object Detection with Real-World Distance and Direction", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('s'):  # Save frame
            cv2.imwrite(f"saved_frame_{frame_count}.jpg", frame)
            print(f"Frame {frame_count} saved.")
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "_main_":
    while not selected_object:
        command = get_voice_command()
        if command in data.keys():
            selected_object = command
            speak_text(f"You selected {selected_object}.")
        else:
            speak_text(f"Invalid object. Available objects are: {', '.join(data.keys())}. Please try again.")

    process_video()