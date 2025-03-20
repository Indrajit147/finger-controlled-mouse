import cv2
import mediapipe as mp
import pyautogui
pyautogui.FAILSAFE = False
# Initialize the webcam
cam = cv2.VideoCapture(0)

# Initialize the hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)  # Track up to 2 hands
mp_drawing = mp.solutions.drawing_utils

# Get the screen size
screen_w, screen_h = pyautogui.size()

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    output = hands.process(rgb_frame)
    landmark_points = output.multi_hand_landmarks

    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        for hand_landmarks, handedness in zip(landmark_points, output.multi_handedness):
            # Get the hand label (left or right)
            hand_label = handedness.classification[0].label

            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Right hand: Move the cursor
            if hand_label == "Right":
                # Get the landmarks for the index finger
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger.x * frame_w)
                y = int(index_finger.y * frame_h)

                # Map the finger position to the screen
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y
                pyautogui.moveTo(screen_x, screen_y)

            # Left hand: Detect click gesture
            elif hand_label == "Left":
                # Get the landmarks for the thumb and index finger
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate the distance between the thumb and index finger
                thumb_x = int(thumb.x * frame_w)
                thumb_y = int(thumb.y * frame_h)
                index_x = int(index_finger.x * frame_w)
                index_y = int(index_finger.y * frame_h)

                distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5

                # If the distance is small, consider it a click
                if distance < 20:
                    pyautogui.click()
                    pyautogui.sleep(0.5)  # Add a small delay to avoid multiple clicks

    # Display the frame
    cv2.imshow('Hand Controlled Mouse', frame)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Release the webcam and close the window
cam.release()
cv2.destroyAllWindows()