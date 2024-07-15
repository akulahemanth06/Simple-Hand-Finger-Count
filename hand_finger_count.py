import cv2
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define the new window size
new_width = 1280
new_height = 720

# Function to count raised fingers
def count_fingers(hand_landmarks, handedness):
    # These are the finger tip landmarks we need
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4

    # Count fingers
    count = 0

    # Check for each finger if it is raised
    for tip in finger_tips:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            count += 1

    # Check for thumb
    if handedness == 'Right':
        if hand_landmarks[thumb_tip].x < hand_landmarks[thumb_tip - 2].x:
            count += 1
    else:
        if hand_landmarks[thumb_tip].x > hand_landmarks[thumb_tip - 2].x:
            count += 1

    return count

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        total_finger_count = 0
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks for counting fingers
            landmarks = hand_landmarks.landmark

            # Determine handedness
            hand_label = handedness.classification[0].label

            # Count fingers
            finger_count = count_fingers(landmarks, hand_label)
            total_finger_count += finger_count

        # Display total finger count
        cv2.putText(frame, f'Total Fingers: {total_finger_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize the frame
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # Display the resized frame
    cv2.imshow('Finger Counting', frame_resized)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
