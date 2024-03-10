import numpy as np
import cv2
import time
import random
import imutils
from cvzone.HandTrackingModule import HandDetector
import time



# Initialize hand detector
detector = HandDetector(detectionCon=0.5, maxHands=2)

# Initialize variables
startGame = False
scores = [0, 0]
initial_time = None
chosen_hand = None

# Initialize video capture
cap = cv2.VideoCapture(0)


# Function to detect hand
def detect_hand(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find contours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if len(contours) > 0:
        # Find the largest contour (the hand)
        hand_contour = max(contours, key=cv2.contourArea)
        return hand_contour
    else:
        return None

def detect_ball_in_hand(hand_region):
    blurred = cv2.GaussianBlur(hand_region, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    greenLower, greenUpper = (20, 100, 100), (30, 255, 255)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(hand_region, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(hand_region, center, 5, (0, 0, 255), -1)
            return True
        else:
            return False

# Function to start the game
def start_game():
    global startGame, initial_time
    startGame = True
    initial_time = time.time()


# Function to reset game variables
def reset_game():
    global startGame, initial_time, chosen_hand
    startGame = False
    initial_time = None
    chosen_hand = None


# Main loop
while True:
    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Detect hands in the frame
    hands, _ = detector.findHands(frame)

    # If two hands are detected, start the game
    if len(hands) == 2 and not startGame:
        start_game()

    # If the game is ongoing
    if startGame:
        # Calculate timer
        timer = int(5 - (time.time() - initial_time))

        # If timer reaches 0, choose a random hand and wait for the user to open it
        if timer == 0 and chosen_hand is None:
            chosen_hand = random.choice(["Left", "Right"])
            print(f"Chosen Hand: {chosen_hand}")

        # Detect open hand in the chosen hand
        for hand in hands:
            # Get hand landmarks
            lmList = hand["lmList"]
            if lmList:
                # Calculate the angles between adjacent fingers
                angles = []
                for i in range(1, 5):
                    x1, y1 = lmList[i * 4][1:]
                    x2, y2 = lmList[i * 4 - 2][1:]
                    x3, y3 = lmList[i * 4 - 1][1:]
                    angle = np.degrees(np.arctan2(y3 - y1, x3 - x1) - np.arctan2(y2 - y1, x2 - x1))
                    angle = np.abs(angle)
                    angle = angle if angle <= 180 else 360 - angle
                    angles.append(angle)

                # Count the number of fingers that have an angle larger than a lower threshold
                extended_fingers = sum(1 for angle in angles if angle > 20)  # Adjust threshold as needed

                if extended_fingers >= 3:
                    text = "Closed"
                else:
                    text = "Open"
                    # Get the bounding box of the hand
                bbox = hand["bbox"]
                # Calculate the top-left corner of the bounding box
                x, y = bbox[0], bbox[1]
                # Draw the text dynamically at the top of the bounding box
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0) if text == "Open" else (0, 0, 255), 2)

                # If the chosen hand is open, check if a ball is detected in it
                if text == "Open" and hand["type"] == chosen_hand:
                    bbox = hand["bbox"]  # Extract bounding box of the hand
                    hand_region = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]  # Extract hand region from frame
                    ball_detected = detect_ball_in_hand(hand_region)
                    if ball_detected:
                        scores[0] += 1
                        print("Ball not detected in chosen hand. Score for User.")
                    else:
                        scores[1] += 1
                        print("Ball detected in chosen hand. Score for Computer.")
                    reset_game()

    # Display scores and timer on the frame
    cv2.putText(frame, f"Player: {scores[0]}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Computer: {scores[1]}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Chosen Hand: {chosen_hand}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Timer: {timer}" if startGame else "Waiting for Hands", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
