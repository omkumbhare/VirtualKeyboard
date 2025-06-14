
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
from pynput.keyboard import Controller, Key

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 520)  # Height
cv2.namedWindow("Virtual Keyboard", cv2.WINDOW_NORMAL)

detector = HandDetector(detectionCon=0.8, minTrackCon=0.2)

keyboard_keys = [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", "?"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
    ["SPACE", "ENTER", "BACKSPACE"]
]

keyboard = Controller()

class Button:
    def __init__(self, pos, text, size=(85, 85)):
        self.pos = pos
        self.size = size
        self.text = text

def draw_buttons(img, button_list):
    """
    Draws buttons on the given image.

    Args:
        img (numpy.ndarray): The image on which the buttons will be drawn.
        button_list (list): A list of Button objects representing the buttons to be drawn.

    Returns:
        numpy.ndarray: The image with the buttons drawn.
    """
    overlay = img.copy()
    alpha = 0.6  #keytransperancy

    for button in button_list:
        x, y = button.pos
        w, h = button.size
        # Draw a rectangle with a translucent white color
        cv2.rectangle(overlay, button.pos, (int(x + w), int(y + h)),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(overlay, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)

    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img

# Create Button objects based on keyboard_keys layout
button_list = []
for k in range(len(keyboard_keys)):
    for x, key in enumerate(keyboard_keys[k]):
        if key not in ["SPACE", "ENTER", "BACKSPACE"]:
            button_list.append(Button((100 * x + 25, 100 * k + 50), key))
        elif key == "ENTER":
            button_list.append(Button((100 * x - 30, 100 * k + 50), key, (220, 85)))
        elif key == "SPACE":
            button_list.append(Button((100 * x + 780, 100 * k + 50), key, (220, 85)))
        elif key == "BACKSPACE":
            button_list.append(Button((100 * x + 140, 100 * k + 50), key, (400, 85)))

while True:
    success, img = cap.read()  # Read frame from camera
    img = cv2.flip(img, 1)  # Invert the camera feed horizontally
    allHands, img = detector.findHands(img)  # Find hands in the frame

    if len(allHands) == 0:
        lm_list, bbox_info = [], []
    else:
        # Find landmarks and bounding box info
        lm_list, bbox_info = allHands[0]['lmList'], allHands[0]['bbox']

    img = draw_buttons(img, button_list)  # Draw buttons on the frame

    # Check if landmarks (lmList) are detected
    if lm_list:
        for button in button_list:
            x, y = button.pos
            w, h = button.size

            # Check if index finger (lmList[8]) is within the button bounds
            if x < lm_list[8][0] < x + w and y < lm_list[8][1] < y + h:
                overlay = img.copy()
                cv2.rectangle(overlay, button.pos, (x + w, y + h),
                              (247, 45, 134), cv2.FILLED)  # Highlight the button on hover
                cv2.putText(overlay, button.text, (x + 20, y + 65),
                            cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                img = cv2.addWeighted(overlay, 0.6, img, 1 - 0.6, 0)

                # Calculate distance between thumb (lmList[4]) and index finger (lmList[8])
                distance = np.sqrt((lm_list[8][0] - lm_list[4][0])**2 + (lm_list[8][1] - lm_list[4][1])**2)

                # If distance is small, simulate key press
                if distance < 30:
                    # Check for special keys
                    if button.text not in ['ENTER', "BACKSPACE", "SPACE"]:
                        keyboard.press(button.text)  # Press the key
                        sleep(0.9)  # Small delay for better usability & prevent accidental key presses
                    else:
                        if button.text == "SPACE":
                            keyboard.press(Key.space)
                            keyboard.release(Key.space)
                            sleep(0.9)
                        elif button.text == "ENTER":
                            keyboard.press(Key.enter)
                            keyboard.release(Key.enter)
                            sleep(0.9)
                        elif button.text == "BACKSPACE":
                            keyboard.press(Key.backspace)
                            keyboard.release(Key.backspace)
                            sleep(0.9)
                        else:
                            pass

                    overlay = img.copy()
                    cv2.rectangle(overlay, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(overlay, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                    img = cv2.addWeighted(overlay, 0.6, img, 1 - 0.6, 0)

    # Display the frame with virtual keyboard
    cv2.imshow("Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == 27:  #Exit Loop
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
