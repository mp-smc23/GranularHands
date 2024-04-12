import mediapipe as mp
import cv2
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

RESET_CATEGORY = "Closed_Fist"

class HandDetection:
  def __init__(self) -> None:
      self.cap = cv2.VideoCapture(0)
      self.cap.set(cv2.CAP_PROP_FPS, 5)
      self.detector = self.init_detector()

      self.starting_position = None

  def __del__(self) -> str:
     self.cap.release()
     pass

  def init_detector(self):
    base_options =  python.BaseOptions(model_asset_path='model/gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options, num_hands=2)
    return vision.GestureRecognizer.create_from_options(options)

  def run(self):
    while True: 
        ret, frame = self.cap.read() 

        if not ret: 
            self.cap.release()
            cv2.destroyAllWindows()
            return

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detection_result = self.detector.recognize(mp_image)
        annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

        self.read_resetting_position(detection_result)

        cv2.imshow("Video", cv2.flip(annotated_image, 1) )

        if cv2.waitKey(1) & 0xFF == ord('q'): 
          return

  def read_resetting_position(self, detection_result):
    # Check if the hand is in the starting position which is two fists
    hand_landmarks_list = detection_result.hand_landmarks
    
    if len(hand_landmarks_list) != 2:
      return
    
    handedness = detection_result.handedness
    gestures = detection_result.gestures

    if(gestures[0][0].category_name == RESET_CATEGORY and gestures[1][0].category_name == RESET_CATEGORY):
      print("Resetting position detected")

      left_idx = handedness[0][0].index if handedness[0][0].category_name == "Left" else handedness[1][0].index
      right_idx = int(not left_idx)

      # print(str(hand_landmarks_list[0][0]) + str(left_idx))
      # print(str(hand_landmarks_list[1][0]) + str(right_idx))
      self.starting_position = (hand_landmarks_list[left_idx][0], hand_landmarks_list[right_idx][0])
  
  
  @staticmethod
  def calculate_hand_positions(detection_result):
     pass

  @staticmethod
  def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
      hand_landmarks = hand_landmarks_list[idx]

      # Draw the hand landmarks.
      hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
      ])
      mp.solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style())

    return annotated_image