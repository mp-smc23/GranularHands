import mediapipe as mp
import cv2
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from pythonosc import udp_client

FONT_SIZE = 0.3
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 0) 

RESET_CATEGORY = "Closed_Fist"

class HandDetection:
  def __init__(self, ip, port) -> None:
      self.cap = cv2.VideoCapture(0)
      self.detector = self.init_detector()

      self.starting_positions = None
      self.starting_rotations = None
      self.starting_distance = None

      self.y_offset_left = 0
      self.y_offset_right = 0
      self.left_right_distance = 0
      self.left_angle = 0
      self.right_angle = 0

      self.client = udp_client.SimpleUDPClient(ip, port)


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

        self.read_resetting_position(detection_result)
        self.calculate_hand_positions(detection_result)
        self.send_parameters_osc()

        annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        flipped_video = cv2.flip(annotated_image, 1)
        flipped_video = self.draw_parameters_on_image(flipped_video)

        cv2.imshow("Video", flipped_video)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
          return

  @staticmethod
  def get_hands_indices(handedness):
    if handedness[0][0].category_name == "Left":
      return handedness[0][0].index, 1 - handedness[0][0].index
    return 1 - handedness[0][0].index, handedness[0][0].index

  @staticmethod
  def get_angle_of_hand(hand_landmarks):
    dx = hand_landmarks[9].x - hand_landmarks[0].x
    dy = hand_landmarks[9].y - hand_landmarks[0].y
    return np.arctan2(dx, dy) # shift 90 degrees (so that 0 is pointing up)

  # Check if the hand is in the starting position which is two fists
  def read_resetting_position(self, detection_result):
    hand_landmarks = detection_result.hand_landmarks
    if len(hand_landmarks) != 2:
      return
    
    handedness = detection_result.handedness
    gestures = detection_result.gestures

    if gestures[0][0].category_name == RESET_CATEGORY and gestures[1][0].category_name == RESET_CATEGORY:
      print("Resetting position detected")
      left_idx, right_idx = self.get_hands_indices(handedness)
      self.starting_positions = {"Left":hand_landmarks[left_idx][0],
                                 "Right":hand_landmarks[right_idx][0]}

      self.starting_rotations = {"Left": self.get_angle_of_hand(hand_landmarks[left_idx]),
                                 "Right": self.get_angle_of_hand(hand_landmarks[right_idx])}
      
      self.starting_distance = np.abs(hand_landmarks[left_idx][0].x - hand_landmarks[right_idx][0].x)
      

  def calculate_hand_positions(self, detection_result):
    if self.starting_positions is None or len(detection_result.hand_landmarks) < 2:
      return
    
    left_idx, right_idx = self.get_hands_indices(detection_result.handedness)
    hand_landmarks = detection_result.hand_landmarks

    # calculate the new values
    new_y_offset_left = np.abs(hand_landmarks[left_idx][0].y - self.starting_positions["Left"].y)
    new_y_offset_right = np.abs(hand_landmarks[right_idx][0].y - self.starting_positions["Right"].y)
    new_left_right_distance = np.abs(np.abs(hand_landmarks[left_idx][0].x - hand_landmarks[right_idx][0].x) - self.starting_distance)
    new_left_angle = self.get_angle_of_hand(hand_landmarks[left_idx])# - self.starting_rotations["Left"]
    new_right_angle = self.get_angle_of_hand(hand_landmarks[right_idx])# - self.starting_rotations["Right"]

    # apply low pass filter to smooth the values
    self.y_offset_left = new_y_offset_left * 0.5 + self.y_offset_left * 0.5
    self.y_offset_right = new_y_offset_right * 0.5 + self.y_offset_right * 0.5
    self.left_right_distance = new_left_right_distance * 0.5 + self.left_right_distance * 0.5
    self.left_angle = new_left_angle * 0.5 + self.left_angle * 0.5
    self.right_angle = new_right_angle * 0.5 + self.right_angle * 0.5

  def send_parameters_osc(self):
    self.client.send_message("/y_offset_left", self.y_offset_left)
    self.client.send_message("/y_offset_right", self.y_offset_right)
    self.client.send_message("/left_right_distance", self.left_right_distance)
    self.client.send_message("/left_angle", self.left_angle)
    self.client.send_message("/right_angle", self.right_angle)

  def draw_parameters_on_image(self, image):
    height, width = image.shape[1] // 10, image.shape[0] // 3.5
    cv2.rectangle(image, (0, 0), (int(width), int(height * 1.2)), (255, 255, 255), -1)

    cv2.putText(image, f"Y offset left: {self.y_offset_left:.2f}", (5, int(height * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    cv2.putText(image, f"Y offset right: {self.y_offset_right:.2f}", (5, int(height * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    cv2.putText(image, f"Left right distance: {self.left_right_distance:.2f}", (5, int(height * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    cv2.putText(image, f"Left angle: {self.left_angle:.2f}", (5, int(height * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    cv2.putText(image, f"Right angle: {self.right_angle:.2f}", (5, height), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image

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