import streamlit as st
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import random

 

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

# --- Page Configuration ---
st.set_page_config(
    page_title="Gestures & Hands",
    page_icon="✋",
    layout="wide"
)

st.title("🖐️ MediaPipe Gesture Recognition")
st.markdown("Webcam gesture recognition using MediaPipe and Streamlit.")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
run_camera = st.sidebar.checkbox("Start Camera", value=False)
camera_index = st.sidebar.number_input("Camera Index", value=0, min_value=0, step=1)

with st.sidebar.expander("Visual Settings"):
    landmark_color_hex = st.color_picker("Landmark Color", "#FF0000")
    connection_color_hex = st.color_picker("Connection Color", "#FFFFFF")
    line_thickness = st.slider("Line Thickness", 1, 10, 2)
    
    landmark_color = hex_to_bgr(landmark_color_hex)
    connection_color = hex_to_bgr(connection_color_hex)

with st.sidebar.expander("Detection Settings"):
    ok_threshold = st.slider("OK Gesture Threshold", 0.01, 0.20, 0.05, 0.01)

# --- MediaPipe Setup ---
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode



# Manual simple connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

# Global variable for async result
recognition_result_list = []

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global recognition_result_list
    recognition_result_list = result



# --- Helper Functions (Ported from main.py) ---

def is_ok_gesture(landmarks, threshold=0.05):
    """
    Checks if the hand landmarks form an 'OK' gesture.
    """
    # Thumb tip (4) and Index finger tip (8)
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]

    # Calculate distance between thumb tip and index tip
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x)**2 +
        (thumb_tip.y - index_tip.y)**2 +
        (thumb_tip.z - index_tip.z)**2
    )

    # Threshold for "touching"
    if distance < threshold:
        wrist = landmarks[0]
        
        def get_dist_2d(p1, p2):
             return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

        # Check Middle
        if get_dist_2d(wrist, landmarks[12]) < get_dist_2d(wrist, landmarks[10]): return False
        # Check Ring
        if get_dist_2d(wrist, landmarks[16]) < get_dist_2d(wrist, landmarks[14]): return False
        # Check Pinky
        if get_dist_2d(wrist, landmarks[20]) < get_dist_2d(wrist, landmarks[18]): return False

        return True
    
    return False

def count_fingers(landmarks):
    # Modified to reuse the new function
    status = get_fingers_status(landmarks)
    return status.count(True)

def get_fingers_status(landmarks):
    """
    Returns a list of 5 booleans [Thumb, Index, Middle, Ring, Pinky]
    indicating if the finger is extended.
    """
    fingers = []
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    pinky_mcp = landmarks[17]
    
    def get_dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    if get_dist(thumb_tip, pinky_mcp) > get_dist(thumb_ip, pinky_mcp):
        fingers.append(True)
    else:
        fingers.append(False)
        
    # Other 4 fingers: Tip y < PIP y
    # Index (8) vs (6)
    fingers.append(landmarks[8].y < landmarks[6].y)
    # Middle (12) vs (10)
    fingers.append(landmarks[12].y < landmarks[10].y)
    # Ring (16) vs (14)
    fingers.append(landmarks[16].y < landmarks[14].y)
    # Pinky (20) vs (18)
    fingers.append(landmarks[20].y < landmarks[18].y)
    
    return fingers

def draw_hand_landmarks_cv2(image, hand_landmarks, connection_color, landmark_color, thickness):
    h, w, _ = image.shape
    for connection in HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        x1, y1 = int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h)
        x2, y2 = int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h)
        cv2.line(image, (x1, y1), (x2, y2), connection_color, thickness) 
    
    for landmark in hand_landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (x, y), thickness * 2, landmark_color, -1) 
        cv2.circle(image, (x, y), thickness * 2, (255, 255, 255), 1)

# --- Main App Logic ---

frame_placeholder = st.empty()
status_text = st.sidebar.empty()

# Mode Selection
app_mode = st.sidebar.selectbox("Choose Mode", ["Air Canvas", "Rock Paper Scissors"])

# Add Canvas Controls
if app_mode == "Air Canvas":
    with st.sidebar.expander("Air Canvas Settings", expanded=True):
        draw_color_hex = st.color_picker("Pen Color", "#00FF00")
        draw_color = hex_to_bgr(draw_color_hex)
        canvas_clear = st.button("Clear Canvas")

# RPS Controls
rps_state = {"status": "waiting", "start_time": 0, "result_text": ""}
if app_mode == "Rock Paper Scissors":
    start_game_btn = st.sidebar.button("Start Round")

if run_camera:
    # Initialize Gesture Recognizer
    try:
        # Load model with options
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=print_result,
            num_hands=2
        )
        
        recognizer = GestureRecognizer.create_from_options(options)
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            st.error(f"Cannot access camera with index {camera_index}.")
            run_camera = False
        else:
            status_text.success("Camera Running...")
            
            # Canvas Variables
            canvas_image = None
            prev_x, prev_y = 0, 0
            
            # Game Variables
            game_active = False
            game_start_time = 0
            game_result_display_time = 0
            cpu_move = ""
            user_move = ""
            winner = ""

            while cap.isOpened() and run_camera:
                # Check for start button click
                # In Streamlit, buttons return True only on the run they are clicked.
                # If the user clicks, Streamlit BREAKS the loop and reruns the script.
                # So when we re-enter, start_game_btn is True.
                if app_mode == "Rock Paper Scissors" and 'start_game_btn' in locals() and start_game_btn and not game_active:
                     game_active = True
                     game_start_time = time.time()
                     game_result_display_time = 0
                     # Reset button state hack not needed as loop restarts on interaction usually

                success, frame = cap.read()
                if not success:
                    st.warning("Failed to read frame.")
                    break
                
                # Flip and convert
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if app_mode == "Air Canvas":
                    # Initialize canvas if needed
                    if canvas_image is None or ('canvas_clear' in locals() and canvas_clear):
                        canvas_image = np.zeros_like(frame)
                        canvas_clear = False # Reset flag

                # MediaPipe Image
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Timestamp
                timestamp = int(time.time() * 1000)
                
                # Async recognize
                recognizer.recognize_async(mp_image, timestamp)
                
                # Draw results on the frame
                if recognition_result_list:
                    # Check if it has 'hand_landmarks'
                    if hasattr(recognition_result_list, 'hand_landmarks'):
                        for i, hand_landmarks in enumerate(recognition_result_list.hand_landmarks):
                            
                            # Custom Drawing
                            draw_hand_landmarks_cv2(frame, hand_landmarks, connection_color, landmark_color, line_thickness)
                            
                            # Gesture Info (Common)
                            if hasattr(recognition_result_list, 'gestures') and len(recognition_result_list.gestures) > i:
                                gesture = recognition_result_list.gestures[i][0]
                                category_name = gesture.category_name
                                score = gesture.score
                                
                                # --- MODE: Air Canvas ---
                                if app_mode == "Air Canvas":
                                    fingers = get_fingers_status(hand_landmarks)
                                    ix, iy = int(hand_landmarks[8].x * frame.shape[1]), int(hand_landmarks[8].y * frame.shape[0])
                                    
                                    # 1. Draw Mode: Only Index is up
                                    if fingers[1] and not any(fingers[2:]):
                                        cv2.circle(frame, (ix, iy), 15, draw_color, -1)
                                        if prev_x == 0 and prev_y == 0:
                                            prev_x, prev_y = ix, iy
                                        cv2.line(canvas_image, (prev_x, prev_y), (ix, iy), draw_color, line_thickness + 5)
                                        prev_x, prev_y = ix, iy
                                        
                                    # 2. Erase Mode: All fingers up
                                    elif all(fingers):
                                        palm_x, palm_y = int(hand_landmarks[9].x * frame.shape[1]), int(hand_landmarks[9].y * frame.shape[0])
                                        cv2.circle(frame, (palm_x, palm_y), 40, (0, 0, 0), 2)
                                        cv2.circle(canvas_image, (palm_x, palm_y), 40, (0, 0, 0), -1)
                                        prev_x, prev_y = 0, 0 
                                    else:
                                        prev_x, prev_y = 0, 0

                                    # Regular Text Overlay
                                    if is_ok_gesture(hand_landmarks, ok_threshold):
                                        category_name = "OK"
                                        score = 1.00
                                    finger_count = count_fingers(hand_landmarks)
                                    text = f"{category_name} ({score:.2f}) | Fingers: {finger_count}"
                                    wrist_x = int(hand_landmarks[0].x * frame.shape[1])
                                    wrist_y = int(hand_landmarks[0].y * frame.shape[0]) - 20
                                    cv2.putText(frame, text, (wrist_x, wrist_y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                                # --- MODE: Rock Paper Scissors ---
                                elif app_mode == "Rock Paper Scissors":
                                    # Just display current gesture for feedback
                                    wrist_x = int(hand_landmarks[0].x * frame.shape[1])
                                    wrist_y = int(hand_landmarks[0].y * frame.shape[0]) - 20
                                    cv2.putText(frame, category_name, (wrist_x, wrist_y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                # --- Game Logic Outside Landmark Loop (Draws text even if no hand detected) ---
                if app_mode == "Rock Paper Scissors" and game_active:
                    elapsed = time.time() - game_start_time
                    
                    if elapsed < 3:
                        # Countdown
                        countdown_text = str(3 - int(elapsed))
                        cv2.putText(frame, countdown_text, (frame.shape[1]//2 - 50, frame.shape[0]//2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5)
                    elif elapsed < 4:
                        # CAPTURE MOMENT (approx at 3.0s)
                        if elapsed < 3.2 and winner == "": # Only calculate once
                            # Determine user move from *last known result*
                            # Note: This uses the frame's detection.
                            current_gesture = "Unknown"
                            if recognition_result_list and recognition_result_list.gestures:
                                current_gesture = recognition_result_list.gestures[0][0].category_name
                            
                            # Map to RPS
                            if current_gesture == "Closed_Fist": user_move = "Rock"
                            elif current_gesture == "Open_Palm": user_move = "Paper"
                            elif current_gesture == "Victory": user_move = "Scissors"
                            else: user_move = "Unknown"
                            
                            # CPU Move
                            cpu_move = random.choice(["Rock", "Paper", "Scissors"])
                            
                            # Determine Winner
                            if user_move == "Unknown":  
                                winner = "No Move"
                            elif user_move == cpu_move:
                                winner = "Draw"
                            elif (user_move == "Rock" and cpu_move == "Scissors") or \
                                 (user_move == "Paper" and cpu_move == "Rock") or \
                                 (user_move == "Scissors" and cpu_move == "Paper"):
                                 winner = "You Win!"
                            else:
                                winner = "CPU Wins!"
                            
                            game_result_display_time = time.time()

                    # Display Result (for 5 seconds after capture)
                    if winner != "":
                         # Text
                         cv2.putText(frame, f"You: {user_move}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                         cv2.putText(frame, f"CPU: {cpu_move}", (frame.shape[1]-250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                         
                         color = (0, 255, 0) if "Win" in winner else (0, 0, 255)
                         if winner == "Draw": color = (255, 255, 0)
                         
                         cv2.putText(frame, winner, (frame.shape[1]//2 - 150, frame.shape[0]//2),
                                     cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                         
                         # Stop game after 5s result display
                         if time.time() - game_result_display_time > 5 and game_result_display_time > 0:
                             game_active = False
                             winner = ""


                if app_mode == "Air Canvas":
                    # Blend Canvas with Frame
                    if canvas_image is not None:
                        gray_canvas = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2GRAY)
                        _, inverse_mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
                        inverse_mask = cv2.cvtColor(inverse_mask, cv2.COLOR_GRAY2BGR)
                        frame = cv2.bitwise_and(frame, inverse_mask)
                        frame = cv2.bitwise_or(frame, canvas_image)

                # Display in Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")
                
                # Small sleep to be nice to CPU?
                # time.sleep(0.01) 
                # Streamlit loops are often CPU intensive. 
                
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'recognizer' in locals():
            recognizer.close()
        status_text.info("Camera Stopped.")

else:
    status_text.info("Check 'Start Camera' in sidebar to begin.")
    frame_placeholder.markdown(
        """
        <div style="padding: 20px; border: 1px dashed #ccc; text-align: center;">
            Camera is stopped.
        </div>
        """,
        unsafe_allow_html=True
    )