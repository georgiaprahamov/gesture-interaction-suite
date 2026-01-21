# 🖐️ AI Gesture Studio

A Streamlit-based computer vision application allowing you to interact with your webcam using hand gestures. Features include real-time gesture recognition, drawing in the air, and playing Rock-Paper-Scissors against the computer.

## Features

### 1. 🖖 Gesture Recognition
*   Real-time hand detection using [MediaPipe](https://developers.google.com/mediapipe).
*   Recognizes gestures like "Closed Fist", "Open Palm", "Victory", "Thumb Up", etc.
*   Special detection for the "OK" sign.
*   Counts extended fingers.

### 2. 🎨 Air Canvas
Draw on the screen simply by moving your finger!
*   **Draw**: Raise only your **Index Finger**.
*   **Erase**: Raise **All Fingers** (Open Palm) to clear.
*   **Hover**: Any other gesture moves the cursor without drawing.
*   Customize pen color and line thickness in the sidebar.

### 3. ✂️ Rock, Paper, Scissors
Play the classic game against an AI opponent.
*   **How to play**:
    1.  Select "Rock Paper Scissors" from the mode dropdown.
    2.  Click "Start Round".
    3.  Wait for the countdown (3... 2... 1...).
    4.  Show your move (Fist=Rock, Palm=Paper, Victory=Scissors).
    5.  See who wins!

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/ai-gesture-studio.git
    cd ai-gesture-studio
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Model:**
    Ensure `gesture_recognizer.task` is in the root directory. If missing, download it from [MediaPipe Tasks](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer).

## Usage

Run the application using Streamlit:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to use the app.

## Customization

Use the sidebar to:
*   Change **Landmark** and **Connection** colors.
*   Adjust **Line Thickness**.
*   Tune the **Sensitivity** for specific gestures.

## Dependencies
*   Python 3.8+
*   Streamlit
*   MediaPipe
*   OpenCV
*   NumPy
