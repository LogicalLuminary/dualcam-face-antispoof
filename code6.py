import cv2
import numpy as np
import face_recognition
from scipy.signal import butter, lfilter, find_peaks
import time, threading, os, requests
from collections import deque

# ---------------- SETTINGS ----------------
FPS = 30
WINDOW_SIZE = 5
UPDATE_INTERVAL = 1
SMOOTHING_WINDOW = 3
RECOGNITION_PERIOD_S = 0.33  
ESP_COOLDOWN = 10
YAW_THRESHOLD = 30
MAX_SKEW_S = 10
SHOW_DEBUG = True

# hybrid detection tuning
ABS_YAW_MIN = 45
DIFF_YAW_MIN = 12
BPM_GRACE_S = 3.0

# ---------------- ESP CONFIG -------------
ESP_IP = "192.168.183.219"
TOKEN = "mytoken123"

def trigger_gpio(pin_state=1):
    try:
        r = requests.get(f"http://{ESP_IP}/gpio",
                         params={"state": str(pin_state), "token": TOKEN},
                         timeout=1.5)
        print(f"[ESP] GPIO14={pin_state} status={r.status_code}")
    except requests.exceptions.RequestException as e:
        print("[ESP] Error:", e)

# ---------------- CAM URLS ----------------
cam1_url = "http://192.168.183.20:5000/video_feed"
cam2_url = "http://192.168.183.244:5000/video_feed"

# ---------------- UTILITIES ---------------
def butter_bandpass_filter(data, lowcut=0.7, highcut=3.0, fs=30, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return lfilter(b, a, data)

def estimate_bpm(signal, fps):
    filtered = butter_bandpass_filter(signal, 0.8, 3.0, fps)
    peaks, _ = find_peaks(filtered, distance=fps/2)
    if len(peaks) > 1:
        ibi = np.diff(peaks) / fps
        return 60 / np.mean(ibi)
    return None

def get_face_yaw(shape):
    left_point, right_point, nose_point = shape[1], shape[15], shape[30]
    d_left  = np.linalg.norm(nose_point - left_point)
    d_right = np.linalg.norm(nose_point - right_point)
    ratio = (d_right - d_left) / (d_left + d_right + 1e-6)
    return np.clip(ratio * 90, -90, 90)

# ---------------- LATEST FRAME READER -----
class LatestFrameReader:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.ts = 0.0
        self.stopped = False
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            now = time.perf_counter()
            with self.lock:
                self.ret = True
                self.frame = frame
                self.ts = now

    def read_latest(self):
        with self.lock:
            return self.ret, (None if self.frame is None else self.frame.copy()), self.ts

    def release(self):
        self.stopped = True
        try:
            self.t.join(timeout=0.5)
        except:
            pass
        self.cap.release()

# ---------------- MODELS & DATA -----------
cv2.setUseOptimized(True)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")

known_faces, known_names = [], []
faces_dir = "face"
for file in os.listdir(faces_dir):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img = face_recognition.load_image_file(os.path.join(faces_dir, file))
        enc = face_recognition.face_encodings(img)
        if len(enc) > 0:
            known_faces.append(enc[0])
            known_names.append(os.path.splitext(file)[0])
print(f"✅ Loaded {len(known_faces)} known faces.")

# ---------------- STATE -------------------
bpm_values = deque(maxlen=SMOOTHING_WINDOW)
yaw_values_cam1 = deque(maxlen=5)
yaw_values_cam2 = deque(maxlen=5)
frame_buffer = []

prev_landmarks_cam1 = None
prev_landmarks_cam2 = None
smoothed_yaw1 = None
smoothed_yaw2 = None
cached_name = "Unknown"

last_update_time = 0.0
last_esp_trigger_time = 0.0
last_recog_time = 0.0
bpm_seen_at = 0.0   # >>> added to track BPM timing

print("✅ Running (synced dual-cam): Recognition + Yaw + BPM + ESP")
print("Press 'q' to quit.\n")

cam1 = LatestFrameReader(cam1_url)
cam2 = LatestFrameReader(cam2_url)

try:
    while True:
        ret1, frame1, ts1 = cam1.read_latest()
        ret2, frame2, ts2 = cam2.read_latest()
        now = time.perf_counter()

        if not (ret1 and ret2):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        if abs(ts1 - ts2) > MAX_SKEW_S:
            if SHOW_DEBUG:
                cv2.putText(frame1, "Desync… waiting", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame2, "Desync… waiting", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Cam1 - Recognition + Yaw", frame1)
            cv2.imshow("Cam2 - BPM + Yaw", frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ---------- CAMERA 1 ----------
        small_frame = cv2.resize(frame1, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        if (now - last_recog_time) >= RECOGNITION_PERIOD_S:
            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.45)
                face_distances = face_recognition.face_distance(known_faces, face_encoding)
                best_match = np.argmin(face_distances) if len(face_distances) > 0 else None
                cached_name = (known_names[best_match] if best_match is not None and matches[best_match]
                               else "Unknown")

                top, right, bottom, left = [v * 2 for v in (top, right, bottom, left)]
                cv2.rectangle(frame1, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame1, cached_name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                face_rect = [left, top, right - left, bottom - top]
                success, landmarks = facemark.fit(gray1, np.array([face_rect], dtype=np.int32))
                if success and len(landmarks) > 0:
                    prev_landmarks_cam1 = landmarks[0][0]
            last_recog_time = now

        if prev_landmarks_cam1 is not None:
            yaw1 = get_face_yaw(prev_landmarks_cam1)
            yaw_values_cam1.append(yaw1)
            smoothed_yaw1 = float(np.mean(yaw_values_cam1))
            cv2.putText(frame1, f"Yaw: {smoothed_yaw1:.1f}°", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # ---------- CAMERA 2 ----------
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
        if len(faces2) > 0:
            x, y, w, h = faces2[0]
            success, landmarks = facemark.fit(gray2, np.array([[x, y, w, h]], dtype=np.int32))
            if success and len(landmarks) > 0:
                prev_landmarks_cam2 = landmarks[0][0]
            roi = frame2[y:y+h, x:x+w]
            green = np.mean(roi[:, :, 1])
            frame_buffer.append(green)
            if len(frame_buffer) > FPS * WINDOW_SIZE:
                frame_buffer.pop(0)
            if (now - last_update_time) > UPDATE_INTERVAL and len(frame_buffer) == FPS * WINDOW_SIZE:
                bpm = estimate_bpm(frame_buffer, FPS)
                if bpm and 40 < bpm < 180:
                    bpm_values.append(bpm)
                    bpm_seen_at = time.time()  # >>> updated here
                last_update_time = now

        smoothed_bpm = float(np.mean(bpm_values)) if bpm_values else 0.0
        if smoothed_bpm > 0:
            cv2.putText(frame2, f"BPM: {int(smoothed_bpm)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if prev_landmarks_cam2 is not None:
            yaw2 = get_face_yaw(prev_landmarks_cam2)
            yaw_values_cam2.append(yaw2)
            smoothed_yaw2 = float(np.mean(yaw_values_cam2))
            cv2.putText(frame2, f"Yaw: {smoothed_yaw2:.1f}°", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # ---------- SPOOF + ESP TRIGGER ----------
        if smoothed_yaw1 is not None and smoothed_yaw2 is not None and cached_name != "Unknown":
            yaw_diff = abs(smoothed_yaw1 - smoothed_yaw2)
            yaw_peak = max(abs(smoothed_yaw1), abs(smoothed_yaw2))
            now_s = time.time()
            have_bpm = (smoothed_bpm > 0) or (now_s - bpm_seen_at <= BPM_GRACE_S)
            real_pose = (yaw_peak >= ABS_YAW_MIN) or (yaw_diff >= DIFF_YAW_MIN)

            cv2.putText(frame1, f"Y1={smoothed_yaw1:.1f}  Y2={smoothed_yaw2:.1f}  dY={yaw_diff:.1f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame2, f"Y1={smoothed_yaw1:.1f}  Y2={smoothed_yaw2:.1f}  dY={yaw_diff:.1f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if real_pose and have_bpm:
                label, color = f"REAL ✅ ΔYaw={yaw_diff:.1f}° | PeakYaw={yaw_peak:.1f}° | BPM={int(smoothed_bpm) if smoothed_bpm>0 else '…'}", (0, 255, 0)
                if now_s - last_esp_trigger_time > ESP_COOLDOWN:
                    print(f"[EVENT] {cached_name} | ΔYaw={yaw_diff:.1f}° | PeakYaw={yaw_peak:.1f}° | BPM={int(smoothed_bpm) if smoothed_bpm>0 else -1}")
                    trigger_gpio(1)
                    last_esp_trigger_time = now_s
            else:
                label, color = f"SPOOF ⚠️ ΔYaw={yaw_diff:.1f}° | PeakYaw={yaw_peak:.1f}°", (0, 0, 255)

            cv2.putText(frame1, label, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame2, label, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # ---------- DISPLAY ----------
        cv2.imshow("Cam1 - Recognition + Yaw", frame1)
        cv2.imshow("Cam2 - BPM + Yaw", frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()
