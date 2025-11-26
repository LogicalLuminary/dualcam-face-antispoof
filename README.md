# Dual-Cam Face Recognition + Yaw + BPM + ESP

#A dual-camera, hardware-integrated face authentication system with real-time anti-spoofing. Uses yaw values from both cameras to calculate facial curveness, and OpenCV-based models to detect and recognize registered faces while blocking photo/video spoof attacks. Includes complete microcontroller hardware integration for secure verification.

This project uses two cameras to perform face recognition (camera 1) while measuring facial yaw and remote photoplethysmography (rPPG/BPM) from camera 2 to detect whether a live subject is real or a spoof (printed/photo/video). When a real subject is detected, the app can trigger a GPIO on an ESP device.

## Files

* `code6.py` — main application: recognition (face_recognition), facial landmarks (LBF), yaw calculation, BPM estimation and ESP triggering.
* `stream_server.py` — simple webcam stream server (if you run a phone/ESP as a camera source).
* `requirements.txt` — python dependencies.
* `lbfmodel.yaml` — LBF landmarks model (must be present in project root or update path).
* `face/` — folder with known face images (filename without extension is used as the person's name).

## Features

* Dual-camera sync (attempts to use frames with similar timestamps)
* Face recognition (camera 1) using `face_recognition` encodings
* Landmark-based yaw estimation (OpenCV LBF)
* rPPG-like BPM estimation from mean green channel on a face ROI
* Hybrid spoof detection using yaw (absolute and differential) + BPM
* ESP trigger via simple HTTP GET with token-based parameter


1. **Create & activate virtual environment**
NOTE -  Use python 3.11 

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. **Install dependencies**

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> **Windows notes:** `face_recognition` depends on `dlib`. If `pip install dlib` fails, install Visual Studio Build Tools and CMake first, or use a prebuilt wheel. See the repository README for `dlib` for platform-specific guidance.

3. **Place models & faces**

* Ensure `lbfmodel.yaml` is in the project root (or update the path in the code).
* Add one or more face images to the `face/` directory. Filenames (without extension) become the labels used by recognition.

4. **Edit configuration** (optional)
   Open `code6.py` and update camera URLs and ESP values near the top:

5. **Run**
   Start any required stream servers (for phones or remote cameras) on both the devices. Then run the main script:

```powershell
python stream_server.py   # if you're using the included stream server
python code6.py
```
```py
Use your camera urls and servers IP address.
cam1_url = "http://192.168.183.20:5000/video_feed"
cam2_url = "http://192.168.183.244:5000/video_feed"
ESP_IP = "192.168.183.219"
TOKEN = "mytoken123"
```

## Usage

* Two windows will open: `Cam1 - Recognition + Yaw` and `Cam2 - BPM + Yaw`.
* Press `q` inside the OpenCV window to quit.

## Configuration parameters

Several tunable constants are at the top of `code6.py`:

* `FPS`, `WINDOW_SIZE`, `UPDATE_INTERVAL` — affect BPM buffer and update frequency
* `RECOGNITION_PERIOD_S` — how often recognition runs
* `ESP_COOLDOWN` — minimum seconds between ESP triggers
* `ABS_YAW_MIN`, `DIFF_YAW_MIN`, `BPM_GRACE_S` — spoof detection thresholds and grace period for BPM

These can be tweaked for your camera distances, lighting and face sizes.

## Dependencies

Key packages (see `requirements.txt` in repo):

* `opencv-contrib-python` (for cv2.face LBF)
* `face_recognition` (wraps `dlib`) — heavy dependency
* `numpy`, `scipy` (signal processing), `requests`

If you have issues installing `opencv-contrib-python` and `dlib` on Windows, consider using Python 3.9/3.10 wheels or WSL / a Linux VM.

## Troubleshooting

* **No faces loaded / recognition fails**: Ensure `face/` contains clear frontal images and `lbfmodel.yaml` is present.
* **BPM is zero / noisy**: Increase `WINDOW_SIZE`, improve lighting, or use a tighter face ROI.
* **Stream problems**: Verify camera URLs (try opening in a browser). If using `stream_server.py` from a phone, ensure the phone and PC are on same network.
* **dlib build failures (Windows)**: Install Visual Studio Build Tools and CMake or install a prebuilt wheel.

## Security & Privacy

* Don't commit large raw face image datasets or private tokens to a public repo. Put secrets (like `TOKEN`) in a `.env` file and add it to `.gitignore`.

## Suggested `.gitignore`

```
__pycache__/
*.pyc
.venv/
.env
.vscode/
.idea/
```

