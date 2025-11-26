# import cv2
# from flask import Flask, Response

# app = Flask(_name_)
# cap = cv2.VideoCapture(0)  # open webcam

# def gen_frames():
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if _name_ == '_main_':
#     app.run(host='0.0.0.0', port=5000)


import cv2
from flask import Flask, Response

app = Flask(__name__)

# Open webcam and set to 480p
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 20)  # 20 FPS is smooth yet efficient

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Ensure frame is 480p
        frame = cv2.resize(frame, (640, 480))

        # Encode with moderate compression (quality 50–60)
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)