import cv2
import smtplib
import time
from flask import Flask, Response
from threading import Thread

thres = 0.55  # Confidence threshold
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Paths for the model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Initialize the object detection model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Email settings
fromaddr = "autoemail1234321@gmail.com"
password = "lleyvkvmwiyrpury"
toaddr = "autoemail1234321@gmail.com"

email_count = 0
email_limit = 5


app = Flask(__name__)

cap = cv2.VideoCapture(0)

def send_alert_email():
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Intruder Detected!"

    # Email body with link to live video
    body = "<h1>Intruder Detected!</h1><h4><a style='color:black; background-color:lightblue; padding:10px 30px; text-decoration:none; border-radius:5px;' href='http://127.0.0.1:5000/'>Click Here To See The Live Video</a></h4>"
    msg.attach(MIMEText(body, 'HTML'))

    filename = "opencv_frame_0.png"
    attachment = open(filename, "rb")
    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', f"attachment; filename= {filename}")
    msg.attach(p)

    # Send the email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, password)
    server.send_message(msg)
    server.quit()

def getObjs(img, draw=True, objects=[]):
    global email_count

    img_counter = 0
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=0.2)

    if len(objects) == 0:
        objects = classNames

    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]

            if className in objects:
                objectInfo.append([box, className])

                if className == 'person' and email_count < email_limit:
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    cv2.imwrite(img_name, img)
                    print(f"{img_name} written!")
                    send_alert_email()  # Send alert email
                    email_count += 1
                    time.sleep(5)

                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask_app():
    """Start Flask app in a separate thread."""
    app.run(debug=False)

if __name__ == "__main__":
    # Start the Flask app in a new thread for live feed
    flask_thread = Thread(target=start_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    # Main detection loop
    while True:
        success, img = cap.read()
        if not success:
            break
        result, objectInfo = getObjs(img)
        print(objectInfo)
        cv2.imshow("Output", img)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()
