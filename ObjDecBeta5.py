import cv2
import smtplib
import time

thres = 0.55  # Confidence level that it is actually an object
classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#Dataset Paths

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#Passing paths to the object detection model

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjs(img, draw=False, objects=[]):
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    fromaddr = "autoemail1234321@gmail.com"
    password = "lleyvkvmwiyrpury"
    toaddr = "autoemail1234321@gmail.com"

    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "We have detected an unknown person!"

    # string to store the body of the mail
    body = "<h1> Intruder Detected! </h1> <h4> <a style='color:black; background-color:lightblue; padding:10px 30px; text-decoration:none; border-radius:5px;' href='http://127.0.0.1:5000/'> Click Here To See The Live Video </a> </h4>"

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'HTML'))

    filename = "opencv_frame_0.png"
    attachment = open(filename, "rb")

    # instance of MIMEBase and named as p
    p = MIMEBase('application', 'octet-stream')

    # To change the payload into encoded form
    p.set_payload((attachment).read())

    # encode into base64
    encoders.encode_base64(p)

    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    # attach the instance 'p' to instance 'msg'
    msg.attach(p)

    # creates SMTP session
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, password)

    # Converts the Multipart msg into a string
    text = msg.as_string()

    img_counter = 0
    ret, frame = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=0.2) #nmsThreshold will prevent overlapping detection, lower values will have greater effects
    print(bbox)

    if len(objects) == 0:
        objects = classNames

    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]

            if className in objects:
                objectInfo.append([box, className])

                if className == 'person':
                    # counter = counter + 1
                    img_name = "opencv_frame_{}.png".format(img_counter)
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))

                    server.send_message(msg)
                    time.sleep(5)

                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(10, 70)

    while True:
        success, img = cap.read()
        result, objectInfo = getObjs(img)
        print(objectInfo)
        cv2.imshow("Output", img)
        #not having a waitkey will stop the program after detecting the first image!
        cv2.waitKey(1)
