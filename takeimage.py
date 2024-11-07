import cv2
# import smtplib
#
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.base import MIMEBase
# from email import encoders
#
# fromaddr = "autoemail1234321@gmail.com"
# password = "maktab2021"
# toaddr = "sultanhuzaifa0@gmail.com"
#
# msg = MIMEMultipart()
#
# # storing the senders email address
# msg['From'] = fromaddr
# msg['To'] = toaddr
# msg['Subject'] = "Person Found"
#
# # string to store the body of the mail
# body = "Body_of_the_mail"
#
# # attach the body with the msg instance
# msg.attach(MIMEText(body, 'plain'))
#
# filename = "opencv_frame_0.png"
# attachment = open(filename, "rb")
#
# # instance of MIMEBase and named as p
# p = MIMEBase('application', 'octet-stream')
#
# # To change the payload into encoded form
# p.set_payload((attachment).read())
#
#
# # encode into base64
# encoders.encode_base64(p)
#
# p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
#
# # attach the instance 'p' to instance 'msg'
# msg.attach(p)
#
# # creates SMTP session
# server = smtplib.SMTP('smtp.gmail.com', 587)
# server.starttls()
# server.login(fromaddr,password)
#
# # Converts the Multipart msg into a string
# text = msg.as_string()
#
#

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        # img_counter += 1
        # server.send_message(msg)
        

cam.release()

cv2.destroyAllWindows()