import psutil
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import sys

# Access the passed arguments
arg1 = int(sys.argv[1])



def send_email(subject, body):
    # Email configuration
    sender_email = "cryptotraderbot.mf@gmail.com"
    receiver_email = "krzysztof.majchrzak10.km@gmail.com"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = "cryptotraderbot.mf@gmail.com"
    smtp_password = "nugnfkdrvppzbywi"

    # Create message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Connect to SMTP server and send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(message)

def check_process(process_pid):
    try:
        psutil.Process(process_pid)

    except psutil.NoSuchProcess:
        print(f"Process with PID {process_pid} not found or no longer exists.")
        subject = f"Process with PID {process_pid} has died!"
        body = f"The process with PID {process_pid} is no longer running."
        send_email(subject, body)
        print("Email sent")
        exit()

# Main loop to continuously check the process
while True:
    check_process(arg1)