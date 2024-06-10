import boto3

# Create an SES client
ses_client = boto3.client('ses', region_name='us-east-1')

# Specify the email content
sender_email = "vidyarang.ai@gmail.com"
recipient_email = "chitranshuharbola@gmail.com"
subject = "Test Email"
body = "This is a test email sent using Amazon SES."

# Send the email
try:
    response = ses_client.send_email(
        Destination={
            'ToAddresses': [recipient_email],
        },
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': body,
                },
            },
            'Subject': {
                'Charset': 'UTF-8',
                'Data': subject,
            },
        },
        Source=sender_email
    )
    print("Email sent successfully!")
    print("Message ID:", response['MessageId'])
except Exception as e:
    print("Error sending email:", e)
