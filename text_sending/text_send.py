import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_sms_via_email(phone_number, carrier, message, sender_email, sender_password):
    """
    Send SMS via email-to-SMS gateway
    
    Args:
        phone_number: Your 10-digit phone number (no dashes/spaces)
        carrier: Your carrier name
        message: Text message to send
        sender_email: Your email address
        sender_password: Your email app password
    """
    
    # Carrier SMS gateways
    carriers = {
        'verizon': '@vtext.com',
        'att': '@txt.att.net',
        'tmobile': '@tmomail.net',
        'sprint': '@messaging.sprintpcs.com',
        'boost': '@smsmyboostmobile.com',
        'cricket': '@sms.cricketwireless.net',
        'uscellular': '@email.uscc.net',
        'metropcs': '@mymetropcs.com'
    }
    
    if carrier.lower() not in carriers:
        print(f"Carrier '{carrier}' not supported. Available: {list(carriers.keys())}")
        return False
    
    # Create SMS email address
    sms_email = phone_number + carriers[carrier.lower()]
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = sms_email
    msg['Subject'] = "test msg"  # Keep empty for SMS
    
    # Add message body
    msg.attach(MIMEText(message, 'plain'))
    
    try:
        # Gmail SMTP setup (adjust for other providers)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        
        # Send message
        text = msg.as_string()
        server.sendmail(sender_email, sms_email, text)
        server.quit()
        
        print(f"SMS sent successfully to {phone_number}")
        return True
        
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Configuration - UPDATE THESE VALUES
    PHONE_NUMBER = "4088966141"  # Your 10-digit phone number
    CARRIER = "verizon"          # Your carrier (lowercase)
    MESSAGE = "Test message from Python script!"
    SENDER_EMAIL = "anirban.kundu@gmail.com"     # Your email
    SENDER_PASSWORD = "rvih klof vwha patq"     # Your email app password
    
    # Send the SMS
    send_sms_via_email(PHONE_NUMBER, CARRIER, MESSAGE, SENDER_EMAIL, SENDER_PASSWORD)
