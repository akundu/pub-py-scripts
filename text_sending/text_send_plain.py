import smtplib

def send_sms_via_email(phone_number, carrier, message, sender_email, sender_password):
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

    sms_email = phone_number + carriers[carrier.lower()]
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)

        # Notice: no multipart, just a plain string message
        server.sendmail(sender_email, sms_email, message)  
        server.quit()
        
        print(f"SMS sent successfully to {phone_number}")
        return True
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return False

# Example usage
if __name__ == "__main__":
    PHONE_NUMBER = "4088966141"  # Your 10-digit phone number
    CARRIER = "verizon"          # Your carrier (lowercase)
    MESSAGE = "Test message from Python script!"
    SENDER_EMAIL = "anirban.kundu@gmail.com"     # Your email
    SENDER_PASSWORD = "rvih klof vwha patq"     # Your email app password
    
    send_sms_via_email(PHONE_NUMBER, CARRIER, MESSAGE, SENDER_EMAIL, SENDER_PASSWORD)
