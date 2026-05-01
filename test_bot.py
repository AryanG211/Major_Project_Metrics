# import requests

# bot_token = "8434898025:AAFwKZ54XDsvuVmnctb8na5WAZeufA1UZTw"
# chat_id = 1443963349
# text = "Criminal detected: John Doe at Camera 0 with confidence 0.85"

# requests.get(f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={text}")

import onnxruntime as ort
print("Device:", ort.get_device())