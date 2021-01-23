import requests
import time

trycnt = 3
while trycnt > 0:
    try:
        resp = requests.post('http://localhost:5000/predict', files={'file': open('image_03.jpg', 'rb')})
        print('Succeed')
        print(resp.json())
        trycnt = 0
    except:
        if trycnt <= 0:
            print('Failed to connect')
        else:
            trycnt -= 1
        time.sleep(0.5)

#print(resp.json())
