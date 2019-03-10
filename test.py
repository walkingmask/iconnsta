import base64
from io import BytesIO
import json
from PIL import Image
import requests


data = {}

data['key'] = 'qwertyuiopasdfghjklzxcbnm'

image = Image.open('image/style/style1.png')
buffered = BytesIO()
image.save(buffered, format='png')
data['style'] = base64.b64encode(buffered.getvalue()).decode('utf-8')
data['style_size'] = 256

image = Image.open('image/content/content1.png')
buffered = BytesIO()
image.save(buffered, format='png')
data['content'] = base64.b64encode(buffered.getvalue()).decode('utf-8')
data['content_size'] = 256

data['ratio'] = 0.5

api = 'http://127.0.0.1:5000/api'
headers = {'content-type': 'application/json'}
# response = requests.post(api, headers=headers, json=data)
response = requests.post(api, headers=headers, json={})

result = response.json()['result'].encode()
Image.open(BytesIO(base64.b64decode(result))).show()
