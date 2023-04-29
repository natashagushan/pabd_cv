import PIL.Image
import sys
import io
from requests import request

img = PIL.Image.open('../data/dog.jpg')
type(img)

buffer = io.BytesIO()
img.save(buffer, format='JPEG')

with buffer as buf:
    buffer.seek(0)
    response = request('POST', 'http://localhost:1234/classify', data=buf)

print(response.content)
