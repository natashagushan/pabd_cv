import unittest
from requests import request
import io
import PIL.Image
from flack import Flask

class MyTestCase(unittest.TestCase):
    def test_200(self):
        response = requests.request('GET', 'http://localhost:1774/')
        sample = response.content.decode()
        self.assertEqual(sample, 'Home page')  # add assertion here

    def test_classify(self):
        img = PIL.Image.open('../data/dog.jpg')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')

        with buffer as buf:
            buffer.seek(0)
            response = request('POST', 'http://localhost:1774/classify', data=buf)

        out = response.content.decode('utf-8')
        expected = 'келли, Пембрук, Немецкая овчарка'
        self.assertEqual(out, expected)


if __name__ == '__main__':
    unittest.main()
