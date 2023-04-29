"""Flask   2.2.3"""
from flask import Flask

app = Flask('Image classifier')


@app.route('/')
def home():
    """return string 'Home page' for / page"""
    return 'Home page'


if __name__ == '__main__':
    app.run(port=1800) #номер зачетки
    input()
