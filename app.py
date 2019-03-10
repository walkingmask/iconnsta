import base64
from io import BytesIO
import json

from flask import Flask, request, jsonify
from PIL import Image

from iconst import Iconnster, STYLE_MODEL, TRANSFORMER_MODEL


app = Flask(__name__)
iconnster = Iconnster(STYLE_MODEL, TRANSFORMER_MODEL)


def load_image(data, name):
    if name in data:
        image_binary = base64.b64decode(data[name].encode())
        return Image.open(BytesIO(image_binary))
    return None


@app.route('/', methods=['GET'])
def root():
    return 'hello, world'


@app.route('/api', methods=['POST'])
def api():
    data = json.loads(request.data)

    style = load_image(data, 'style')
    content = load_image(data, 'content')
    style_size = data['style_size'] if 'style_size' in data else None
    content_size = data['content_size'] if 'content_size' in data else None
    ratio = data['ratio'] if 'ratio' in data else None

    result = iconnster.transfer(style, content, style_size,
                                content_size, ratio)

    if iconnster.message:
        return jsonify({'error': iconnster.message})


    buffered = BytesIO()
    result.save(buffered, format='png')
    return jsonify({'result': base64.b64encode(buffered.getvalue()).decode('utf-8')})


if __name__ == '__main__':
    app.run(debug=True)
