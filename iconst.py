from pathlib import Path

import tensorflow as tf
import numpy as np
from PIL import Image


STYLE_MODEL = str(Path(__file__).parent/'saved_model_style')
TRANSFORMER_MODEL = str(Path(__file__).parent/'saved_model_transformer')
STYLE_IMAGES = [Image.open(f) for f in Path(__file__).parent.glob('image/style/*.png')]
CONTENT_IMAGES = [Image.open(f) for f in Path(__file__).parent.glob('image/content/*.png')]
MIN_SIZE = 100
MAX_SIZE = 256


def get_random_style():
    index = np.random.randint(0, len(STYLE_IMAGES))
    return STYLE_IMAGES[index]


def get_random_content():
    index = np.random.randint(0, len(CONTENT_IMAGES))
    return CONTENT_IMAGES[index]


def get_random_size():
    return np.random.randint(MIN_SIZE, MAX_SIZE + 1)


def get_random_style_ratio():
    return np.random.rand()


def validate_image(image):
    # TODO: check image type in [jpg, png]
    if image.size[0] != image.size[1]:
        return 1, 'image must be square'
    return 0, None


def validate_size(size):
    if size < MIN_SIZE or size > MAX_SIZE:
        return 1, "size must be {} to {}".format(MIN_SIZE, MAX_SIZE)
    return 0, None


def validate_ratio(ratio):
    if ratio < 0.0 or ratio > 1.0:
        return 1, 'ratio must be 0 to 1'
    return 0, None


class Iconnster:
    def __init__(self, style_model, transformer_model):
        self.style_graph = tf.Graph()
        self.style_sess = tf.Session(graph=self.style_graph)
        self.transfer_graph = tf.Graph()
        self.transfer_sess = tf.Session(graph=self.transfer_graph)
        self._load_models(style_model, transformer_model)

        self.message = None

    def _load_models(self, style_model, transformer_model):
        with self.style_sess.as_default() as sess:
            tf.saved_model.loader.load(
                sess, [tf.saved_model.tag_constants.SERVING], style_model)
            self.style_in = sess.graph.get_tensor_by_name('Placeholder:0')
            self.bottleneck_out = \
                sess.graph.get_tensor_by_name('mobilenet_conv/Conv/BiasAdd:0')

        with self.transfer_sess.as_default() as sess:
            tf.saved_model.loader.load(
                sess, [tf.saved_model.tag_constants.SERVING], transformer_model)
            self.bottleneck_in = sess.graph.get_tensor_by_name('Placeholder_1:0')
            self.content_in = sess.graph.get_tensor_by_name('Placeholder:0')
            self.styled = \
                sess.graph.get_tensor_by_name('transformer/expand/conv3/conv/Sigmoid:0')

    def transfer(self, style=None, content=None, style_size=None,
                 content_size=None, style_ratio=None,):
        if style is None:
            style = get_random_style()
        if content is None:
            content = get_random_content()
        if style_size is None:
            style_size = get_random_size()
        if content_size is None:
            content_size = get_random_size()
        if style_ratio is None:
            style_ratio = get_random_style_ratio()

        error, message = validate_image(style)
        if error:
            self.message = message
            return
        error, message = validate_image(content)
        if error:
            self.message = message
            return
        error, message = validate_size(style_size)
        if error:
            self.message = message
            return
        error, message = validate_size(content_size)
        if error:
            self.message = message
            return

        style = style.resize((style_size, style_size))
        style = style.convert('RGB')
        style = np.asarray(style, dtype=np.float32) / 255.0
        style = np.expand_dims(style, 0)

        original_content = content.copy()
        content = content.resize((content_size, content_size))
        content = content.convert('RGB')
        content = np.asarray(content, dtype=np.float32) / 255.0
        content = np.expand_dims(content, 0)

        with self.style_sess.as_default() as sess:
            bottleneck = \
                sess.run(self.bottleneck_out, feed_dict={self.style_in: style})
            if style_ratio != 1.0:
                style_bottleneck = bottleneck
                identity_bottleneck = \
                    sess.run(self.bottleneck_out, feed_dict={self.style_in: content})
                style_bottleneck_scaled = style_ratio * style_bottleneck
                identity_bottleneck_scaled = (1.0 - style_ratio) * identity_bottleneck
                bottleneck = style_bottleneck_scaled + identity_bottleneck_scaled
        with self.transfer_sess.as_default() as sess:
            styled = \
                sess.run(self.styled, feed_dict={self.bottleneck_in: bottleneck,
                                                 self.content_in: content})

        styled = np.squeeze(styled)
        styled = np.uint8(styled * 255)
        styled = Image.fromarray(styled).resize(original_content.size)

        rgb = np.asarray(styled)
        alpha = np.asarray(original_content.convert('RGBA'))[:, :, 3:]
        result = np.concatenate([rgb, alpha], axis=2)

        return Image.fromarray(result)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_style', default=STYLE_MODEL)
    parser.add_argument('--saved_model_transformer', default=TRANSFORMER_MODEL)
    parser.add_argument('--style')
    parser.add_argument('--content')
    parser.add_argument('--style_size', type=int)
    parser.add_argument('--content_size', type=int)
    parser.add_argument('--style_ratio', type=float)
    args = parser.parse_args()

    iconnster = Iconnster(args.saved_model_style, args.saved_model_transformer)
    styled = iconnster.transfer(args.style, args.content,
                                args.style_size, args.content_size,
                                args.style_ratio)
    if iconnster.message:
        print(iconnster.message)
    else:
        styled.show()
