# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import argparse
from datetime import datetime
import os

from utils.params import *
from utils.tiramisu import *
from data.preprocess import load_image

import numpy as np
from PIL import Image

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import categorical_crossentropy

def img_resize(img):
    h, w, _ = img.shape
    nvpanels = h // 224
    nhpanels = w // 224
    new_h, new_w = h, w
    if nvpanels * 224 != h:
        new_h = (nvpanels + 1) * 224
    if nhpanels * 224 != w:
        new_w = (nhpanels + 1) * 224
    if new_h == h and new_w == w:
        return img
    else:
        return (np.array(Image.fromarray(img).resize((new_w, new_h))) \
                / 255. - mu) / std

def split_panels(img):
    h, w, _ = img.shape
    num_vert_panels = h // 224
    num_hor_panels = w // 224
    panels = []
    for i in range(num_vert_panels):
        for j in range(num_hor_panels):
            panels.append(img[i*224:(i+1)*224, j*224:(j+1)*224])
    return np.stack(panels)

def combine_panels(img, panels):
    h, w, _ = img.shape
    num_vert_panels = h // 224
    num_hor_panels = w // 224
    total = []
    p = 0
    for i in range(num_vert_panels):
        row = []
        for j in range(num_hor_panels):
            row.append(panels[p])
            p += 1
        total.append(np.concatenate(row, axis=1))
    return np.concatenate(total, axis=0)

def reshape_pred(pred): return pred.reshape(224, 224, 2)[:, :, 1]

def prediction_mask(img, target):
    layer1 = Image.fromarray(((img * std + mu) * 255).astype('uint8'))
    layer2 = Image.fromarray(
        np.concatenate(
            4 * [np.expand_dims(
                (225 * (1 - target)).astype('uint8'), axis=-1
            )], axis=-1
        )
    )
    result = Image.new('RGBA', layer1.size)
    result = Image.alpha_composite(result, layer1.convert('RGBA'))
    return Image.alpha_composite(result, layer2)

def wally_predict(model, img):
    rimg = img_resize(img)
    panels = split_panels(rimg)
    pred_panels = model.predict(panels, batch_size=6)
    pred_panels = np.stack([reshape_pred(pred) for pred in pred_panels])
    return rimg, combine_panels(rimg, pred_panels)

def main():
    now = datetime.now()
    today = now.strftime('%Y-%m-%d__%H-%M-%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('imgs', nargs='*', default=[])
    parser.add_argument('--model', default='models/model_2000epochs.h5')
    parser.add_argument('--output', default=f'data/imgs/predictions/{today}/')
    parser.add_argument('--size', type=tuple, default=(2800, 1760), help='resolution to load image')

    args = parser.parse_args()

    img_input = Input(shape=INPUT_SHAPE)
    x = create_tiramisu(
        2, img_input,
        nb_layers_per_block=[4, 5, 7, 10, 12, 15],
        p=0.2, wd=1e-4
    )
    model = Model(img_input, x)

    model.compile(
        loss=categorical_crossentropy,
        optimizer=RMSprop(1e-3),
        metrics=['accuracy'],
        sample_weight_mode='temporal'
    )

    model.load_weights(args.model)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for i, img in enumerate(args.imgs):
        full_img = load_image(img, args.size)
        full_img_r, full_pred = wally_predict(model, full_img)
        mask = prediction_mask(full_img_r, full_pred)
        mask.save(os.path.join(args.output, str(i+1) + '.png'))

if __name__ == '__main__':
    main()