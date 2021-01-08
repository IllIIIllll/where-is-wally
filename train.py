# © 2021 지성. all rights reserved.
# <llllllllll@kakao.com>
# MIT License

import argparse

from data.generator import *
from utils.tiramisu import *
from utils.params import INPUT_SHAPE

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs', default='data/imgs/imgs.npy')
    parser.add_argument('--trgs', default='data/imgs/trgs.npy')
    parser.add_argument('--wally-sub-imgs', default='data/imgs/wally_sub_imgs.npy')
    parser.add_argument('--wally-sub-trgs', default='data/imgs/wally_sub_trgs.npy')
    parser.add_argument('--model', default='models/model.h5')
    parser.add_argument('--tot-bs', type=int, default=6, help='total batch size')
    parser.add_argument('--prop', type=float, default=.34, help='proportion')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--spe', type=int, default=6, help='steps per epoch')

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

    gen_mix = seg_gen_mix(
        args.wally_sub_imgs,
        args.wally_sub_trgs,
        args.imgs, args.trgs,
        tot_bs=args.tot_bs,
        prop=args.prop
    )

    model.fit(
        gen_mix,
        epochs=args.epochs,
        steps_per_epoch=args.spe,
        verbose=1,
        callbacks=[
            ModelCheckpoint(
                f'checkpoints/model_{args.epochs}_epochs_{{epoch}}.h5',
                save_freq=int(args.eppochs * args.spe / 5)
            )
        ]
    )

    model.save(args.model)

if __name__ == '__main__':
    main()