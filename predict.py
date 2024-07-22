import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from data_loading import BasicDataset
from edge_detector_model import EdgeDetectorModel
from utils.draw import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                resolution):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, resolution=resolution, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        mask = output.argmax(dim=1)

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Filenames of output images')
    parser.add_argument('--resolution', default=224, type=int, metavar='N',
                    help='input NxN image resolution of model (default: 224x224) ')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--canny-path', metavar='CANNY', help='Path to True Canny masks for visualization (--viz)', default="")
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    Path(args.output).mkdir(parents=True, exist_ok=True)

    in_files = [f.path for f in os.scandir(args.input) if f.is_file()]
    if args.viz:
        if args.canny_path is not None and os.path.exists(args.canny_path):
            canny_masks_files = {os.path.splitext(os.path.basename(f.path))[0]:f.path for f in os.scandir(args.canny_path) if f.is_file()}
        else:
            canny_masks_files = {}
    out_files = get_output_filenames(args)

    net = EdgeDetectorModel(bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    train_statistic = state_dict.pop('train_statistics')
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        original = Image.open(filename).resize((args.resolution, args.resolution))
        img = original.convert("L")

        mask = predict_img(net=net,
                           full_img=img,
                           resolution=args.resolution,
                           device=device)

        if not args.no_save:
            out_filename = f"{os.path.splitext(os.path.basename(filename))[0]}.png"
            out_file_path = os.path.join(args.output, out_filename)
            result = mask_to_image(mask, mask_values)
            result.save(out_file_path)

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            filename_only = os.path.splitext(os.path.basename(filename))[0]
            canny_file = canny_masks_files.get(filename_only, None)
            if canny_file is not None:
                canny_img = Image.open(canny_file)
            else:
                canny_img = None
            plot_img_and_mask(original, mask, canny_img)
