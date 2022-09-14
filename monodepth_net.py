from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import monodepth2.networks
from monodepth2.layers import disp_to_depth
from monodepth2.utils import download_model_if_doesnt_exist

def load_depth_net(device, model_name='mono_stereo_640x192'):
    """Function to predict for a single image or folder of images
    """
    assert model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    model_path = os.path.join("monodepth2","models", model_name)
    print("-> Loading depth model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    depth_encoder = monodepth2.networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
    depth_encoder.load_state_dict(filtered_dict_enc)
    depth_encoder.to(device)
    depth_encoder.eval()

    print("   Loading depth decoder")
    depth_decoder = monodepth2.networks.DepthDecoder(
        num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    return depth_encoder, depth_decoder, feed_width, feed_height

def run_depth(image,device,depth_encoder, depth_decoder, feed_width, feed_height):
    input_image = image.convert('RGB') # Do we need to convert?
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    # PREDICTION
    input_image = input_image.to(device)
    features = depth_encoder(input_image)
    depth = depth_decoder(features)

    # TODO transform outputs back to PIL format and resize into original size

    return depth