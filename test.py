# -*- coding: utf-8 -*-
import os
#os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,90).__str__()
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from networks import Gen
import argparse
from utils import af2rgb_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='examples/breast001', help='input folder name.')
    parser.add_argument('--ckp', type=str, default='breast001', help='checkpoint folder name.')
    parser.add_argument('--output', type=str, default='results/breast001_cmit', help='output folder name.')
    parser.add_argument('--input_dim', type=int, default=3, help='number of input channels')
    parser.add_argument('--sobel', action='store_true',help='whether use sobel features or not.') # --sobel to use sobel, default is not using sobel.
    opts = parser.parse_args()

    checkpoint_name=opts.ckp
    input_folder=opts.input
    output_folder=opts.output
    if checkpoint_name[-3:]=='.pt':
        checkpoint_name=checkpoint_name[:-3]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        1#,assert False, 'output folder exists, delete it first'
    
    state_dict = torch.load('checkpoints/%s/gen.pt'%(checkpoint_name),map_location='cuda:0')

    gen_a = Gen(opts.input_dim)
    gen_b = Gen(opts.input_dim)
    gen_a.load_state_dict(state_dict['a'])
    gen_b.load_state_dict(state_dict['b'])

    gen_a.cuda()
    gen_b.cuda()
    gen_a.eval()
    gen_b.eval()
    encode = gen_a.encode
    decode = gen_b.decode
    transform_list = []
    transform_list +=[transforms.ToTensor()]
    if opts.input_dim==1:
        transform_list +=[transforms.Normalize((0.5, ),(0.5, ))]
    elif opts.input_dim==2:
        transform_list +=[transforms.Normalize((0.5, 0.5),(0.5, 0.5))]
    else:
        transform_list +=[transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)

    for f in os.listdir(input_folder):
        image_path='%s/%s'%(input_folder,f)
        af2rgb_test(image_path,'tmp_thin_sobel.png',opts.sobel) # +sobel, save as tmp_thin_sobel.png
        im=cv2.imread('tmp_thin_sobel.png')[:,:,::-1] # rgb
        #imgij=Image.fromarray(np.uint8(im))
        imgij=Image.fromarray(im)
        imgij=transform(imgij)
        imgij = torch.unsqueeze(imgij[:opts.input_dim,:,:],0).float()
        realA=imgij.cuda().detach()
        content, _ = encode(realA)
        fake_B = decode(content)
        image_numpy = fake_B[0, :, :, :].cpu().float().detach().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        elif image_numpy.shape[0] == 2:
            image_numpy = np.tile(image_numpy, (2, 1, 1))
            image_numpy = image_numpy[:-1,:,:]
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        image_numpy = image_numpy.astype(np.uint8)
        Image.fromarray(image_numpy).save('%s/fake_cmit_%s'%(output_folder,f))


