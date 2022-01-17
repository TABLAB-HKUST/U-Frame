import argparse
import torch
import os
import time
from CMIT import CMIT
from utils import save_image,get_data_list,get_image,af2rgb
import shutil
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--ckp', type=str, default='demo', help='Checkpoint name.')
parser.add_argument('--dataroot', type=str, default='datasets/demo', help='Path to the data file.')
parser.add_argument('--batch_size', type=int, default=2, help="batch_size")
parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=40, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--overlap_size', type=int, default=64, help='overlapping size. 0< os <overlap_max')
parser.add_argument('--tb_dir', type=str, default='tmp', help='shuffle the input dataset.')
parser.add_argument('--lr', type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument('--gamma', type=float, default=0.0, help='gamma for focal loss.')
parser.add_argument('--tol', type=int, default=256, help='tolerance size')
parser.add_argument('--input_size', type=int, default=256, help='input patch size of the network')
parser.add_argument('--input_dim', type=int, default=3, help='number of input channels')
parser.add_argument('--iter', type=int, default=2000, help='iterations per epoch')
parser.add_argument('--save_mode', type=str, default='simple', help='save mode: simple | full')
parser.add_argument('--sobel', action='store_true',help='whether use sobel features or not.')
opts = parser.parse_args()


h=cv2.imread('%s/%s'%(opts.dataroot,os.listdir(opts.dataroot)[0])).shape[0]
print('Raw image height is: %d. Make sure it is true all over the dataset.'%(h))
edge_size=int((h-opts.tol)/2)
pix_start=int(edge_size-(opts.input_size-opts.tol)/2)
assert pix_start>=opts.overlap_size, 'overlap_size should be smaller than pix_start.'
assert opts.save_mode in ['simple','full'], 'error of save_mode'
assert opts.tol<=h, 'tolerance size should be smaller than image size'

checkpoint_folder='checkpoints/%s'%(opts.ckp)
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)
elif os.path.exists(checkpoint_folder):
    print('%s Already exists. Change to another name'%(checkpoint_folder))

if os.path.exists(opts.tb_dir):
    for f in os.listdir(opts.tb_dir):
        os.remove(os.path.join(opts.tb_dir, f)) 
else:
    os.makedirs(opts.tb_dir)

torch.backends.cudnn.benchmark = True

dataroot_new=af2rgb(opts.dataroot,opts.sobel) # add sobel feature, rename files to 1,2,..., and saved in "./tmp_for_sobel" folder
filename,cinfo,prob,Nclass=get_data_list(dataroot_new)
Nsample=cinfo.shape[0]
print(cinfo.shape,prob.shape,Nsample,Nclass)

cmit = CMIT(opts,Nclass)
cmit.cuda()
    
iter_per_epoch=int(max(opts.iter,Nsample)/opts.batch_size)


"""
if opts.patchsize==256:
    if opts.tol==128:
        pix_start=64 # 64+128+64=256 (h=128+128+128=384)
    elif opts.tol==64:
        pix_start=32 # 96+64+96=256 (h=128+64+128=320)
    elif opts.tol==256:
        pix_start=128 # 0+256+0=256 (h=128+256+128=512)
    else:
        assert False
elif opts.patchsize==128:
    if opts.tol==128:
        pix_start=128
    elif opts.tol==64:
        pix_start=96 # 32+64+32=128
    else:
        assert False
elif opts.patchsize==64:
    if opts.tol==64:
        pix_start=128
    else:
        assert False
else:
    assert False
"""

for epoch in range(1,opts.n_epochs + opts.n_epochs_decay + 1):
    epoch_start_time = time.time()  # timer for entire epoch
    for i in range(iter_per_epoch):
        images_a,images_b,cinfoi=get_image(dataroot_new,filename,prob,opts.batch_size,opts.overlap_size,cinfo,pix_start,opts.input_size)
        images_a=images_a[:,:opts.input_dim,:,:]
        images_b=images_b[:,:opts.input_dim,:,:]
        images_a=images_a.cuda().detach()
        images_b=images_b.cuda().detach()
        cmit.set_input(opts.batch_size,cinfoi)
        cmit.dis_update(images_a, images_b)
        cmit.gen_update(images_a, images_b)
    cmit.update_learning_rate()
    torch.cuda.synchronize()
    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opts.n_epochs + opts.n_epochs_decay, time.time() - epoch_start_time))
    save_image(cmit.get_fake_b(images_a),images_a,images_b,opts.tb_dir,epoch)
cmit.save(checkpoint_folder,save_mode=opts.save_mode)
shutil.rmtree(dataroot_new, ignore_errors=True) 
print('Training finished.')