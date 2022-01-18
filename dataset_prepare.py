import os
import cv2 
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='source image path.')
    parser.add_argument('--target', type=str, help='target image path.')
    parser.add_argument('--output', type=str, help='output path.')
    parser.add_argument('--tol', type=int, help='tolerance size')
    opts = parser.parse_args()

    output_folder=opts.output
    if os.path.exists(output_folder): 
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f)) 
    else: # otherwise make a new folder
        os.makedirs('%s'%(output_folder))
    im_source=cv2.imread(opts.source,1)
    im_target=cv2.imread(opts.target,1)
    h,w=im_source.shape[:2]
    print(im_source.shape,im_target.shape)
    assert im_target.shape[0]==h and im_target.shape[1]==w, 'error size not match for image_a and image_b'
    dw=dh=opts.tol
    omax=128 # maximal overlapping_size
    k=0
    patch_size=opts.tol
    for i in range(128,h-patch_size-omax+1,dh):
        for j in range(128,w-patch_size-omax+1,dw):
            nameij='%s/TRAIN_%d_%d_class_%d.png'%(opts.output,i,j,k)
            cv2.imwrite(nameij,np.concatenate((im_source[(i-128):(i+patch_size+128),(j-128):(j+patch_size+128),:],im_target[(i-128):(i+patch_size+128),(j-128):(j+patch_size+128),:]),axis=1))
            k=k+1







