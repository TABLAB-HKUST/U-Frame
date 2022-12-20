from torch.optim import lr_scheduler
from torchvision import transforms
import torch.nn as nn
import torch
import os
import math
import numpy as np
from PIL import Image
from random import randint
import cv2


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=0.01):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*torch.sqrt(h_tv/count_h+w_tv/count_w)/batch_size
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def af2rgb(folder,sobel_feature):
    folder_out='tmp_for_sobel'#%(randint(1,1000000))
    if not os.path.exists(folder):
        assert False, 'folder not exist.'
    if os.path.exists(folder_out):
        for f in os.listdir(folder_out):
            os.remove(os.path.join(folder_out, f)) 
    else:
        os.makedirs(folder_out)
    
    filename=os.listdir(folder)
    for i, f in enumerate(filename):
        x=cv2.imread('%s/%s'%(folder,f),1)
        h,w=x.shape[:2]
        xgray=x[:,:int(w/2),0]
        sobelx = cv2.Sobel(xgray, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(xgray, cv2.CV_64F, 0, 1)
        sobelx=(sobelx/1000+1)*255/2
        sobely=(sobely/1000+1)*255/2
        sobelx=cv2.convertScaleAbs(sobelx)
        sobely=cv2.convertScaleAbs(sobely)
        #s.append([sobelx,sobely])
        if sobel_feature==True:
            x[:,:int(w/2),:]=np.stack((xgray,sobelx,sobely),axis=2)
        cv2.imwrite('%s/%d.png'%(folder_out,i+1),x)
    return folder_out

def af2rgb_test(af,af_out,sobel_feature):
    img=cv2.imread(af,1)
    if sobel_feature==True:
        img=img[:,:,0]
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0).astype(np.float32)
        sobelx=(sobelx/1000+1)*255/2
        sobelx=sobelx.astype(np.uint8)

        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1).astype(np.float32)
        sobely=(sobely/1000+1)*255/2
        sobely=sobely.astype(np.uint8)
        img=np.stack((img,sobelx,sobely),axis=2)
    cv2.imwrite(af_out,img)

def save_image(fake_B,real_A,real_B,pathname,epoch):
    img=torch.cat((real_A,fake_B,real_B),3)
    n_show=np.min([3,img.size(0)])
    for j in range(n_show): # img.size(0)
        image_numpy = img[j, :, :, :].cpu().float().detach().numpy().copy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        cc='epoch_%d_sample_%d.png'%(epoch,j)
        image_path = os.path.join(pathname,cc)
        Image.fromarray(image_numpy.astype(np.uint8)).save(image_path)

def get_corr(x,patch_lens):
    C=1e-8
    patch_af=x[128:(128+patch_lens),128:(128+patch_lens),:]
    img_he=x[128:(128+patch_lens),(3*128+patch_lens):(3*128+2*patch_lens),:]
    gray_af=255-(patch_af[:,:,0] * 0.114  + patch_af[:,:,1] * 0.587+ patch_af[:,:,2] * 0.299).astype(np.float) # 0.11B+ 0.59G + 0.30R
    gray_he=(img_he[:,:,0] * 0.114 + img_he[:,:,1] * 0.587+ img_he[:,:,2] *0.299).astype(np.float)
    norm_af=(gray_af-np.mean(gray_af)+C)/(np.std(gray_af)+C)
    norm_he=(gray_he-np.mean(gray_he)+C)/(np.std(gray_he)+C)
    return np.sum(norm_af*norm_he)/(norm_af.shape[0]*norm_af.shape[1]-1)

def get_data_list(dataroot):
    expa,exp_he=0.5,1.0
    patch_size=cv2.imread(os.path.join(dataroot,os.listdir(dataroot)[0])).shape[0]
    patch_len=patch_size-256
    print('Tolerance size is: %d'%(patch_len))
    filename=os.listdir(dataroot)
    Nsample=len(filename)
    cinfo=[]
    prob=np.zeros((Nsample,256))
    r_he=np.zeros(Nsample)
    for i, f in enumerate(filename):
        cinfo.append(int(f[:f.rfind('.')]))
        #cinfo.append(list(map(int,f[(f.find('(')+1):f.find(')')].split(','))))
        x_input=cv2.imread('%s/%s'%(dataroot,f),1)
        rri=get_corr(x_input,patch_len)
        r_he[i]=(1+rri)*0.5 # [-1,1] ->[0,1]
        x=x_input[:,:patch_size,0] # [512,512]
        x=x[128:128+patch_len,128:128+patch_len]
        for j in range(256):
            prob[i,j]=(x==j).sum()
    cinfo=np.array(cinfo)
    Nclass=np.max(cinfo)
    
    probG=np.sum(prob,axis=0)/np.sum(prob)
    prob_sample=np.zeros(Nsample)
    for i in range(Nsample):
        pi=prob[i,:]/np.sum(prob[i,:])
        prob_sample[i]=(r_he[i]**exp_he)/(np.sum(pi*probG)**expa)
    prob_sample=prob_sample/np.sum(prob_sample)
    
    return filename,cinfo,prob_sample,Nclass

def get_image(dataroot,filename,prob,batch_size,randrange,cinfo,pix_start=64,patchlen=256):
    transform_list = []
    transform_list += [transforms.RandomHorizontalFlip()]
    transform_list +=[transforms.ToTensor()]
    transform_list +=[transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    Nsample=cinfo.shape[0]
    idx=np.random.choice(Nsample, batch_size, p=prob)
    AB = Image.open(os.path.join(dataroot, filename[idx[0]])).convert('RGB')
    cinfoi=cinfo[idx]
    w, h = AB.size
    w2 = int(w / 2)
    A = AB.crop((0, 0, w2, h)) 
    B = AB.crop((w2, 0, w, h))

    sx = randint(pix_start-randrange,pix_start+randrange-1)
    sy = randint(pix_start-randrange,pix_start+randrange-1)
    At=transform(A.crop((sx,sy,sx+patchlen,sy+patchlen))).unsqueeze(0)
    Bt=transform(B.crop((sx,sy,sx+patchlen,sy+patchlen))).unsqueeze(0)

    if batch_size>1:
        for i in range(1,batch_size):
            AB = Image.open(os.path.join(dataroot, filename[idx[i]])).convert('RGB')
            w, h = AB.size
            w2 = int(w / 2)
            A = AB.crop((0, 0, w2, h)) 
            B = AB.crop((w2, 0, w, h))
            sx = randint(pix_start-randrange,pix_start+randrange-1)
            sy = randint(pix_start-randrange,pix_start+randrange-1)
            Ati=transform(A.crop((sx,sy,sx+patchlen,sy+patchlen))).unsqueeze(0)
            Bti=transform(B.crop((sx,sy,sx+patchlen,sy+patchlen))).unsqueeze(0)
            #sxk.append(sx)
            #syk.append(sy)
            At=torch.cat((At,Ati),0)
            Bt=torch.cat((Bt,Bti),0)
    return At,Bt,cinfoi

def get_scheduler_cycle(optimizer, opt):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0) #

    return init_fun

