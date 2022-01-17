from networks import Dis, Gen
from utils import weights_init, get_scheduler_cycle, TVLoss
import torch
import torch.nn as nn
import os

class CMIT(nn.Module):
    def __init__(self,opt,Nclass):
        super(CMIT, self).__init__()
        self.gen_a = Gen(opt.input_dim)  # auto-encoder for domain a
        self.gen_b = Gen(opt.input_dim)  # auto-encoder for domain b
        self.dis_a = Dis(Nsam=Nclass+1,opt=opt)  # discriminator for domain a
        self.dis_b = Dis(Nsam=Nclass+1,opt=opt)  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.Nclass = Nclass
        # Setup the optimizers
        beta1 = 0.5
        beta2 = 0.999
        self.tvloss=TVLoss(TVLoss_weight=0.01)

        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=opt.lr, betas=(beta1, beta2), weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=opt.lr, betas=(beta1, beta2), weight_decay=0.0001)
        self.dis_scheduler = get_scheduler_cycle(self.dis_opt,opt)
        self.gen_scheduler = get_scheduler_cycle(self.gen_opt,opt)

        # Network weight initialization
        self.apply(weights_init('kaiming'))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def gen_update(self, x_a, x_b):
        self.gen_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)
        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) 
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) 

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss_semi(x_ba,self.classLabel)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss_semi(x_ab,self.classLabel)
        
        self.loss_tv = self.tvloss(x_ab)
        # total loss
        #0.5*self.lambda_cyc * self.loss_gen_recon_x_a + \
        #0.5*self.lambda_cyc * self.loss_gen_recon_x_b + \
        self.loss_gen_total = self.loss_gen_adv_a + self.loss_gen_adv_b + 10.0 * self.loss_gen_recon_x_a + \
                              0.01 * self.loss_gen_recon_kl_a + 10.0 * self.loss_gen_recon_x_b + \
                              0.01 * self.loss_gen_recon_kl_b + 10.0 * self.loss_gen_cyc_x_a + \
                              0.01 * self.loss_gen_recon_kl_cyc_aba + 10.0 * self.loss_gen_cyc_x_b + \
                              0.01 * self.loss_gen_recon_kl_cyc_bab + self.loss_tv
        self.loss_gen_total.backward()
        self.gen_opt.step()
    
    def get_fake_b(self, x_a):
        self.eval()
        x_ab = []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            x_ab.append(self.gen_b.decode(h_a))
        x_ab = torch.cat(x_ab)
        self.train()
        return x_ab

    def dis_update(self, x_a, x_b):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss_semi(x_ba.detach(), x_a, self.fake_vect, self.classLabel)
        self.loss_dis_b = self.dis_b.calc_dis_loss_semi(x_ab.detach(), x_b, self.fake_vect, self.classLabel)
        self.loss_dis_total = self.loss_dis_a + self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        lr1 = self.dis_opt.param_groups[0]['lr']
        lr2 = self.gen_opt.param_groups[0]['lr']
        print('learning rate dis = %.7f, leanring rate gen = %.7f' % (lr1,lr2))
    
    def set_input(self,batch_size,cinfoi):
        classLabel=torch.zeros(batch_size,self.Nclass+1,dtype=torch.float32)#,self.fnSize,self.fnSize)
        fake_vect=torch.zeros(batch_size,self.Nclass+1,dtype=torch.float32)
        for i in range(batch_size):
            sinfo=cinfoi[i]
            classLabel[i,sinfo]+=1
            fake_vect[i,0]=1
        self.classLabel=classLabel.unsqueeze(2).unsqueeze(3).cuda().detach()
        self.fake_vect=fake_vect.unsqueeze(2).unsqueeze(3).cuda().detach()

    def save(self, snapshot_dir,save_mode='simple'):
        # Save generators, discriminators, and optimizers
        if save_mode=='full':
            gen_name = os.path.join(snapshot_dir, 'gen.pt')
            dis_name = os.path.join(snapshot_dir, 'dis.pt')
            opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
            torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
            torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
            torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
        else:
            gen_name = os.path.join(snapshot_dir, 'gen.pt')
            torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
