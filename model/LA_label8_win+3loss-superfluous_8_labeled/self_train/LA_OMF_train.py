from asyncore import write
import imp
import os
import nibabel as nib
from sre_parse import SPECIAL_CHARS
import sys
from xml.etree.ElementInclude import default_loader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np
from medpy import metric
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
import pdb

from yaml import parse
from skimage.measure import label
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import losses, ramps, feature_memory, contrastive_losses, test_3d_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, update_ema_variables , vigil_loss
#vigil
import matplotlib.pyplot as plt

'''import h5py
file = h5py.File('/home/vigil/Desktop/BCP-main/data/byh_data/SSNet_data/LA/2018LA_Seg_Training Set/0RZDK210BSMWAA6467LU/mri_norm2.h5','r')
file.keys()
exit()'''
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/vigil/Desktop/BCP-main/data/byh_data/SSNet_data/LA', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='BCP', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--pre_max_iteration', type=int,  default=15000, help='maximum pre-train iteration to train')
parser.add_argument('--self_max_iteration', type=int,  default=15000, help='maximum self-train iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.1, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=8, help='trained samples')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--consistency', type=float, default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='10.0', help='magnitude')
# -- setting of BCP
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
parser.add_argument('--mask_ratio', type=float, default=2/3, help='ratio of mask/image')
# -- setting of mixup
parser.add_argument('--u_alpha', type=float, default=2.0, help='unlabeled image ratio of mixuped image')
parser.add_argument('--loss_weight', type=float, default=0.5, help='loss weight of unimage term')
args = parser.parse_args()

nii_save_path='/home/vigil/Desktop/labelnum16/'
#vigil:========================================================================================================================
def vigil_mask(laborplab):
    batch_size, l_x, l_y, l_z = laborplab.shape[0],laborplab.shape[1],laborplab.shape[2],laborplab.shape[3]
    #'laborplab':lab=1:white)
    loss_mask = torch.ones(batch_size, l_x, l_y, l_z).cuda()
    mask = torch.ones(l_x, l_y, l_z).cuda()
    #laborplab = laborplab.narrow(0,1,1)
    #laborplab = torch.squeeze(laborplab , dim=0)
    mask1=laborplab[0,:,:,:]
    mask2=laborplab[1,:,:,:]

    mask1 = mask - mask1
    mask2 = mask - mask2
    
    loss_mask[0,:,:,:] = mask1
    loss_mask[1,:,:,:] = mask2

	
    return mask1.long(),mask2.long(),loss_mask.long()
    
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam   
    
def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div
CE = nn.CrossEntropyLoss(reduction='none') 

consistency_criterion = softmax_mse_loss

#vigil:========================================================================================================================   
    
def get_cut_mask(out, thres=0.5, nms=0):
    #'out'[2, 2, 112, 112, 80])
    probs = F.softmax(out, 1)
    #'prob'[2, 2, 112, 112, 80])
    masks = (probs >= thres).type(torch.int64)
    #masks.size()[2, 2, 112, 112, 80]
    masks = masks[:, 1, :, :].contiguous()
    #'masks'[2, 112, 112, 80])
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    
    return torch.Tensor(batch_list).cuda()

def save_net_opt(net, optimizer, path):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
    }
    torch.save(state, str(path))

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pre_max_iterations = args.pre_max_iteration
self_max_iterations = args.self_max_iteration
base_lr = args.base_lr
CE = nn.CrossEntropyLoss(reduction='none')

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2

def pre_train(args, snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    DICE = losses.mask_DiceLoss(nclass=2)

    model.train()
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = pre_max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'][:args.labeled_bs], sampled_batch['label'][:args.labeled_bs]
		
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:] 
           
            with torch.no_grad():
                #img_mask, loss_mask = context_mask(img_a, args.mask_ratio)
                img_mask_1 , img_mask_2 , loss_mask = vigil_mask(lab_a)
            
		
            """Mix Input"""
            img_a=img_a.squeeze()
            
            img_b=img_b.squeeze()
            
            
            volume_batch = img_a * ( 1 - loss_mask) + img_b * loss_mask
            volume_batch = torch.unsqueeze(volume_batch , dim=1)
            label_batch = lab_a * ( 1 - loss_mask) + lab_b * loss_mask

            
            outputs, _ = model(volume_batch)
            #===========================================
            img_a = torch.unsqueeze(img_a,dim=1)
            img_b = torch.unsqueeze(img_b,dim=1)
            
            outputs_a, _ = model(img_a)
            outputs_b, _ = model(img_b)
            #===========================================
            loss_ce = F.cross_entropy(outputs, label_batch)
            loss_dice = DICE(outputs, label_batch)
            loss1 = (loss_ce + loss_dice) / 2
            #===========================================
            loss_ce_a = F.cross_entropy(outputs_a, lab_a)
            loss_dice_a = DICE(outputs_a, lab_a)
            loss2 = (loss_ce_a + loss_dice_a) / 2
            
            loss_ce_b = F.cross_entropy(outputs_b, lab_b)
            loss_dice_b = DICE(outputs_b, lab_b)
            loss3 = (loss_ce_b + loss_dice_b) / 2
            #===========================================
            loss=loss1+loss2+loss3
            iter_num += 1
            writer.add_scalar('pre/loss_dice', loss_dice, iter_num)
            writer.add_scalar('pre/loss_ce', loss_ce, iter_num)
            writer.add_scalar('pre/loss_all', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_dice: %03f, loss_ce: %03f'%(iter_num, loss, loss_dice, loss_ce))
            #vigil=====================================================================================================
            # change lr
            if iter_num % 10000 == 0:
                lr_ = base_lr * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            #vigil=====================================================================================================       
                    
            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)
                    # torch.save(model.state_dict(), save_mode_path)
                    # torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= pre_max_iterations:
                break

        if iter_num >= pre_max_iterations:
            iterator.close()
            break
    writer.close()


def self_train(args, pre_snapshot_path, self_snapshot_path):
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    ema_model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    for param in ema_model.parameters():
            param.detach_()   # ema_model set
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)
    sub_bs = int(args.labeled_bs/2)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    pretrained_model = os.path.join(pre_snapshot_path, f'{args.model}_best_model.pth')
    load_net(model, pretrained_model)
    load_net(ema_model, pretrained_model)
    
    model.train()
    ema_model.train()
    writer = SummaryWriter(self_snapshot_path+'/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    iter_num = 0
    best_dice = 0
    max_epoch = self_max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            #volume_batch.size()[8, 1, 112, 112, 80])
            #label_batch.size()[8, 112, 112, 80])
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            img_a, img_b = volume_batch[:sub_bs], volume_batch[sub_bs:args.labeled_bs]
            #img_a   [2, 1, 112, 112, 80]
            lab_a, lab_b = label_batch[:sub_bs], label_batch[sub_bs:args.labeled_bs]
            #lab_a   [2, 112, 112, 80]

            unimg_a, unimg_b = volume_batch[args.labeled_bs:args.labeled_bs+sub_bs], volume_batch[args.labeled_bs+sub_bs:]
            
            dont_use_unimg_a, dont_use_unimg_b = label_batch[args.labeled_bs:args.labeled_bs+sub_bs], label_batch[args.labeled_bs+sub_bs:]#just for comparison
            #unimg_a  [2, 1, 112, 112, 80]
            #unimg_b  [2, 1, 112, 112, 80]
            

            with torch.no_grad():
                unoutput_a, _ = ema_model(unimg_a)
                unoutput_b, _ = ema_model(unimg_b)
                #unoutput_a  [2, 2, 112, 112, 80]
                #unoutput_b  [2, 2, 112, 112, 80]
                plab_a = get_cut_mask(unoutput_a, nms=1)
                #'plab'[2, 112, 112, 80])
                plab_b = get_cut_mask(unoutput_b, nms=1)
                #img_mask, loss_mask = context_mask(img_a, args.mask_ratio)
                img_mask_a1, img_mask_a2 , loss_mask_a = vigil_mask(lab_a)#torch.equal(img_mask_a1, img_mask_a2) False

                #'img_mask_a'[112, 112, 80])
                img_mask_b1, img_mask_b2 , loss_mask_b = vigil_mask(plab_a)
                img_mask_c1, img_mask_c2 , loss_mask_c = vigil_mask(plab_b)
                
            consistency_weight = get_current_consistency_weight(iter_num // 150)

 #vigil:========================================================================================================================
            img_a=img_a.squeeze()
            unimg_a=unimg_a.squeeze()
            img_b=img_b.squeeze()
            unimg_b=unimg_b.squeeze()
            
            mixc_img = unimg_b * (1 - loss_mask_c) + unimg_a * loss_mask_c
            mixc_img= torch.unsqueeze(mixc_img,dim=1)
            mixc_lab = plab_b * (1 - loss_mask_c) + plab_a * loss_mask_c

            unimg_a = torch.unsqueeze(unimg_a,dim=1)
            unimg_b = torch.unsqueeze(unimg_b,dim=1)
 #vigil:========================================================================================================================

            outputs_c, _ = model(mixc_img)

            loss_a = consistency_criterion(outputs_c, unoutput_a).mean()
            loss_b = 0

            loss = 1.7 * loss_a

            iter_num += 1
            writer.add_scalar('Self/consistency', consistency_weight, iter_num)
            writer.add_scalar('Self/loss_a', loss_a, iter_num)
            writer.add_scalar('Self/loss_all', loss, iter_num)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss: %03f, loss_a: %03f'%(iter_num, loss, loss_a))

            #update_ema_variables(model, ema_model, 0.9999999)

             # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.8 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % 200 == 0:
                model.eval()
                dice_sample = test_3d_patch.var_all_case_LA(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4)
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(self_snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(self_snapshot_path,'{}_best_model.pth'.format(args.model))
                    # save_net_opt(model, optimizer, save_mode_path)
                    # save_net_opt(model, optimizer, save_best_path)
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()
              
            if iter_num % 200 == 1:
                '''  
                unimg_b = unimg_b.repeat(2, 1, 1, 1, 1)
                T=8
                stride = unimg_b.shape[0] // 2
                preds = torch.zeros([stride * T, 2, 112, 112, 80]).cuda()
                
                for i in range(T//2):
                    b_inputs = unimg_b + torch.clamp(torch.randn_like(unimg_b) * 0.1, -0.2, 0.2)
                    with torch.no_grad():
                         b_pred,_ = model(b_inputs)
                         preds[2 * stride * i:2 * stride * (i + 1)] = b_pred
                preds = F.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, 2, 112, 112, 80)
                preds = torch.mean(preds, dim=0)  #(batch, 2, 112,112,80)
                uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)
                nib.save(nib.Nifti1Image(uncertainty[0,0,:,:,:].cpu().numpy().astype(np.float32), np.eye(4)), nii_save_path + "%02d_img_uncertaintybbb.nii.gz" % iter_num)
                nib.save(nib.Nifti1Image(unimg_a[0,0,:,:,:].cpu().numpy().astype(np.float32), np.eye(4)), nii_save_path + "%02d_unimga_.nii.gz" % iter_num)
                nib.save(nib.Nifti1Image(unimg_b[0,0,:,:,:].cpu().numpy().astype(np.float32), np.eye(4)), nii_save_path + "%02d_unimgb_.nii.gz" % iter_num)
                '''  
            if iter_num % 200 == 1:       #yuan lai shi 200   
                
                nib.save(nib.Nifti1Image(mixc_img[0,0,:,:,:].cpu().numpy().astype(np.float32), np.eye(4)), nii_save_path + "%02d_img.nii.gz" % iter_num)
                nib.save(nib.Nifti1Image(mixc_lab[0,:,:,:].cpu().numpy().astype(np.float32), np.eye(4)), nii_save_path + "%02d_gt.nii.gz" % iter_num)
                nib.save(nib.Nifti1Image(plab_b[0,:,:,:].cpu().numpy().astype(np.float32), np.eye(4)), nii_save_path + "%02d_plab_b.nii.gz" % iter_num)                
                nib.save(nib.Nifti1Image(plab_a[0,:,:,:].cpu().numpy().astype(np.float32), np.eye(4)), nii_save_path + "%02d_plab_a.nii.gz" % iter_num)
                outputs_c_nii = F.softmax(outputs_c, dim=1)
                nii = outputs_c_nii[0,1,...].permute(2,0,1) # y
                nii =  (nii >= 0.5)
                nib.save(nib.Nifti1Image(nii[:,:,:].cpu().numpy().astype(np.float32), np.eye(4)), nii_save_path + "%02d_outputs_c.nii.gz" % iter_num)                                       	
                ins_width = 2
                B,C,H,W,D = unoutput_a.size()#outputs_l [B2, C2, H112, W112, D80]
                snapshot_img = torch.zeros(size = (D, 3, 3*H + 3 * ins_width, W + ins_width), dtype = torch.float32)
                #80,3,112*3+6,112+2
                

                snapshot_img[:,:, H:H+ ins_width,:] = 1    #1:white
                snapshot_img[:,:, 2*H + ins_width:2*H + 2*ins_width,:] = 1
                snapshot_img[:,:, 3*H + 2*ins_width:3*H + 3*ins_width,:] = 1
                snapshot_img[:,:, :,W:W+ins_width] = 1

                unoutput_a_soft = F.softmax(unoutput_a, dim=1)
                seg_out = unoutput_a_soft[0,1,...].permute(2,0,1) #
                seg_out =  (seg_out >= 0.5)
                target =  dont_use_unimg_a[0,...].permute(2,0,1)
                train_img = unimg_a[0,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                
                writer.add_images('Epoch_%d_Iter_%d_emaunoutputa'% (epoch, iter_num), snapshot_img)

                outputs_c_soft = F.softmax(outputs_c, dim=1)
                seg_out = outputs_c_soft[0,1,...].permute(2,0,1) # y
                seg_out =  (seg_out >= 0.5)
                target =  mixc_lab[0,...].permute(2,0,1)
                train_img = mixc_img[0,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_outputc'% (epoch, iter_num), snapshot_img)
                
                output_b_soft = F.softmax(unoutput_b, dim=1)
                seg_out = output_b_soft[0,1,...].permute(2,0,1) # y
                seg_out =  (seg_out >= 0.5)
                target =  dont_use_unimg_b[0,...].permute(2,0,1)
                train_img = unimg_b[0,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:, 0, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 1, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out
                snapshot_img[:, 2, 2*H+ 2*ins_width:3*H+ 2*ins_width,:W] = seg_out

                writer.add_images('Epoch_%d_Iter_%d_emaunoutputb'% (epoch, iter_num), snapshot_img)


            if iter_num >= self_max_iterations:
                break

        if iter_num >= self_max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    ## make logger file
    pre_snapshot_path = "/home/vigil/Desktop/BCP-main/model/BCP/LA_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "/home/vigil/Desktop/BCP-main/model/BCP/LA_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    print("Starting BCP training.")
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
    shutil.copy('./LA_BCP_train.py', self_snapshot_path)
    # -- Pre-Training
    #logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    #logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    #logging.info(str(args))
    #pre_train(args, pre_snapshot_path)
    # -- Self-training
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    
