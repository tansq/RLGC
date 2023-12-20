from iidnet import *
from torchvision import transforms
import torch
transform = transforms.Compose([
             np.float32,
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ])

def load_iid():
    giid_model = IID_Model().cuda()
    #giid_model.load(path='./latest/')
    giid_model.load()
    giid_model.eval()
    return giid_model


def iid_gra(iid, image, mask):
    image = image.astype('float') / 255.
    norm_image = torch.tensor(image, dtype=torch.float)
    for i in range(image.shape[0]):
        norm_image[i,:,:,:] = transform(image[i,:,:,:].transpose(1,2,0))
    mask = torch.from_numpy(mask).float()
    image = torch.autograd.Variable(norm_image, requires_grad=False)
    mo, lo = iid.process(image.cuda(),mask.cuda())
    return  mo.cpu().detach().numpy()


def get_loss_reward(pre_msk, cur_msk, gt, init_msk):
    pre_dist = np.power((pre_msk - gt),2)
    cur_dist = np.power((cur_msk - gt),2)
    reward = cur_dist - pre_dist
    #print(np.mean(reward))
    pre_False_Pred = np.round(pre_msk) != gt
    cur_False_Pred = np.round(cur_msk) != gt
    reward_weight_positive = np.logical_and(cur_False_Pred, (pre_False_Pred == False)).astype(np.float64)*100
    reward_weight_negative = np.logical_and(pre_False_Pred, (cur_False_Pred == False)).astype(np.float64)*100
    reward_weight_positive[reward_weight_positive==0] = 1
    reward_weight_negative[reward_weight_negative==0] = 1
    for i in range(reward.shape[0]):
        tmp = reward[i,0,:,:]
        r_max = tmp.max()
        r_min = tmp.min()
        r_mean = tmp.mean()
        norm_tmp = (tmp-r_mean)/(r_max-r_min)
        reward[i,0,:,:] = norm_tmp
    reward = reward * reward_weight_positive * reward_weight_negative
    pixel_wise_reward = np.ones([reward.shape[0],3,reward.shape[2],reward.shape[3]],dtype=np.float32)
    pixel_wise_reward[:,0,:,:] = reward[:,0,:,:]
    pixel_wise_reward[:,1,:,:] = reward[:,0,:,:]
    pixel_wise_reward[:,2,:,:] = reward[:,0,:,:]

    return pixel_wise_reward

def get_visual_reward(cur_img, pre_img, ori_img):
    cur_dist = np.power((cur_img - ori_img),2)
    pre_dist = np.power((pre_img - ori_img),2)
    reward = pre_dist - cur_dist
    return reward

def get_f1_and_iou_and_auc(gts,predicts):
    #print(gts)
    #print(predicts)
    f1, iou, auc = [],[],[]
    H,W,C = gts.shape[2],gts.shape[3],gts.shape[1]
    
    for i in range(gts.shape[0]):
        groundtruth = gts[i,:,:,:]
        predict_mask = predicts[i,:,:,:]
        auc.append(
                        roc_auc_score(
                            (groundtruth.reshape(H * W * C)).astype("int"),
                            predict_mask.reshape(H * W * C) ,
                        )
                    )
        
        predict_mask = np.round(predict_mask)
        predict_mask = predict_mask>0
        groundtruth = groundtruth>0
        seg_inv = np.logical_not(predict_mask)
        gt_inv = np.logical_not(groundtruth)
        true_pos = float(np.logical_and(predict_mask, groundtruth).sum())  # float for division
        true_neg = np.logical_and(seg_inv, gt_inv).sum()
        false_pos = np.logical_and(predict_mask, gt_inv).sum()
        false_neg = np.logical_and(seg_inv, groundtruth).sum()
        f1.append(2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6))
        cross = np.logical_and(predict_mask, groundtruth)
        union = np.logical_or(predict_mask, groundtruth)
        tmp_iou = np.sum(cross) / (np.sum(union) + 1e-6)
        if np.sum(cross) + np.sum(union) == 0:
            iou.append(1)
        else:
            iou.append(tmp_iou)
        
    return np.mean(f1),np.mean(iou), np.mean(auc)
            
