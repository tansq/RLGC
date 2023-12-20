import torch
import numpy as np
import cv2
from tqdm import tqdm
import State as State
from pixelwise_a3c import *
from FCN import *
from reward import *
from mini_batch_loader import MiniBatchLoader
import matplotlib.pyplot as plt
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

MOVE_RANGE = 27
EPISODE_LEN = 3
TESTING_MAX_EPISODE_LEN = 10
GAMMA = 0.8
N_ACTIONS = 27
BATCH_SIZE = 1
LR = 1e-3
img_size = 256
img_channel = 3
TRAINING_DATA_PATH = "./train.txt"
TESTING_DATA_PATH = "./samples.txt"
IMAGE_DIR_PATH = ""
SAVE_PATH = "./models/RLGC.pth"


def main():
    model = EfficientUnet(N_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    iid = load_iid()

    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        img_size,
        img_size)

    current_state = State.State((BATCH_SIZE, img_channel, img_size, img_size), MOVE_RANGE)
    agent = PixelWiseA3C_InnerState(model, optimizer, BATCH_SIZE, EPISODE_LEN, GAMMA)

    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    indices = np.random.permutation(test_data_size)
    
    best_f1 = 1

    best_img = np.zeros((BATCH_SIZE, img_channel, img_size, img_size), dtype=np.float32)
    best_msk = np.zeros((BATCH_SIZE, 1, img_size, img_size), dtype=np.float32)
    for i_index in tqdm(range(0, test_data_size), ncols=70, initial=0):
        best_f1 = 1
        best_iou = 1
        best_auc = 1
        r = indices[i_index: i_index + BATCH_SIZE]

        mask, path, raw_x = mini_batch_loader.load_testing_data(r)
        img_name = path[0].split('/')[-1]
        action = np.zeros((BATCH_SIZE, img_size, img_size))
        current_state.reset(raw_x)
        init_mask = iid_gra(iid, current_state.image, mask)
        current_mask = init_mask.copy()
        init_f1, init_iou, init_auc = get_f1_and_iou_and_auc(mask, init_mask)
        best_f1 = init_f1

        best_f1 = init_f1 
        print("Init:  F1: "+str(init_f1)+" IOU: "+str(init_iou)+" AUC: "+str(init_auc))
        best_img = current_state.image.copy()
        best_msk = init_mask.copy()
        for t in range(TESTING_MAX_EPISODE_LEN):
            previous_image = current_state.image.copy()
            previous_mask = current_mask.copy()
            action = agent.act(current_state.state) #只返回 action 足够
            move_map = current_state.step(action)
            current_mask = iid_gra(iid, current_state.image, mask)
            current_f1, current_iou, current_auc = get_f1_and_iou_and_auc(mask, current_mask)
            if  (current_f1 >best_f1 or np.abs(current_f1 - best_f1)<0.02) and t>=2:
                print("Early Stop:"+str((current_f1)))
                break
            print("Iter: "+str(t+1)+" F1: "+str(current_f1)+" IOU: "+str(current_iou)+" AUC: "+str(current_auc)) 
            if best_f1>current_f1:
                pre_best_f1 = best_f1
                best_f1 = current_f1
                best_iou = current_iou
                best_auc = current_auc
                best_img, best_msk = current_state.image.copy(), current_mask.copy()
        img_name = path[0].split('/')[-1]
            
        cv2.imwrite("./iid_early_stop/original/img/"+img_name,raw_x[0,:,:,:].transpose(1,2,0)[..., ::-1])
        cv2.imwrite("./iid_early_stop/original/gt/"+img_name, mask[0,0,:,:]*255.)
        cv2.imwrite("./iid_early_stop/original/msk/"+img_name, np.round(init_mask[0,0,:,:])*255.)
        cv2.imwrite("./iid_early_stop/adversarial/img/"+img_name, best_img[0,:,:,:].transpose(1,2,0)[..., ::-1])
        cv2.imwrite("./iid_early_stop/adversarial/msk/"+img_name, np.round(best_msk[0,0,:,:])*255.)
    
if __name__ == '__main__':
    main()
