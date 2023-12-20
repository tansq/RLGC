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
EPISODE_LEN = 5
MAX_EPISODE = 5001
GAMMA = 0.8
N_ACTIONS = 27
BATCH_SIZE = 16
LR = 1e-3
img_size = 256
img_channel = 3
TRAINING_DATA_PATH = "./train.txt"
TESTING_DATA_PATH = "./test.txt"
SAVING_EPISODE = 500
IMAGE_DIR_PATH = ""
SAVE_PATH = "./models/"
def main():
    model = EfficientUnet(N_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    i_index = 0

    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        img_size,
        img_size)

    current_state = State.State((BATCH_SIZE, img_channel, img_size, img_size), MOVE_RANGE)
    agent = PixelWiseA3C_InnerState(model, optimizer, BATCH_SIZE, EPISODE_LEN, GAMMA)

    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    iid = load_iid()
    cur_f1 = 0
    cur_auc = 0
    for n_epi in tqdm(range(0, MAX_EPISODE), ncols=70, initial=0):
        r = indices[i_index: i_index + BATCH_SIZE]
        mask, path, raw_x = mini_batch_loader.load_training_data(r)
        action = np.zeros((BATCH_SIZE, img_size, img_size))
        current_state.reset(raw_x)
        reward = np.zeros(raw_x.shape, np.float32) 
        sum_reward = 0
        init_mask = iid_gra(iid, current_state.image, mask)
        current_mask = init_mask.copy()
        init_f1, init_iou, init_auc = get_f1_and_iou_and_auc(mask, init_mask)
        print("Init:  F1: "+str(init_f1)+" IOU: "+str(init_iou)+" AUC: "+str(init_auc))
        for t in range(EPISODE_LEN):
            previous_image = current_state.image.copy()
            previous_mask = current_mask.copy()
            action = agent.act_and_train(current_state.state, reward) 
            current_mask = iid_gra(iid, current_state.image, mask)
            reward = get_loss_reward(previous_mask, current_mask, mask, init_mask) + 0.01*get_visual_reward(current_state.image, previous_image, raw_x)
            
            print("Reward: "+str(np.mean(reward)))
            
            cur_f1, cur_iou, cur_auc = get_f1_and_iou_and_auc(mask, current_mask)
            print("Iter: "+str(t+1)+" F1: "+str(cur_f1)+" IOU: "+str(cur_iou)+" AUC: "+str(cur_auc))
            sum_reward += np.mean(reward) * np.power(GAMMA, t)
        agent.stop_episode_and_train(current_state.state, reward, True)
        if i_index + BATCH_SIZE >= train_data_size:
            i_index = 0
            indices = np.random.permutation(train_data_size)
        else:
            i_index += BATCH_SIZE

        if i_index + 2 * BATCH_SIZE >= train_data_size:
            i_index = train_data_size - BATCH_SIZE

        if n_epi % SAVING_EPISODE == 0:
            torch.save(model.state_dict(), SAVE_PATH+str(n_epi)+'.pth')
if __name__ == '__main__':
    main()
