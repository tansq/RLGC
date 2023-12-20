import numpy as np
import sys
import cv2
action_map = {0:[0,0,0],1:[0,0,1],2:[0,0,-1],
                3:[0,1,0],4:[0,1,1],5:[0,1,-1],
                6:[0,-1,0],7:[0,-1,1],8:[0,-1,-1],
                9:[1,0,0],10:[1,0,1],11:[1,0,-1],
                12:[1,1,0],13:[1,1,1],14:[1,1,-1],
                15:[1,-1,0],16:[1,-1,1],17:[1,-1,-1],
                18:[-1,0,0],19:[-1,0,1],20:[-1,0,-1],
                21:[-1,1,0],22:[-1,1,1],23:[-1,1,-1],
                24:[-1,-1,0],25:[-1,-1,1],26:[-1,-1,-1]}
class State():
    def __init__(self, size, move_range):
        self.state = np.zeros(size, dtype=np.float32)
        self.move_range = np.float32(move_range)
        self.image = np.zeros(size, dtype=np.float32)
        
    def reset(self, x):
        self.image = x
        self.state = np.clip(x/255., a_min=0., a_max=1.)
        
    def step(self, act):
        move_map = np.zeros(self.image.shape, dtype = np.float32)
        for i in range(np.int(self.move_range)):
            if i == 0:
                continue
            tmp = np.ones(self.image.shape, np.float32)
            tmp[:,0,:,:] = tmp[:,0,:,:]*action_map[i][0]
            tmp[:,1,:,:] = tmp[:,1,:,:]*action_map[i][1]
            tmp[:,2,:,:] = tmp[:,2,:,:]*action_map[i][2]
            flag = np.zeros(act.shape, np.float32)
            flag[act==i] = 1
            flag = flag[:,np.newaxis,:,:]
            flag = flag.repeat(3,1)
            move = np.multiply(tmp,flag)
            self.image = self.image+move
            self.image = np.clip(self.image, a_min=0., a_max=255)
            self.state = self.image/255.
            move_map += move
        return move_map
