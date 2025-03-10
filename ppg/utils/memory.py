from random import Random
import numpy as np
import pickle
import os
import cv2


class ReplayBuffer:
    def __init__(self, log_dir, buffer_size=100000):
        self.buffer_size = buffer_size
        self.random = Random()
        self.count = 0
        self.log_dir = log_dir
        self.buffer_ids = []

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.z_fill = 6

    def __call__(self, index):
        return self.buffer[index]

    def store(self, transition):
        folder_name = os.path.join(self.log_dir, 'transition_' + str(self.count).zfill(5))
        if not os.path.exists(folder_name):
            # raise Exception
            os.mkdir(folder_name)

        cv2.imwrite(os.path.join(folder_name, 'heightmap.exr'), transition['state'])
        cv2.imwrite(os.path.join(folder_name, 'target_mask.png'), transition['target_mask'])
        pickle.dump(transition['action'], open(os.path.join(folder_name, 'action'), 'wb'))

        # Save everything that obs contains
        for i in range(len(transition['obs']['color'])):
            cv2.imwrite(os.path.join(folder_name, 'color_' + str(i) + '.png'), transition['obs']['color'][i])
            cv2.imwrite(os.path.join(folder_name, 'depth_' + str(i) + '.exr'), transition['obs']['depth'][i])
            cv2.imwrite(os.path.join(folder_name, 'seg_' + str(i) + '.png'), transition['obs']['seg'][i])
        pickle.dump(transition['obs']['full_state'], open(os.path.join(folder_name, 'full_state'), 'wb'))

        self.buffer_ids.append(self.count)
        if self.count < self.buffer_size:
            self.count += 1

    def sample(self, given_batch_size=1):
        if self.count < given_batch_size:
            batch_size = self.count
        else:
            batch_size = given_batch_size

        batch_id = self.random.sample(self.buffer_ids, 1)[0]

        folder_name = os.path.join(self.log_dir, 'transition_' + str(batch_id).zfill(5))
        state = cv2.imread(os.path.join(folder_name, 'heightmap.exr'), -1)
        action = pickle.load(open(os.path.join(folder_name, 'action'), 'rb'))

        return state, action

    def clear(self):
        self.buffer_ids.clear()
        self.count = 0

    def size(self):
        return self.count

    def seed(self, random_seed):
        self.random.seed(random_seed)