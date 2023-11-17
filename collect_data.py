import yaml
import numpy as np
import copy
import argparse
import os
import cv2

from ppg.environment import Environment
from ppg.agent import PushGrasping
from ppg.utils.memory import ReplayBuffer

from object_comparison import get_grasped_object
from ppg.object_segmenter import ObjectSegmenter
import shutil

def collect_random_dataset(args):
    log_folder = 'logs_tmp'
    # Create a folder logs if it does not exist
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    # Create a buffer to store the data
    memory = ReplayBuffer(os.path.join(log_folder, 'ppg-dataset'))

    env = Environment(assets_root='assets/', objects_set='seen')
    env.singulation_condition = args.singulation_condition

    with open('yaml/bhand.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    policy = PushGrasping(params)
    policy.seed(args.seed)

    rng = np.random.RandomState()
    rng.seed(args.seed)

    segmenter = ObjectSegmenter()
    TRAIN_EPISODES_DIR = "save/misc/train/episodes"

    for j in range(args.n_samples):
        episode_seed = rng.randint(0, pow(2, 32) - 1)
        env.seed(episode_seed)
        obs = env.reset()
        print('Episode seed:', episode_seed)

        while not policy.init_state_is_valid(obs):
            obs = env.reset()

        id = 1
        processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], obs['depth'][id], dir=TRAIN_EPISODES_DIR, plot=True)

        for i in range(15):
            state = policy.state_representation(obs)

            try:
                # Select action
                action = policy.guided_exploration(state)
            except:
                obs = env.reset()
                continue
            
            env_action = policy.action(action)

            # Step environment.
            next_obs, grasp_info = env.step(env_action)

            if grasp_info['stable']:
                target_id, target_mask = get_grasped_object(processed_masks, action)

                if target_id != -1:
                    cv2.imwrite(os.path.join("save/misc/train", "target_mask.png"), target_mask)

                    transition = {'obs': obs, 'state': state, 'target_mask': target_mask, 'action': action, 'label': grasp_info['stable']}
                    memory.store(transition)

                    # processed_masks = new_processed_masks

            print(action)
            print(grasp_info)
            print('---------')

            if policy.terminal(obs, next_obs):
                break

            obs = copy.deepcopy(next_obs)
            processed_masks, pred_mask, raw_masks = segmenter.from_maskrcnn(obs['color'][id], obs['depth'][id], dir=TRAIN_EPISODES_DIR, plot=True)

        delete_episodes_misc(TRAIN_EPISODES_DIR)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_samples', default=10000, type=int, help='')
    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--singulation_condition', action='store_true', default=False, help='')
    return parser.parse_args()




def delete_episodes_misc(path):
    # Try to remove the tree; if it fails, throw an error using try...except.
    try:
        shutil.rmtree(path)
    except OSError as e:
        None

    if not os.path.exists(path):
        os.mkdir(path)

def recreate_train():
    train_path = "save/misc/train"

    # Try to remove the tree; if it fails, throw an error using try...except.
    try:
        shutil.rmtree(train_path)
    except OSError as e:
        None
        
    if not os.path.exists(f'{train_path}/episodes'):
        os.makedirs(f'{train_path}/episodes')


if __name__ == "__main__":
    args = parse_args()
    recreate_train()

    collect_random_dataset(args)
