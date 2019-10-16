import numpy as np
import code_util.constants as const
import json
import os

class CheckpointManager:
    '''
    - Saves checkpoint on every curriculum-level (suffix checkpoint_lvl_x)
    - Saves best model for every curriculum-level (suffix best_lvl_x)
    - Saves curriculum-lvl, episod-nr, best reward in checkpoint.json
    '''

    def __init__(self, state, global_model, save_best_after_min=2, save_ckpt_after_min=2):
        self.global_model = global_model
        self.state = state
        self.best_reward = -np.inf
        self.last_curriculum_level = 0
        self.last_save_best_on_episode_nr = 0
        self.last_ckpt_on_episode_nr = 0
        self.save_best_after_min = save_best_after_min
        self.save_ckpt_after_min = save_ckpt_after_min

        if not os.path.exists(const.model_path):
            os.makedirs(const.model_path)


    def does_file_exist(self, file_name):
        return os.path.exists(file_name)


    def load_checkpoint_model(self):
        if self.does_file_exist(const.checkpoint_file):
            with open(const.checkpoint_file, 'r') as f:  
                last_training = json.load(f)

            self.best_reward = last_training['best_reward']
            self.last_ckpt_on_episode_nr = last_training['last_ckpt_on_episode_nr']
            self.state.curriculum_level = last_training['curriculum_level']
            
            curr_level = str(self.state.curriculum_level)
            self.global_model.load_model(const.model_path, const.suffix_checkpoint +'_lvl_'+ curr_level)

            return self.last_ckpt_on_episode_nr
        else:
            raise ValueError('No checkpoint.json found at', const.checkpoint_file)


    def try_save_model(self, episode_nr, reward):
        # Reset best reward on curriculum-level-change
        if self.last_curriculum_level < self.state.curriculum_level:
            self.last_curriculum_level = self.state.curriculum_level
            self.best_reward = 0
            
        if reward > self.best_reward:
            # Save a best model not more than every 'save_best_after_min' steps
            if self.last_save_best_on_episode_nr + self.save_best_after_min <=  episode_nr:
                self.last_save_best_on_episode_nr = episode_nr
                self.best_reward = reward
                curr_level = str(self.state.curriculum_level)
                self.global_model.save_model(const.model_path, const.suffix_best +'_lvl_'+ curr_level)

        # Save ckpt every 'save_best_after_min' steps
        if episode_nr % self.save_ckpt_after_min == 0:
            curr_level = str(self.state.curriculum_level)
            self.global_model.save_model(const.model_path, const.suffix_checkpoint +'_lvl_'+ curr_level)
            with open(const.checkpoint_file, 'w') as f:  
                json.dump({
                    'curriculum_level' : self.state.curriculum_level,
                    'best_reward' : float(self.best_reward),
                    'last_ckpt_on_episode_nr' : int(episode_nr)
                }, f)



