import numpy as np
import code_util.constants as const
import json
import os


class CheckpointManager:
    def __init__(self, global_model, authorized_worker, save_best_after_min=2, save_ckpt_after_min=4):
        self.global_model = global_model
        self.best_reward = -np.inf
        self.last_save_best_on_episode_nr = 0
        self.last_ckpt_on_episode_nr = 0
        self.save_best_after_min = save_best_after_min
        self.save_ckpt_after_min = save_ckpt_after_min
        self.authorized_worker = authorized_worker

        if not os.path.exists(const.model_path):
            os.makedirs(const.model_path)

    def does_file_exist(self, file_name):
        return os.path.exists(file_name)

    def load_best_model(self):
        self.global_model.load_model(const.model_path, 'best')

    def load_checkpoint_model(self):
        if self.does_file_exist(const.checkpoint_file):
            last_training = json.load(const.checkpoint_file)
            self.best_reward = last_training['best_reward']
            self.last_ckpt_on_episode_nr = last_training['last_ckpt_on_episode_nr']
            self.global_model.load_model(const.model_path, 'checkpoint')
            return self.last_ckpt_on_episode_nr
        else:
            raise ValueError('No checkpoint.json found')

    def try_save_model(self, episode_nr, reward, worker_name):
        if worker_name != self.authorized_worker:
            return

        if reward > self.best_reward:
            if self.last_save_best_on_episode_nr + self.save_best_after_min <  episode_nr:
                self.best_reward = reward
                self.last_save_best_on_episode_nr = episode_nr
                self.global_model.save_model(const.model_path, 'best')

        elif self.last_ckpt_on_episode_nr + self.save_ckpt_after_min <  episode_nr:
            self.last_ckpt_on_episode_nr = episode_nr
            self.global_model.save_model(const.model_path, 'checkpoint')
            with open(const.checkpoint_file, 'w') as outfile:  
                json.dump({
                    'best_reward' : self.best_reward,
                    'last_ckpt_on_episode_nr' : episode_nr,
                }, outfile)



