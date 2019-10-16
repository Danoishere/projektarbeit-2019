
import numpy as np

def augment_with_last_frames(params, num_agents, new_obs, episode_buffers):
        # See network for construction of obs
        single_obs_len = params.frame_size
        full_obs_len = params.vec_tree_state_size
        all_obs = []
        for i in range(num_agents):
            episode_buffer = episode_buffers[i]
            # Get last observation (n frames) for this agent
            if len(episode_buffer) == 0:
                last_obs = np.zeros(single_obs_len)
            else:
                last_obs = episode_buffer[-1][0]

            # Remove last frame from the last frame -> l = n-1
            last_obs_n = last_obs[:single_obs_len]

            obs = np.zeros(full_obs_len)

            # Start = new obs
            obs[:single_obs_len] = new_obs[i]

            # Fill remaining n-1 slots with last obs
            obs[single_obs_len:single_obs_len+len(last_obs_n)] = last_obs_n
            all_obs.append(obs)
        
        return np.vstack(all_obs).astype(np.float32)