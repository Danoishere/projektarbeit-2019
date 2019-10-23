class RunStatistics:
    def __init__(self,num_agents,num_agents_done,total_reward,width,height,steps_needed,evaluation_round,trained_on_curriculum_level):
        self.num_agents = num_agents
        self.num_agents_done = num_agents_done
        self.total_reward = total_reward
        self.width = width
        self.height = height
        self.evaluation_round = evaluation_round
        self.steps_needed = steps_needed,
        self.trained_on_curriculum_level = trained_on_curriculum_level

    def to_dict(self):
        return {
            'num_agents' : self.num_agents,
            'num_agents_done' : self.num_agents_done,
            'total_reward' : self.total_reward,
            'width' : self.width,
            'height' : self.height,
            'evaluation_round' : self.evaluation_round,
            'steps_needed' : self.steps_needed,
            'trained_on_curriculum_level' : self.trained_on_curriculum_level
        }

    def __str__(self):
        return str(self.to_dict())