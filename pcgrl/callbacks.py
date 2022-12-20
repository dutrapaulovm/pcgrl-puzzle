import os
from pcgrl.log import ResultsWriter
from abc import ABC

class BasePCGRLCallback(ABC):
    """
    Base class for callback.

    :param verbose:
    """
    def __init__(self):
        super(BasePCGRLCallback, self).__init__()
        # Number of time the callback was called
        self.n_calls = 0  # type: int        

    def on_create_env(self):
        """
        This method is called when the env is created
        """
        self._on_create_env()

    def  _on_create_env(self):
        pass
    
    def _on_reset(self):        
        return True
    
    def on_after_step(self, actions, reward, done, info, hist):
        self._on_after_step(actions, reward, done, info, hist)
            
    def _on_after_step(self, actions, reward, done, info, hist):
        pass
    
    def on_before_step(self, actions):
        self._on_before_step(actions)    
    
    def _on_before_step(self, actions):
        pass    
    
    def on_reset(self):
        self._on_reset()
    
    def _on_step(self):
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self):
        """
        This method will be called by the model after each call to ``env.step()``.
        """
        self.n_calls += 1        

        return self._on_step()

PCGRLCallback = BasePCGRLCallback()

class InfoWriterPCGRLPuzzleCallback(BasePCGRLCallback):

    def __init__(self, path = None):
        super(InfoWriterPCGRLPuzzleCallback, self).__init__()
        self.columnsnames = {"reward_game",  "reward",  "discount_reward",  "bonus_factor",  "experience_bonus", "done_bonus", "done_penalty", "reward_experience_bonus", "changes_penalty",  "piece_penalty", "counter_changes", "counter_done", "representation", "counter_done_interations", "counter_done_max_changes", "is_done_success", "agent", "segments", "entropy", "entropy_map","historical", 'rewards_sum', 'discount_rewards'}
        self.results_writer = None
        self.path = path
        self.save_logger = True
                                
    def _on_create_env(self):        
        if self.path  is not None:
            self.results_writer = ResultsWriter(
                filename="Info.csv",
                path=self.path,                 
                fieldsnames=self.columnsnames
            )
        else:
            self.results_writer = None
    
    def _on_after_step(self, reward, done, info, hist):
        if (done):
            
            info['rewards_sum'] = sum(hist['rewards'])            

            if (self.save_logger):                                      
                if (self.path is None or not os.path.exists(self.path)):
                    raise ValueError('Set a valid path to save logger.')

                self.results_writer.write_row(info)
                    
    def _on_before_step(self, actions):
        pass
    
    def _on_step(self):    
        return True
    
    def _on_reset(self):
        return True