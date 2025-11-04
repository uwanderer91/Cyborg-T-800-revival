from vizdoom import * 
import random
import time 
from gym import Env
from gym.spaces import Discrete, Box
import cv2
import numpy as np
class VizDoomGym(Env): 
    def __init__(self, render=False, config='vizdoom/ViZDoom/scenarios/take_cover.cfg'): 
        # Inherit from Env
        super().__init__()
        # Setup the game 
        self.game = DoomGame()
        self.game.load_config(config)
        
        if render == False: 
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        
        self.game.init()
        
        self.observation_space = Box(low=0, high=255, shape=(60,60,1), dtype=np.uint8) 
        self.action_space = Discrete(9)
        
        # Game variables: HEALTH DAMAGE_TAKEN HITCOUNT SELECTED_WEAPON_AMMO
        self.health = 0
        self.hitcount = 0
        self.ammo = 52 ## CHANGED
        
        
    def step(self, action):
        actions = np.identity(self.action_space.n-1)
        rew = 0
        if action != 0:
            rew = self.game.make_action(actions[action-1], 4)
        else:
            rew = self.game.make_action([0, 0, 0, 0, 0, 0, 0, 0], 4)
        #if movement_reward < 0:
        #    movement_reward *= 2
        
        reward = 0 
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            
            game_variables = self.game.get_state().game_variables
            health, hitcount, ammo = game_variables
            
            health_delta = health - self.health
            self.health = health
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.ammo
            self.ammo = ammo

            if health_delta > 0:
                health_delta = health_delta*4
            
            reward = rew*6 + health_delta*50 + hitcount_delta*1400 + ammo_delta*250
            info = ammo
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 
        
        info = {"info":info}
        done = self.game.is_episode_finished()
        
        return state, reward, done, info 
    
    def render(self): 
        pass
    
    def reset(self): 
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)
    
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        gray = gray[90:200, 0:320]
        resize = cv2.resize(gray, (64,64), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (64,64,1))
        return state
    
    def close(self):
        self.game.close()