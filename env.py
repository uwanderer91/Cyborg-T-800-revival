from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.mouse_button import MouseButton
from selenium.webdriver import Keys, ActionChains
import random
import time
from gym import Env
from gym.spaces import Discrete, Box
import cv2
import numpy as np

class DiepEnv(Env):
    def __init__(self):
        super().__init__()
        chrome_options = Options()
        chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
        self.driver = webdriver.Chrome(options=chrome_options)

    def step(self, action):
        
        pass

    def render(self): 
        pass

    def reset(self):
        restart_button = self.driver.find_element(By.ID, "game-over-continue")
        restart_button.click()
        time.sleep(1)
        play_button = self.driver.find_element(By.ID, "spawn-button")
        play_button.click()
        time.sleep(1)

        self.canvas = self.driver.find_element(By.ID, "canvas")
        return self.get_obs()

    def get_scr(self):
        screen = self.canvas.screenshot_as_png
        image_array = cv2.imdecode(np.frombuffer(screen, np.uint8), cv2.IMREAD_COLOR)
        return image_array
    
    def get_obs(self):
        image_array = self.get_scr()
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray,190,255,cv2.THRESH_BINARY)[1]
        gray = gray[:, 250:-250]
        gray = cv2.resize(gray, (128,128))
        gray = np.reshape(gray, (128,128,1))
        return gray

    def close(self):
        pass