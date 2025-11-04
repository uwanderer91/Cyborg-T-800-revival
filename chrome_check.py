from math import *
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.mouse_button import MouseButton
from selenium.webdriver import Keys, ActionChains
import numpy as np
import matplotlib.pyplot as plt
import cv2

chrome_options = Options()
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
driver = webdriver.Chrome(options=chrome_options)
restart_button = driver.find_element(By.ID, "game-over-continue")
restart_button.click()
time.sleep(1)
play_button = driver.find_element(By.ID, "spawn-button")
play_button.click()
time.sleep(1)

canvas = driver.find_element(By.ID, "canvas")
canvas_size = canvas.size
center_x = canvas_size["width"]//2
center_y = canvas_size["height"]//2

action_builder = ActionBuilder(driver, duration=2)
action_chain = ActionChains(driver)
    
# for i in range(0, 45):
#     rad_vec_len = 20
#     action_builder.pointer_action.move_to_location(center_x+rad_vec_len*cos(radians(i*8)), center_y+rad_vec_len*sin(radians(i*8)))
#     action_builder.perform()

# for i in range(0, 500):
#     action_chain.key_down("a")
#     action_chain.perform()
#     time.sleep(0.02)
#     action_chain.key_up("a")
#     action_chain.perform()

for i in range(0, 1000):
    screen = canvas.screenshot_as_png
    image_array = cv2.imdecode(np.frombuffer(screen, np.uint8), cv2.IMREAD_COLOR)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    # plt.imshow(image_array)
    # plt.axis('off')
    # plt.show()
