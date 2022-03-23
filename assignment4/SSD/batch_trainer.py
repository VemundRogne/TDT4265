'''
Utility to automatically train multiple configs
'''

import train

train.train("configs/ssd300_small_and_more_anchors.py", False)
train.train("configs/ssd300_small_and_tall_anchors.py", False)