'''
Utility to automatically train multiple configs
'''

import train

train.train("configs/a_ssd300.py", False)
train.train("configs/a_ssd300_small_anchors.py", False)
train.train("configs/a_ssd300_small_and_tall_anchors.py", False)