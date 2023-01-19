import tcod
import gym
import numpy as np
from .utils import Utils

class Visualization:
    @staticmethod
    def visualize(env_name, env_shape, policy):
        if env_name == "CliffWalking-v0":
            rows, cols = env_shape
            buffer = np.zeros(
                shape=(rows+2, cols+2),
                dtype=tcod.console.Console.DTYPE,
                order="C",
            )
            for i in range(rows+2):
                for j in range(cols+2):
                    if i == 0 and j == 0:
                        buffer[i][j] = ord(Utils.get_CP437_ch(201))
                    elif i == rows+1 and j == 0:
                        buffer[i][j] = ord(Utils.get_CP437_ch(200))
                    elif i == 0 and j == cols+1:
                        buffer[i][j] = ord(Utils.get_CP437_ch(187))
                    elif i == rows+1 and j == cols+1:
                        buffer[i][j] = ord(Utils.get_CP437_ch(188))
                    elif i == 0 or i == rows+1:
                        buffer[i][j] = ord(Utils.get_CP437_ch(205))
                    elif j == 0 or j == cols+1:
                        buffer[i][j] = ord(Utils.get_CP437_ch(186))
                    elif i == rows and j not in {1, cols}:
                        buffer[i][j] = ord(Utils.get_CP437_ch(223))
                    elif i == rows and j == cols:
                        buffer[i][j] = ord(Utils.get_CP437_ch(1))
                    else:
                        state = (i-1)*cols + (j-1)
                        action = np.argmax(policy[state])
                        arrow_idx = 24
                        if state == 36:
                            print(f"action in state 36: {action}")
                        if action == 0:
                            arrow_idx = 24
                        elif action == 1:
                            arrow_idx = 26
                        elif action == 2:
                            arrow_idx = 25
                        elif action == 3:
                            arrow_idx = 27
                        buffer[i][j] = ord(chr(tcod.tileset.CHARMAP_CP437[arrow_idx]))
            print(tcod.console.Console(rows+2, cols+2, order="C", buffer=buffer))