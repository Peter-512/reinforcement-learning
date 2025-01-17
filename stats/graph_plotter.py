import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def clear_directory(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory '{directory}' does not exist.")
        return

    # List all files in the directory
    files = os.listdir(directory)

    # Loop through the files and remove them
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete: {file_path} ({e})")


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        try:
            os.mkdir(directory)
            print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Failed to create directory: {directory} ({e})")


class Stats:
    def __init__(self, size: list, episodes_per_iteration: int = 200):
        if len(size) != 2:
            raise ValueError("Size must be a list of length 2")
        self.size = size
        self.name = "quiver"
        self.graph_count = 0
        self.e = episodes_per_iteration
        create_directory_if_not_exists("./screenshots")
        clear_directory("./screenshots")

    def generate_quiver_from_data(self, data: np.ndarray, map_arr: np.ndarray):
        global u, v
        fig, ax = plt.subplots(figsize=(data.shape[1], data.shape[0]))
        # add extra left margin to grid
        ax.set_xlim(-0.5, data.shape[1] - .5)
        ax.set_ylim(-0.5, data.shape[0] - .5)

        arrow_length = 1.2

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = 'blue'
                if map_arr[i, j] == b'H':
                    color = 'red'
                elif map_arr[i, j] == b'G':
                    color = 'green'

                direction = data[i, j]

                if direction == 0:
                    u = -arrow_length
                    v = 0
                elif direction == 1:
                    u = 0
                    v = -arrow_length
                elif direction == 2:
                    u = arrow_length
                    v = 0
                elif direction == 3:
                    u = 0
                    v = arrow_length

                x = j
                y = data.shape[0] - 1 - i

                ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1.5, color=color, pivot='mid', width=0.02,
                          headwidth=4)
        plt.title(f"Episode Nr. {self.graph_count * self.e}")
        plt.savefig(f"./screenshots/{self.graph_count}_policy.png")
        plt.close()
        self.graph_count += 1

    def generate_final_policy_gif(self, data: np.ndarray, map_arr: np.ndarray, name: str = 'output'):
        self.generate_quiver_from_data(data, map_arr)
        self.generate_gif(name)
        clear_directory("./screenshots")

    def generate_gif(self, name: str):
        png_files = os.listdir("./screenshots")

        # Create a list to store each frame (PNG image)
        frames = []

        # Load each PNG file and append it to the frames list
        for file in png_files:
            img = Image.open(f'./screenshots/{file}')
            frames.append(img)
        frames.sort(key=lambda x: int(x.filename.split("_")[0].split("/")[2]))
        # Set the output GIF file name and duration between frames (in milliseconds)
        create_directory_if_not_exists("./runs")
        output_gif = f"./runs/{name}.gif"
        # set duration variable using the number of episodes per iteration
        duration = self.e * 2

        # Save the frames as an animated GIF
        frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration, loop=0)
