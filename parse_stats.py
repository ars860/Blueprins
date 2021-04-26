import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from statistics import mean

import numpy as np


def parse_stats_double_number(file_name):
    with open(file_name) as file:
        lines = file.readlines()
        numbers_regex = r'([\d.\-e]+) ([\d.\-e]+)'

        if re.match(numbers_regex, lines[0]) is None:
            numbers_regex = r'([\d.\-e]+)'

        points = []
        for line in lines:
            matched = re.match(numbers_regex, line)
            points.append(list(map(float, matched.groups())))

        points = np.vstack(points)
        plt.plot(range(len(points)), points)
        plt.show()


def parse_stats_and_plot(file_name):
    numbers_regex = r'[\d.\-e ]+'

    with open(file_name) as file:
        line = file.readline()

        if re.match(numbers_regex, line) is not None:
            parse_stats_double_number(file_name)
        else:
            parse_stats_and_plot_my_format(file_name)


def parse_stats_and_plot_my_format(file_name):
    with open(file_name) as file:
        lines = file.readlines()

        line_regex = r'^Epoch:(\d+)/\d+, Step:\d+/\d+, Loss:([\d.]+)$'
        comments = []
        losses = []
        epochs_cnt = 0

        cur_epoch, cur_losses = -1, []
        for i, line in enumerate(lines):
            if line == "\n":
                continue

            try_match = re.match(line_regex, line)
            if try_match is not None:
                epoch, loss = try_match.groups()
                if epoch != cur_epoch:
                    if len(cur_losses) > 0:
                        losses.append(mean(cur_losses))
                    cur_epoch, cur_losses = epoch, []
                    epochs_cnt += 1

                cur_losses.append(float(loss))
            else:
                comments.append((line, epochs_cnt))

        print(comments)
        # print(losses)
        plt.plot(range(len(losses)), losses)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='stats.txt')

    args = parser.parse_args()

    parse_stats_and_plot(Path() / args.path)
