import argparse
import re
import os.path as path
from pathlib import Path

import matplotlib.pyplot as plt
from statistics import mean

import numpy as np


def parse_stats_double_number(file_name, take=None):
    with open(file_name) as file:
        lines = file.readlines()

        if take is not None:
            lines = lines[:take]

        numbers_regex = r'([\d.\-e]+) ([\d.\-e]+)'

        if re.match(numbers_regex, lines[0]) is None:
            numbers_regex = r'([\d.\-e]+)'

        points = []
        for line in lines:
            matched = re.match(numbers_regex, line)
            points.append(list(map(float, matched.groups())))

        points = np.vstack(points)
        _, name = path.split(file_name)
        plt.plot(range(len(points)), points, label=[f'{name}: train', f'{name}: test'])
        # l1.set_label(f'{file_name}: train')
        # l2.set_label(f'{file_name}: train')
        # plt.legend([l1, l2], [f'{file_name}: train', f'{file_name}: train'])
        # plt.legend()
        # plt.show()


def parse_stats_and_plot(file_name, take=None):
    numbers_regex = r'[\d.\-e ]+'

    with open(file_name) as file:
        line = file.readline()

        if re.match(numbers_regex, line) is not None:
            parse_stats_double_number(file_name, take)
        else:
            parse_stats_and_plot_my_format(file_name, take)


def parse_stats_and_plot_my_format(file_name, take=None):
    with open(file_name) as file:
        lines = file.readlines()

        if take is not None:
            lines = lines[:take]

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
        # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='stats.txt')
    parser.add_argument("--take", type=int, default=None)
    parser.add_argument("--second_path", type=str, default=None)

    args = parser.parse_args()

    parse_stats_and_plot(Path() / args.path, args.take)

    if args.second_path is not None:
        parse_stats_and_plot(Path() / args.second_path, args.take)

    plt.legend()
    plt.show()
