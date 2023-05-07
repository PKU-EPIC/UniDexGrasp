#!/usr/bin/env python3

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np
import tqdm
from tensorboard.backend.event_processing import event_accumulator


def find_all_files(root_dir, pattern):
    """Find all files under root_dir according to relative pattern."""
    file_list = []
    for dirname, _, files in os.walk(root_dir):
        for f in files:
            absolute_path = os.path.join(dirname, f)
            if re.match(pattern, absolute_path):
                file_list.append(absolute_path)
    return file_list


def group_files(file_list, pattern):
    res = defaultdict(list)
    for f in file_list:
        match = re.search(pattern, f)
        key = match.group() if match else ''
        res[key].append(f)
    return res


def csv2numpy(csv_file):
    csv_dict = defaultdict(list)
    reader = csv.DictReader(open(csv_file))
    for row in reader:
        for k, v in row.items():
            csv_dict[k].append(eval(v))
    return {k: np.array(v) for k, v in csv_dict.items()}


def convert_tfevents_to_csv(root_dir, alg_type, env_num, env_step, refresh=False):
    """Recursively convert test/rew from all tfevent file under root_dir to csv.

    This function assumes that there is at most one tfevents file in each directory
    and will add suffix to that directory.

    :param bool refresh: re-create csv file under any condition.
    """
    if alg_type == 'sarl':
        tfevent_files = find_all_files(root_dir, re.compile(r"^.*tfevents.*$"))
    elif alg_type == 'marl':
        tfevent_files = find_all_files(root_dir, re.compile(r"^.*tfevents.*.13$"))
    else:
        print("wrong alg_type!")


    print(f"Converting {len(tfevent_files)} tfevents files under {root_dir} ...")
    result = {}
    with tqdm.tqdm(tfevent_files) as t:
        for tfevent_file in t:
            t.set_postfix(file=tfevent_file)
            output_file = os.path.join(os.path.split(tfevent_file)[0], "test_rew.csv")

            if os.path.exists(output_file) and not refresh:
                content = list(csv.reader(open(output_file, "r")))
                if content[0] == ["env_step", "rew", "time"]:
                    for i in range(1, len(content)):
                        content[i] = list(map(eval, content[i]))
                    result[output_file] = content
                    continue

            
            ea = event_accumulator.EventAccumulator(tfevent_file)
            ea.Reload()
            initial_time = ea._first_event_timestamp
            content = [["env_step", "rew", "time"]]

            
            if alg_type == "sarl":
                for i, test_rew in enumerate(ea.scalars.Items("Train/mean_reward")):
                    content.append(
                        [
                            test_rew.step * env_step * env_num,   # if env is to lift a pot, change it as test_rew.step * 20 * 2048
                            round(test_rew.value, 4),
                            round(test_rew.wall_time - initial_time, 4),
                        ]
                    )
        
            elif alg_type == 'marl':
                for i, test_rew in enumerate(ea.scalars.Items("train_episode_rewards")):
                    content.append(
                        [
                            test_rew.step,
                            round(test_rew.value, 4),
                            round(test_rew.wall_time - initial_time, 4),
                        ]
                    )
                    
            csv.writer(open(output_file, 'w')).writerows(content)
            result[output_file] = content
    return result


def merge_csv(csv_files, root_dir, remove_zero=False):
    """Merge result in csv_files into a single csv file."""
    assert len(csv_files) > 0
    if remove_zero:
        for v in csv_files.values():
            if v[1][0] == 0:
                v.pop(1)

    sorted_keys = sorted(csv_files.keys())
    sorted_values = [csv_files[k][1:] for k in sorted_keys]
    content = [
        ["env_step", "rew", "rew:shaded"] +
        list(map(lambda f: "rew:" + os.path.relpath(f, root_dir), sorted_keys))
    ]

    for rows in zip(*sorted_values):
        array = np.array(rows)
        # assert len(set(array[:, 0])) == 1, (set(array[:, 0]), array[:, 0])
        # line = [rows[0][0], round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4)]
        line = [round(array[:, 0].mean(), 4), round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4)]
        line += array[:, 1].tolist()
        content.append(line)
    output_path = os.path.join(root_dir, f"test_rew_{len(csv_files)}seeds.csv")
    print(f"Output merged csv file to {output_path} with {len(content[1:])} lines.")
    csv.writer(open(output_path, "w")).writerows(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--alg-name',
        type=str,
        default='happo'
    )    
    parser.add_argument(
        '--alg-type',
        type=str,
        default='marl',
        help="single-agent: sarl; multi-agent: marl"
    )
    parser.add_argument(
        '--env-num',
        type=int,
        default=2048,
        help="the number of parallel simulations"
    )
    parser.add_argument(
        '--env-step',
        type=int,
        default=8,
        help="the environment lifting a pot : 20; other environments: 8"
    )
    parser.add_argument(
        '--refresh',
        action="store_true",
        help="Re-generate all csv files instead of using existing one."
    )
    parser.add_argument(
        '--remove-zero',
        action="store_true",
        help="Remove the data point of env_step == 0."
    )
    parser.add_argument('--root-dir', type=str)
    args = parser.parse_args()
    
    args.root_dir = '{}/{}'.format(args.root_dir,args.alg_name)

    csv_files = convert_tfevents_to_csv(args.root_dir, args.alg_type, args.env_num, args.env_step, args.refresh)
    merge_csv(csv_files, args.root_dir, args.remove_zero)
