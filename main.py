import pulp, json, itertools, time
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from tqdm import trange, tqdm
from argparse import ArgumentParser, Namespace
from utils import to_directed, rank_by_order, randomzie, phone_mapping, solve_LP_static_G, solve_LP_temporal_G

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default="./output/")
    parser.add_argument("--input_dir", type=Path, default="./input/")

    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--randomize", type=bool, default=False)
    parser.add_argument("--G_type", type=str, default="static", choices=["static", "temporal"])

    parser.add_argument("--freq", type=str, default="", choices=["", "date", "half_day", "hour"])

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    # Read File
    df = pd.read_csv(args.input_dir/"input.csv")
    df["Time"] = df["Time"].apply(pd.to_datetime)

    if not args.randomize:
        if args.G_type == "static":
            ranking = solve_LP_static_G(phone_mapping(df))
        elif args.G_type == "temporal":
            ranking = solve_LP_temporal_G(phone_mapping(df), args.freq)
    else:
        ranking_lst = []
        for i in trange(args.iters):
            if args.G_type == "static":
                rand_df = pd.concat([randomzie(df[df["Type"] == "Voice"]), randomzie(df[df["Type"] == "SMS"])])
                output = solve_LP_static_G(phone_mapping(rand_df))
                ranking_lst.append(output.T)
            else:
                rand_df = randomzie(df[df["Type"] == "Voice"])
                output = solve_LP_temporal_G(phone_mapping(rand_df), args.freq)
                ranking_lst.append(output.T)
        ranking = pd.concat(ranking_lst)
    ranking.to_csv(args.output_dir/"{}_{}_ranomize_{}.csv".format(args.G_type, args.freq, args.randomize))
