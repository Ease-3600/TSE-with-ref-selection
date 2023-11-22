#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/6/29 21:08
# @Author : EASE
# @Version：V 0.1
# @File : pipeline_sd
# @desc : 两种模式，一种是直接运行，根据csv输出结果rttm；另一种是函数，输入音频路径，输出结果

from pyannote.audio import Pipeline
import argparse
import pandas as pd
import numpy as np
import os
import pdb


def sd_from_path(audio_path):
    diarization = pipeline(audio_path)
    return diarization

def sd_from_df(df_md, rttm_dir):
    for _, row in df_md.iterrows():
        key = row["mixture_ID"]
        mix_path = row["mixture_path"]
        diarization = sd_from_path(mix_path)
        with open(os.path.join(rttm_dir, f"{key}.rttm"), "w") as rttm:
            diarization.write_rttm(rttm)


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadate_dir', type=str, default="/data1/LibriSpeech/LibriCse/Cse2Mix/60/metadata/mixture_train-100_mix_clean.csv",
                        help='Path to librispeech root directory')
    args = parser.parse_args()
    df_md = pd.read_csv(args.metadate_dir)
    '''
    for n_src in [2]:
        for overlap in [30]:
            metadate_dir = f"/home/lsh/zyr/LibriCse/Cse{n_src}Mix/{overlap}/metadata/mixture_train-100_mix_clean.csv"
            df_md = pd.read_csv(metadate_dir)
            rttm_dir = f"./sd3/osd/cse{n_src}mix{overlap}ov"
            os.makedirs(rttm_dir, exist_ok=True)
            sd_from_df(df_md, rttm_dir)
            print(f"cse{n_src}mix{overlap}ov processed")
