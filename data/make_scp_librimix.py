#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2022/9/23 17:48
# @Author : EASE
# @Version：V 0.1
# @File : make_scp.py
# @desc : 创建Libri2Mix数据集的scp文件

import os
import argparse
import pandas as pd
import pdb
import random

SPK_LIST = list()
EMPTY_PATH = "./data/empty.wav"
fix_rate=True

def choose_samespk(datatype, audioid):
    # 根据audioid，从LibriSpeech中找到同一人的不同语音，返回绝对路径
    spk_id = audioid.split("-")[0]
    if spk_id not in SPK_LIST:
        SPK_LIST.append(spk_id)
    files_list = []
    # 对应说话人在LibriSpeech中的路径
    spkpath = os.path.join(args.LibriSpeech, datatype, spk_id)
    for root, dirs, files in os.walk(spkpath):
        for file in files:
            if file.endswith(".flac") and file.split(".")[0] != audioid:
                files_list.append(file)
    aux_flac = random.sample(files_list, 1)[0]
    aux_path = os.path.join(spkpath, aux_flac.split("-")[1], aux_flac.split(".")[0] + "-norm.wav")
    aux_id = aux_flac.split(".")[0]
    return aux_id, aux_path


def choose_diffspk(datatype, audioid1, audioid2):
    # 根据audioid，从LibriSpeech中找到不同语音，返回绝对路径
    spk1_id = audioid1.split("-")[0]
    spk2_id = audioid2.split("-")[0]
    dataset_path = os.path.join(args.LibriSpeech, datatype)
    spklist = os.listdir(dataset_path)
    spklist.remove(spk1_id)
    spklist.remove(spk2_id)
    spklist.remove("normalize-resample.sh")
    spklist.remove(".complete") 
    spk_id = random.sample(spklist, 1)[0]
    spkpath = os.path.join(dataset_path, spk_id)

    aux_path = None
    aux_id = None
    for root, dirs, files in os.walk(spkpath):
        for file in files:
            if file.endswith(".flac"):
                aux_path = os.path.join(root, file.split(".")[0] + "-norm.wav")
                aux_id = file.split(".")[0]
                if os.path.exists(aux_path):
                    break
    return aux_id, aux_path


def create_scp(stat, output_path, df):
    # key dir
    # key：src1_src2_aux
    f_mix = open(os.path.join(output_path, "mix.scp"), "w", encoding="utf-8")
    f_ref = open(os.path.join(output_path, "ref.scp"), "w", encoding="utf-8")
    f_aux = open(os.path.join(output_path, "aux.scp"), "w", encoding="utf-8")

    # 记录目前LibiriMix所在的路径
    base_dir = args.data_dir
    # [max,min]
    mode = base_dir.split("/")[-2] + "/"
    # df = pd.read_csv(metadata_path)
    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        mixture_id = row["mixture_ID"]
        mixture_path = os.path.join(base_dir, row["mixture_path"].split(mode)[-1])
        source_1_path = os.path.join(base_dir, row["source_1_path"].split(mode)[-1])
        source_2_path = os.path.join(base_dir, row["source_2_path"].split(mode)[-1])        

        aux1_id, aux1_path = choose_samespk(stat, mixture_id.split("_")[0])
        aux2_id, aux2_path = choose_samespk(stat, mixture_id.split("_")[1])
        aux3_id, aux3_path = choose_diffspk(stat, mixture_id.split("_")[0], mixture_id.split("_")[1])
        key1 = mixture_id + "_" + aux1_id
        key2 = mixture_id + "_" + aux2_id
        key3 = mixture_id + "_" + aux3_id

        # save src1
        f_mix.write(key1 + " " + mixture_path+"\n")
        f_ref.write(key1 + " " + source_1_path + "\n")
        f_aux.write(key1 + " " + aux1_path + "\n")
        # save src2
        f_mix.write(key2 + " " + mixture_path + "\n")
        f_ref.write(key2 + " " + source_2_path + "\n")
        f_aux.write(key2 + " " + aux2_path + "\n")
        '''
        # save absent
        f_mix.write(key3 + " " + mixture_path + "\n")
        f_ref.write(key3 + " " + EMPTY_PATH + "\n")
        f_aux.write(key3 + " " + aux3_path + "\n")
        '''
        
    f_mix.close()
    f_ref.close()
    f_aux.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv_dir",
                        type=str,
                        default="/data1/LibriSpeech/Libri2Mix/wav16k/max/metadata/mixture_train-100_mix_clean.csv",
                        help="训练集csv路径")
    parser.add_argument("--dev_csv_dir",
                        type=str,
                        default="/data1/LibriSpeech/Libri2Mix/wav16k/max/metadata/mixture_dev_mix_clean.csv",
                        help="验证集csv路径")
    parser.add_argument("--test_csv_dir",
                        type=str,
                        default="/data1/LibriSpeech/Libri2Mix/wav16k/max/metadata/mixture_test_mix_clean.csv",
                        help="测试集csv路径")
    parser.add_argument("--LibriSpeech",
                        type=str,
                        default="/data1/LibriSpeech/LibriSpeech/",
                        help="LibriSpeech路径")
    parser.add_argument("--data_dir",
                        type=str,
                        default="/data1/LibriSpeech/Libri2Mix/wav16k/max/",
                        help="数据集路径(train/test的上一级)")
    parser.add_argument("--dataset",
                        type=str,
                        default="xmax_Libri2Mix",
                        help="数据集名称")
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv_dir)
    df_tr = df
    # df_tr = df[3000:]
    output_path_train = os.path.join("./data", args.dataset, "tr")
    os.makedirs(output_path_train, exist_ok=True)
    create_scp("train-clean-100", output_path_train, df_tr)

    df_cv = pd.read_csv(args.dev_csv_dir)
    # df_cv = df.iloc[:3000]
    output_path_dev = os.path.join("./data", args.dataset, "cv")
    os.makedirs(output_path_dev, exist_ok=True)
    create_scp("dev-clean", output_path_dev, df_cv)
       
    output_path_test = os.path.join("./data", args.dataset, "tt")
    os.makedirs(output_path_test, exist_ok=True)
    df_tt = pd.read_csv(args.test_csv_dir)
    create_scp("test-clean", output_path_test, df_tt)
    
    # 将所有spkid存为.spk文件
    with open("./data/libri2mix.spk", "w", encoding="utf-8") as f:
        for spk in SPK_LIST:
            f.write(spk + "\n")
    
