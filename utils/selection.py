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
import torchaudio
import torch
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from einops import rearrange
import pdb
from pyannote.database.util import load_rttm
from scipy.spatial.distance import cdist
from pyannote.core import Segment

# sd_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
# osd_pipeline = Pipeline.from_pretrained("pyannote/overlapped-speech-detection", use_auth_token="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")
sd_pipeline = Pipeline.from_pretrained("config_sd.yaml", use_auth_token="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")
osd_pipeline = Pipeline.from_pretrained("config_osd.yaml", use_auth_token="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")
emb_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
                device=torch.device("cuda:5"))

def slide(waveform, sample_rate=16000, duration=5.0, step=0.5, pad_constant=0):
    window_size: int = round(duration * sample_rate)
    step_size: int = round(step * sample_rate)
    _, num_samples = waveform.shape
    spk_timeline = []

    # prepare complete chunks
    if num_samples >= window_size:
        chunks = rearrange(
            waveform.unfold(1, window_size, step_size),
            "channel chunk frame -> chunk channel frame",
        )
        num_chunks, _, _ = chunks.shape
        for i in range(num_chunks):
            spk_timeline.append((i*step_size, window_size+i*step_size))
    else:
        num_chunks = 0
    

    # prepare last incomplete chunk
    has_last_chunk = (num_samples < window_size) or (
        num_samples - window_size
    ) % step_size > 0
    if has_last_chunk:
        last_chunk: torch.Tensor = waveform[:, num_chunks * step_size :]
        if pad_constant != 0:
            pad_constant = waveform[:, num_chunks * step_size :][-1][-1].int()
        last_output = np.pad(last_chunk, 
                ((0, 0), (0, window_size - last_chunk.shape[1])),
                "constant", 
                constant_values=(pad_constant, pad_constant))
        if num_chunks != 0:
            chunks = np.append(chunks, last_output[np.newaxis, :], axis=0)
        else:
            chunks = last_output[np.newaxis, :]
        spk_timeline.append((num_chunks*step_size, waveform.shape[1]))
    return spk_timeline, np.array(chunks)


def region(waveform, sample_rate=16000):
    _, num_samples = waveform.shape
    anchor_sizes = [anch * sample_rate for anch in range(1, 20, 1)]
    center = int(num_samples / 2)
    emb_chunks = []
    spk_timeline = []

    for anch in anchor_sizes:
        if anch < num_samples:
            # 正常切分
            emb_chunks.append(emb_model(waveform[:,center-int(anch/2):center+int(anch/2)].unsqueeze(1)))
            spk_timeline.append((center-int(anch/2), center+int(anch/2)))
        else:
            # 计算最后一块、退出
            emb_chunks.append(emb_model(waveform.unsqueeze(1)))
            spk_timeline.append((0, waveform.shape[1]))
            break
    return spk_timeline, np.array(emb_chunks).squeeze(1)


def mask_overlap_wav(overlap_tml, wav):
    sample_rate = 16000
    mask = torch.ones_like(wav)
    for segment in overlap_tml:
        start = int(segment.start * sample_rate)
        end = int(segment.end * sample_rate)
        mask[:, start:end] = 0
    return wav * mask

def get_embeddings(waveform, mode="slide"):
    # 输入: tensor [1, samples]
    # 输出：numpy [batch_size/1, feature_dim]
    if mode=="slide":
        spk_timeline, chunks = slide(waveform)
        return spk_timeline, emb_model(torch.tensor(chunks))
    elif mode=="region":
        return region(waveform)
    else:
        spk_timeline = [(0, waveform.shape[1])]
        return spk_timeline, emb_model(waveform.unsqueeze(0))

def choose_aux(waveform, dia, centroid, mode):
    sample_rate = 16000
    coller = int(0.5 * sample_rate)
    cand_embs = None
    cand_spks = []
    for speech_turn, _, speaker in dia.itertracks(yield_label=True):
        start = int(speech_turn.start * sample_rate)
        end = int(speech_turn.end * sample_rate)
        if (end - start) >= 6 * sample_rate:
            # model_emb 输入: tensor (num_chunks/batch_size, 1/num_channels, num_samples)
            # 输出: numpy [batch_size, feature_dim]
            spk_timeline, embedding = get_embeddings(torch.tensor(waveform[:, start+coller:end-coller]), mode=mode)
            if cand_embs is not None:
                cand_embs = np.concatenate((cand_embs, embedding), axis=0)
            else:
                cand_embs = embedding
            for spk in spk_timeline:
                cand_spks.append((speaker, start+coller+spk[0], start+coller+spk[1]))
    embeddings = torch.tensor(cand_embs).unsqueeze(1)

    # compute distance between embeddings and clusters
    # 输入：tensor [batch_size, 1, dim]
    # 输出：tensor [batch_size, num_speakers]
    e2k_distance = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"),
                centroid,
                metric="cosine",
            ),
            "(c s) k -> c s k",
            c=embeddings.shape[0],
            s=1,
        )
    soft_clusters = 2 - e2k_distance

    choosed_index = np.argmax(soft_clusters, axis=0)[0]
    choosed_chunk = []
    for index in choosed_index:
        dia[Segment(cand_spks[index][1]/sample_rate, cand_spks[index][2]/sample_rate)] = cand_spks[index][0]+"_top1"
        choosed_chunk.append(cand_spks[index])
    return choosed_chunk


def all_clust(df_md, rttm_dir, rttm_gt_dir, mode):
    hit_rate = 0
    for _, row in df_md.iterrows():
        # 获取音频路径
        key = row["mixture_ID"]
        mix_path = row["mixture_path"]
        waveform, sample_rate = torchaudio.load(mix_path)
        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
        # 计算说话人嵌入并聚类，这里修改了audio/pipeline/speaker_diarization
        dia, centroid = sd_pipeline(audio_in_memory)
        dia = dia.extrude(removed=dia.get_overlap())
        # 选择离聚类中心最近的
        choosed_chunk = choose_aux(waveform, dia, centroid, mode)
        with open(os.path.join(rttm_dir, f"{key}.rttm"), "w") as rttm:
            dia.write_rttm(rttm)
        hit_rate += check_choosed_chunk(choosed_chunk, rttm_gt_dir, key)
        # break
    print(rttm_dir, "Hit Rate:", hit_rate/df_md.shape[0])

def osd1(df_md, rttm_dir, rttm_gt_dir, rttm_osd_dir, mode):
    hit_rate = 0
    for _, row in df_md.iterrows():
        # 获取音频路径
        key = row["mixture_ID"]
        mix_path = row["mixture_path"]
        waveform, sample_rate = torchaudio.load(mix_path)
        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
        # 进行重叠检测，并将重叠部分mask掉
        osd = load_rttm(os.path.join(rttm_osd_dir, key+".rttm"))[key]
        # osd = osd_pipeline(audio_in_memory)
        masked_waveform = mask_overlap_wav(osd.get_timeline(), waveform)
        masked_audio_in_memory = {"waveform": masked_waveform, "sample_rate": sample_rate}
        # 计算说话人嵌入并聚类，这里修改了audio/pipeline/speaker_diarization
        dia, centroid = sd_pipeline(masked_audio_in_memory)
        dia = dia.extrude(removed=dia.get_overlap())
        # 选择离聚类中心最近的
        choosed_chunk = choose_aux(waveform, dia, centroid, mode)
        with open(os.path.join(rttm_dir, f"{key}.rttm"), "w") as rttm:
            dia.write_rttm(rttm)
        hit_rate += check_choosed_chunk(choosed_chunk, rttm_gt_dir, key)
        # break
    print(rttm_dir, "Hit Rate:", hit_rate/df_md.shape[0])


def osd2(df_md, rttm_dir, rttm_gt_dir, rttm_osd_dir, mode):
    hit_rate = 0
    for _, row in df_md.iterrows():
        # 获取音频路径
        key = row["mixture_ID"]
        mix_path = row["mixture_path"]
        waveform, sample_rate = torchaudio.load(mix_path)
        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
        # 使用初步重叠检测结果，将重叠部分mask掉
        osd_org = load_rttm(os.path.join(rttm_osd_dir, key+".rttm"))[key]
        masked_waveform = mask_overlap_wav(osd_org.get_timeline(), waveform)
        masked_audio_in_memory = {"waveform": masked_waveform, "sample_rate": sample_rate}
        # 切分为多个段，检测并过滤 这里修改了overlap detection
        osd = osd_pipeline(masked_audio_in_memory)
        masked_waveform = mask_overlap_wav(osd.get_timeline(), masked_waveform)
        masked_audio_in_memory = {"waveform": masked_waveform, "sample_rate": sample_rate}
        # 计算说话人嵌入并聚类，这里修改了audio/pipeline/speaker_diarization
        dia, centroid = sd_pipeline(masked_audio_in_memory)
        dia = dia.extrude(removed=dia.get_overlap())
        # 选择离聚类中心最近的
        choosed_chunk = choose_aux(waveform, dia, centroid, mode)
        with open(os.path.join(rttm_dir, f"{key}.rttm"), "w") as rttm:
            dia.write_rttm(rttm)
        hit_rate += check_choosed_chunk(choosed_chunk, rttm_gt_dir, key)
        # break
    print(rttm_dir, "Hit Rate:", hit_rate/df_md.shape[0])


def check_choosed_chunk(choosed_chunk, rttm_gt_dir, key):
    reference = load_rttm(os.path.join(rttm_gt_dir, f"{key}.rttm"))[key]
    ref_sig = reference.extrude(removed=reference.get_overlap())
    sample_rate = 16000
    hit = 0
    for speech_turn, track, speaker in ref_sig.itertracks(yield_label=True):
        for i in range(len(choosed_chunk)):
            if choosed_chunk[i][1] > speech_turn.start*sample_rate and choosed_chunk[i][2] < speech_turn.end*sample_rate:
                hit += 1
    # print(key, hit, len(choosed_chunk), hit/len(choosed_chunk))
    return hit/len(choosed_chunk)

if __name__ == '__main__':
    for n_src in [2]:
        for overlap in [30]:
            metadate_dir = f"/home/lsh/zyr/LibriCse/Cse{n_src}Mix/{overlap}/metadata/mixture_train-100_mix_clean.csv"
            df_md = pd.read_csv(metadate_dir)
            # slide5 region allclust longest 
            mode = "region"
            rttm_out_dir = f"./sd3/{mode}/cse{n_src}mix{overlap}ov"
            rttm_gt_dir = f"./sd3/gt/cse{n_src}mix{overlap}ov"
            rttm_osd_dir = f"./sd3/osd/cse{n_src}mix{overlap}ov"
            rttm_sd_dir = f"./sd3/sd/cse{n_src}mix{overlap}ov"
            os.makedirs(rttm_out_dir, exist_ok=True)
            if mode == "slide5":
                osd1(df_md, rttm_out_dir, rttm_gt_dir, rttm_osd_dir, "slide")
            elif mode == "region":
                osd1(df_md, rttm_out_dir, rttm_gt_dir, rttm_osd_dir, "region")
            elif mode == "allclust":
                all_clust(df_md, rttm_out_dir, rttm_gt_dir, "allclust")
            print(f"cse{n_src}mix{overlap}ov processed")
        # osd = load_rttm(f"./osd_cse2mix60ov_orgparam/{key}_osd.rttm")["waveform"]
        # torch.save(centroid, f"./osd_cse2mix60ov_orgparam/{key}.pth")
        # dia = load_rttm(f"./osd_cse2mix60ov_orgparam/{key}.rttm")["waveform"]
        # centroid = torch.load(f"./osd_cse2mix60ov_orgparam/{key}.pth")
