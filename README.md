

# TSE-with-ref-selection

target speaker extraction with speaker reference selection

## Dependencies

```shell
pip install -r requirements.txt
```

## Prepare Dataset

```
git clone https://github.com/JorisCos/LibriMix.git
```

## Reference Selection

| Method                                                       | ACC(ov=30%) |
| ------------------------------------------------------------ | ----------- |
| Speaker Diarization and use longest segments                 | 86.0        |
| Split into 5s chunks and clustering                          | 79.5        |
| Overlap detection and clustering with 5s sliding window      | 99.5        |
| Split into regions, overlap detection and clustering with regions | 89.2        |

## Speaker Extraction

```
# train
./train.sh

# evaluation
python cse_test.py
```

| Method                                                       | SI-SDR(model=x8515, ov=30%) | SI-SDR（model=xmax, ov=30%,use ground truth segmentation) |
| ------------------------------------------------------------ | --------------------------- | --------------------------------------------------------- |
| Speaker Diarization and use longest segments                 | 8.77                        | 13.62                                                     |
| Split into 5s chunks and clustering                          | 8.75                        | 14.00                                                     |
| Overlap detection and clustering with 5s sliding window      | 8.89                        | 14.32                                                     |
| Split into regions, overlap detection and clustering with regions | 9.11                        | 14.44                                                     |