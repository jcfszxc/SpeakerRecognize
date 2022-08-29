League of Legends Speaker Recognition System


Quick start
---
1. Install CUDA
2. Install Anaconda3
3. Install dependencies
```python
conda create -n Deep_Speaker python=3.6
conda activate Deep_Speaker
conda install tensorflow-gpu=2.1.0 keras=2.3.1
conda install -c conda-forge pandas librosa
conda install -c conda-forge pyaudio
conda install -c bricew python_speech_features
```
4. Download the data and run training:
https://pan.baidu.com/s/1oI0Q8La1kNQr2V27BTswHA jcfs 

put data in './datasets/'

```python
python train.py
```

Usage
---
pretrained model: https://pan.baidu.com/s/14RABQEDWwUrJ9CKOL7nStA jcfs
```python
python SpeakerRecog.pyw
```
