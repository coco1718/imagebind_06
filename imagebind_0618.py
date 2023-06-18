

#!git clone https://github.com/facebookresearch/ImageBind.git

import matplotlib.font_manager as fm
import os
import matplotlib.pyplot as plt

##!apt-get -qq install fonts-nanum 
fe = fm.FontEntry(
    fname=r'./NanumBarunGothic.ttf', 
    name='NanumBarunGothic')                        
fm.fontManager.ttflist.insert(0, fe)             
plt.rcParams.update({'font.size': 18, 'font.family': 'NanumBarunGothic'})

#!pip install -r ImageBind/requirements.txt
#!pip install soundfile
#!pip install torchaudio==0.13.0
#!pip install pytorchvideo
#!pip install torch # 최신버전 torch가 필요하다


#cd /content/ImageBind

import os


import data
import torch

import sys
sys.path.append('c:\\605pa11_en\min0605\ImageBind')

from models import imagebind_model
from models.imagebind_model import ModalityType

text_list=["bird","A dog","A car"]
image_paths=[".assets/dog_image.jpg",".assets/car_image.jpg",".assets/bird_image.jpg"]
audio_paths=[".assets/car_audio.wav",".assets/bird_audio.wav",".assets/dog_audio.wav"]

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

import torch




torch.cuda.empty_cache()
model = imagebind_model.imagebind_huge(pretrained=True) 


model.eval()
torch.cuda.empty_cache()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
model.to(device) 

torch.cuda.empty_cache()

inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device, batch_size=32),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device, batch_size=32),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device, batch_size=32),
}

with torch.no_grad():
    embeddings = model(inputs)


print(
    "Vision x Text: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
)

print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
)

print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
) 


visionxtext = torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1)
print(torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1))
print(torch.argmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1))

print(torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1))
print(torch.argmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1))
print(torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1))
print(torch.argmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1))


import numpy as np
print(torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1))
visionxaudio = torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1)
visionxaudio = visionxaudio.tolist() 
print(visionxaudio)
vxa = [np.argmax(visionxaudio[0]),np.argmax(visionxaudio[1]),np.argmax(visionxaudio[2])]
print(vxa) 


import cv2 
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
from PIL import Image
import torchaudio
import pytorchvideo
import requests
import os

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import io
import math
import tarfile
import multiprocessing

import scipy
import librosa
import requests
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
from IPython.display import Audio, display

plt.figure(figsize=(8,6))


for i,j in enumerate(vxa): 

  plt.subplot(3, 1, i+1)
  plt.axis('off')  


  def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
       display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
       display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
       raise ValueError("Waveform with more than 2 channels are not supported.")
  waveform, sample_rate = torchaudio.load(audio_paths[j])
  play_audio(waveform, sample_rate)

  plt.imshow(Image.open(image_paths[i]))

  plt.show()

  
    



