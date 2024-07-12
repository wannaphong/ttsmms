# py3-ttsmms

(A fork of ttsmms by wannaphong)
Text-to-speech with The Massively Multilingual Speech (MMS) project from Meta. 

This project want to help you for use Text-to-speech model from MMS project in Python. We aim to keep functioanlity minimal. Other projects such as py3-ttswrapper will provide more features.

Support 1,107 Languages! (See support_list.txt)

- VITS: [GitHub](https://github.com/jaywalnut310/vits)
- MMS: Scaling Speech Technology to 1000+ languages: [GitHub](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)

[Google colab](https://colab.research.google.com/github/wannaphong/ttsmms/blob/main/notebook/test.ipynb)

**Don't forget the TTS model in MMS project use CC-BY-NC license.**

## Install

> pip install py3-ttsmms

** Warning; There are a LOT of dependencies. If you already have Torch or PyTorch installed, you may need to uninstall it and reinstall it with the correct version. **

**NB: We use the same method names as ttsmms, but the code is not compatible.**

## Usage

First, you need to download the model by yourself or use the code below. Note these are ISO 639-1 language codes. Youc an see the full list in support_list.txt.

```python
from ttsmms import download

dir_path = download("eng","./data") # lang_code, dir for save model
```

or download file by yourself eg:


1. Download the language model file. Replace "lang_code" with the language code you want to download. You can see the full list in support_list.txt kur for Kurdish, eng for English, etc.

> curl https://dl.fbaipublicfiles.com/mms/tts/lang_code.tar.gz --output lang_code.tar.gz

2. extract a tar ball archive.

3. create a directory for save model

> mkdir -p data && tar -xzf lang_code.tar.gz -C data/

### Synthesis

```python
from ttsmms import TTS

tts=TTS(dir_path) # or "model_dir_path" your path dir that extract a tar ball archive
wav=tts.synthesis("txt")
# output:
# {
#    'audio_bytes': b'\x00\x0',
#    "x":array(wav array),
#    "sampling_rate": 16000
# }

tts.synthesis("txt",wav_path="example.wav")
# output: example.wav file
```
