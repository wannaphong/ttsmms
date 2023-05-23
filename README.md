# ttsmms
Text-to-speech with The Massively Multilingual Speech (MMS) project

This project want to help you for use Text-to-speech model from MMS project in Python.

Support 1,107 Languages! (See support_list.txt)

- VITS: [GitHub](https://github.com/jaywalnut310/vits)
- MMS: Scaling Speech Technology to 1000+ languages: [GitHub](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)

[Google colab](https://colab.research.google.com/github/wannaphong/ttsmms/blob/main/notebook/test.ipynb)

**Don't forget the TTS model in MMS project use CC-BY-NC license.**

## Install

> pip install ttsmms


## Usage

First, you need to download the model by

**Linux/Mac**

> curl https://dl.fbaipublicfiles.com/mms/tts/lang_code.tar.gz --output lang_code.tar.gz

and extract a tar ball archive.

**Linux/Mac**

> mkdir -p data && tar -xzf lang_code.tar.gz -C data/

and use code in python :D

```python
from ttsmms import TTS

tts=TTS("model_dir_path") # your path dir that extract a tar ball archive
wav=tts.synthesis("txt")
# output:
# {
#    "x":array(wav array),
#    "sampling_rate": 16000
# }

tts.synthesis("txt",wav_path="example.wav")
# output: example.wav file
```