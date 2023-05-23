# ttsmms
Text-to-speech with The Massively Multilingual Speech (MMS) project

This project want to help you for use Text-to-speech model from MMS project in Python.

Support 1,143 Languages! (See support_list.txt)

## Install

> pip install https://github.com/wannaphong/ttsmms/archive/refs/heads/main.zip


## Usage

First, you need to download the model by
> curl https://dl.fbaipublicfiles.com/mms/tts/lang_code.tar.gz --output lang_code.tar.gz

and extract a tar ball archive.

```python
from ttsmms import TTS

tts=TTS("model_dir_path") # your path dir that extract a tar ball archive
wav=tts.synthesis("txt")
# output:
# {
    "x":array(wav array),
    "sampling_rate": 16000
# }

tts.synthesis("txt",wav_path="example.wav")
# output: example.wav file
```