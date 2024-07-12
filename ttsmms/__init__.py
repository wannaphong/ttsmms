# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import glob
import re
import json
import tempfile
import subprocess
import math
import logging
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import ttsmms.commons
import ttsmms.utils
import argparse
from ttsmms.data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from ttsmms.models import SynthesizerTrn
from scipy.io.wavfile import write
from pathlib import Path
import logging


def download(lang, tgt_dir="./"):
    lang_fn, lang_dir = os.path.join(tgt_dir, lang+'.tar.gz'), os.path.join(tgt_dir, lang)
    isExist = os.path.exists(lang_dir)
    if isExist:
        return lang_dir
    Path(tgt_dir).mkdir(parents=True, exist_ok=True)
    if isExist:
        return lang_dir
    from urllib.request import urlretrieve
    url = f"https://dl.fbaipublicfiles.com/mms/tts/{lang}.tar.gz"
    logging.info(f"downloading {lang} from {url}")
    urlretrieve(url, lang_fn)
    Path(lang_dir).mkdir(parents=True, exist_ok=True)
    import tarfile
    file = tarfile.open(lang_fn)
    logging.info(f"extract all {lang} to {lang_dir}")
    file.extractall(tgt_dir)
    file.close()
    logging.info("Done")
    return lang_dir


class TextMapper(object):
    def __init__(self, vocab_file):
        self.symbols = [x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def text_to_sequence(self, text, cleaner_names):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        '''
        sequence = []
        clean_text = text.strip()
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence
    def uromanize(self, text, uroman_pl):
        iso = "xxx"
        with tempfile.NamedTemporaryFile() as tf, \
             tempfile.NamedTemporaryFile() as tf2:
            with open(tf.name, "w") as f:
                f.write("\n".join([text]))
            cmd = f"perl " + uroman_pl
            cmd += f" -l {iso} "
            cmd +=  f" < {tf.name} > {tf2.name}"
            os.system(cmd)
            outtexts = []
            with open(tf2.name) as f:
                for line in f:
                    line =  re.sub(r"\s+", " ", line).strip()
                    outtexts.append(line)
            outtext = outtexts[0]
        return outtext

    def get_text(self, text, hps):
        text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = ttsmms.commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def filter_oov(self, text):
        val_chars = self._symbol_to_id
        txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
        return txt_filt

class TTS:
    def __init__(self, model_dir_path: str, uroman_dir:str=None) -> None:
        self.model_path = model_dir_path
        self.vocab_file = f"{self.model_path}/vocab.txt"
        self.config_file = f"{self.model_path}/config.json"
        self.uroman_dir = uroman_dir
        assert os.path.isfile(self.config_file), f"{self.config_file} doesn't exist"
        self.hps = ttsmms.utils.get_hparams_from_file(self.config_file)
        self.text_mapper = TextMapper(self.vocab_file)
        self.net_g = SynthesizerTrn(
            len(self.text_mapper.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model
        )
        _ = self.net_g.eval()

        self.g_pth = f"{self.model_path}/G_100000.pth"
        _ = ttsmms.utils.load_checkpoint(self.g_pth, self.net_g, None)
        self.sampling_rate=self.hps.data.sampling_rate
        self.is_uroman = self.hps.data.training_files.split('.')[-1] == 'uroman'
    def _use_uroman(self, txt):
        if self.is_uroman != True:
            return txt
        if self.uroman_dir is None:
            tmp_dir = os.path.join(os.getcwd(),"uroman")
            if os.path.exists(tmp_dir) == False:
                cmd = f"git clone https://github.com/isi-nlp/uroman.git {tmp_dir}"
                logging.info(f"downloading uroman and save to {tmp_dir}")
                subprocess.check_output(cmd, shell=True)
            self.uroman_dir = tmp_dir
        uroman_pl = os.path.join(self.uroman_dir, "bin", "uroman.pl")
        logging.info("uromanize")
        txt =  self.text_mapper.uromanize(txt, uroman_pl)
        return txt

    def synthesis(self, txt, wav_path=None, convert_to_pcm16=True):
        txt = self._use_uroman(txt)
        txt = self.text_mapper.filter_oov(txt)
        stn_tst = self.text_mapper.get_text(txt, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)  # .cuda()
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])  # .cuda()
            hyp = self.net_g.infer(
                x_tst, x_tst_lengths, noise_scale=.667,
                noise_scale_w=0.8, length_scale=1.0
            )[0][0, 0].cpu().float().numpy()
        
        if convert_to_pcm16:
            # Convert to 16-bit PCM
            hyp = (hyp * 32767).astype(np.int16)
        
        if wav_path:
            write(wav_path, self.hps.data.sampling_rate, hyp)
            return wav_path
        
        return {"audio_bytes": hyp.tobytes(), "sampling_rate": self.sampling_rate}


def generate():
    parser = argparse.ArgumentParser(description='TTS inference')
    parser.add_argument('--model-dir', type=str, help='model checkpoint dir')
    parser.add_argument('--wav', type=str, help='output wav path')
    parser.add_argument('--txt', type=str, help='input text')
    args = parser.parse_args()
    ckpt_dir, wav_path, txt = args.model_dir, args.wav, args.txt

    vocab_file = f"{ckpt_dir}/vocab.txt"
    config_file = f"{ckpt_dir}/config.json"
    assert os.path.isfile(config_file), f"{config_file} doesn't exist"
    hps = ttsmms.utils.get_hparams_from_file(config_file)
    text_mapper = TextMapper(vocab_file)
    net_g = SynthesizerTrn(
        len(text_mapper.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    net_g#.cuda()
    _ = net_g.eval()

    g_pth = f"{ckpt_dir}/G_100000.pth"

    _ = ttsmms.utils.load_checkpoint(g_pth, net_g, None)

    # print(f"text: {txt}")
    txt = txt.lower()
    txt = text_mapper.filter_oov(txt)
    stn_tst = text_mapper.get_text(txt, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)#.cuda()
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])#.cuda()
        hyp = net_g.infer(
            x_tst, x_tst_lengths, noise_scale=.667,
            noise_scale_w=0.8, length_scale=1.0
        )[0][0,0].cpu().float().numpy()

    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    # print(f"wav: {wav_path}")
    write(wav_path, hps.data.sampling_rate, hyp)
    return


if __name__ == '__main__':
    generate()
