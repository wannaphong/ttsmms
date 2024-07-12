from ttsmms import TTS, download

dir_path = download("eng","./data") # lang_code, dir for save model

tts=TTS(dir_path) # or "model_dir_path" your path dir that extract a tar ball archive
wav=tts.synthesis("hello")
print(wav)