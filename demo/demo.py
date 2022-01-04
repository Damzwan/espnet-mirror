import soundfile as sf
from espnet2.bin.tts_inference import Text2Speech

# decide the input sentence by yourself
print("Please input your favorite sentence!")
x = input()

text2speech = Text2Speech(model_file="exp/tts_train_raw_phn_tacotron_g2p_en_no_space/valid.loss.best.pth",
                          train_config="exp/tts_train_raw_phn_tacotron_g2p_en_no_space/config.yaml")

# synthesis
wav = text2speech(x)["wav"]
sf.write("out.wav", wav.numpy(), text2speech.fs, "PCM_16")
