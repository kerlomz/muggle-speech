import time

from muggle_speech import MuggleSpeech

sdk = MuggleSpeech(mode='wave')
# 文件格式必须是 wav 格式

if __name__ == '__main__':
    for i in range(1000):
        st = time.time()
        # 从 bytes 打开
        # wav_bytes = open(r"test.wav", "rb").read()
        # inputs = sdk.from_bytes(wav_bytes)
        # 从 文件 打开
        inputs = sdk.from_file(r"test.wav")
        predict_text = sdk.predict_file(inputs)
        print(predict_text, time.time() - st)

