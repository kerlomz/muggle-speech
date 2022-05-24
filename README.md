# MUGGLE-SPEECH - 麻瓜中文语音识别



**注：MUGGLE-OCR当初是配套 Captcha-Trainer 项目的SDK调用，经过几轮的重构和迭代，新的框架已经准备好了，将在未来不久的某一天，重新归来。**



**MUGGLE-SPEECH** 是基于 Transformer 的端到端语音识别模型，采用ONNX部署方案。目前在语音识别领域只是试试水，毕竟样本受限，目前免费白嫖到的公开样本来自：AISHELL-1, AISHELL-3, MAGICDATA，尝试申请AISHELL-2，很难过被告知不对个人提供开源。十分欢迎愿意贡献数据集的朋友们，希望能够给社区尽一份绵薄之力。

![](https://test-1255362178.cos.ap-nanjing.myqcloud.com/20220525020842.png)



训练相关代码将整合到 MUGGLE-DL 框架中再开源，初次接触语音识别，还有很多不足的地方，将会慢慢改进。

以下是Python-SDK调用方法

```python
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

```



测试结果：![](https://test-1255362178.cos.ap-nanjing.myqcloud.com/20220525023610.png)

CPU 识别12个字大概 80-100毫秒 左右。



## 安装命令

`pip install muggle-speech`



## 交流群

857149419 （1群，已满）

934889548（2群）
