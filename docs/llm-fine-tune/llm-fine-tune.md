
# LLM Fine-tuning


## 架构

![MiniMind](../static/imgs/minimind.png)

1. hello输入，经过tokenizer转化为某个id，然后embedding转化为向量，进入Transformer
2. 首先经过GQA，在内部RMSNorm，不用减去平均值的正则化，防止梯度问题
3. 然后在Linear层里和QKV矩阵计算，获取对应向量，其中Q和K进行RoPE位置编码，然后点积，掩码处理后softmax，获得一个“相关性矩阵”，然后输入V点积，获得具体的上下文侧重信息，最后送入一个Linear层，将多头的多个向量拼接，投影回原维度，最后加上最初的输入，实现残差
4. GQA层之后也加上最初的embedding，保证残差，之后送入FFN
5. 在FFN里进行归一化RMSNorm，然后进行SwiGLU门控，左侧的Linear负责升维，右侧进入SiLU，非线性激活，然后两个线性层处理后的内容进行计算（不清楚这里是什么符号 ），之后送入新的Linear层浓缩降维，最后Dropout一部分防止过拟合，依旧残差处理
6. 出FFN之后依旧应用残差
7. 出Transformer层之后RMSNorm进行归一化，然后Linear层，得到整个词汇表每个词的得分
8. 之后softmax变为概率，然后再反向tokenizer转化为具体的词world
