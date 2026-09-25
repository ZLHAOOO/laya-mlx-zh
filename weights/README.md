# 权重下载

完整权重（MLX 格式，614MB，mmBERT encoder + 中文微调决策头）托管在 Hugging Face：

> **https://huggingface.co/ZLHAOOO/laya-mlx-zh**

命令行下载：

```bash
pip install huggingface_hub
huggingface-cli download ZLHAOOO/laya-mlx-zh --local-dir weights/ckpt-zh-v4-mlx
```

或网页直接下载。放到本目录（`weights/ckpt-zh-v4-mlx/`）后，快速开始里的示例代码即可运行。

权重基于 convaiinnovations/laya-multilingual（Apache-2.0）微调，可自由使用/修改/分发。
