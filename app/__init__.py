"""Application backend package.

注意:此处必须在任何第三方库 import 之前设置 HuggingFace 离线模式。
huggingface_hub 在 import 时会立即快照 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE
常量,而 langchain_chroma 等依赖链会在进程早期加载 huggingface_hub。
若在此之后才设置环境变量(例如 ingestion.LocalEmbeddingClient 构造时),
快照已固化,离线开关不生效,模型加载仍会向 huggingface.co 发起联网检查,
在国内网络下会连接超时并指数退避重试,导致 /chat 长时间无响应。
"""
import os

# 嵌入模型已固定缓存于本地(models/embedding),强制离线,禁止联网检查。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
