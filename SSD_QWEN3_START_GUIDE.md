# SSD (Speculative Speculative Decoding) with Qwen3-32B

本文档提供了在vLLM中使用SSD加速Qwen3-32B推理的完整启动指南。

## 前置条件

- 已安装vLLM（包含我们的SSD集成）
- 已下载Qwen3-32B和对应的小模型草稿模型
- 足够的GPU显存

---

## 启动命令示例

### 1. 基础SSD同步模式

使用标准投机解码流程，适合入门测试：

```bash
vllm serve Qwen/Qwen3-32B-Instruct \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 4 \
  --speculative-config '{
    "model": "Qwen/Qwen3-1.8B-Instruct",
    "method": "ssd",
    "num_speculative_tokens": 6,
    "ssd_async": false
  }' \
  --port 8000 \
  --host 0.0.0.0
```

### 2. SSD异步模式（推荐生产环境）

启用树状缓存和异步投机，性能最佳：

```bash
vllm serve Qwen/Qwen3-32B-Instruct \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 4 \
  --speculative-config '{
    "model": "Qwen/Qwen3-1.8B-Instruct",
    "method": "ssd",
    "num_speculative_tokens": 7,
    "ssd_async": true,
    "ssd_async_fan_out": 3,
    "ssd_max_cached_sequences": 10000,
    "ssd_jit_speculate": false
  }' \
  --port 8000 \
  --host 0.0.0.0
```

### 3. 完整优化配置（生产环境推荐）

包含所有优化参数的完整配置：

```bash
vllm serve Qwen/Qwen3-32B-Instruct \
  --model Qwen/Qwen3-32B-Instruct \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --dtype auto \
  --quantization fp8 \
  --enable-prefix-caching \
  --speculative-config '{
    "model": "Qwen/Qwen3-1.8B-Instruct",
    "method": "ssd",
    "num_speculative_tokens": 7,
    "ssd_async": true,
    "ssd_async_fan_out": 3,
    "ssd_max_cached_sequences": 10000,
    "ssd_jit_speculate": false,
    "ssd_sampler_x": null,
    "draft_tensor_parallel_size": 1
  }' \
  --port 8000 \
  --host 0.0.0.0 \
  --api-key your-api-key-here
```

---

## 参数说明

### SSD专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `method` | string | - | 必须设置为 `"ssd"` |
| `model` | string | - | 草稿模型路径 |
| `num_speculative_tokens` | int | - | 每次投机的token数（K） |
| `ssd_async` | bool | false | 是否启用异步SSD模式 |
| `ssd_async_fan_out` | int | 3 | 异步模式的分支因子（F） |
| `ssd_max_cached_sequences` | int | 10000 | 树状缓存最大序列数 |
| `ssd_jit_speculate` | bool | false | 缓存未命中时是否使用JIT投机 |
| `ssd_sampler_x` | float \| null | null | 验证器的sampler X参数 |
| `draft_tensor_parallel_size` | int | - | 草稿模型的张量并行度 |

### 调优建议

1. **`num_speculative_tokens` (K)**:
   - 推荐值：5-8
   - 更大的K可能提高接受率，但也增加验证开销

2. **`ssd_async_fan_out` (F)**:
   - 推荐值：3-5
   - 控制预先预测的验证结果数量
   - 更大的F增加缓存命中率，但也增加内存使用

3. **显存优化**:
   - 减小 `max-num-seqs` 如果显存不足
   - 调整 `gpu-memory-utilization` 控制显存使用

---

## 使用curl测试服务

启动服务后，可以使用curl测试：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key-here" \
  -d '{
    "model": "Qwen/Qwen3-32B-Instruct",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

---

## Python API示例

也可以使用Python API：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-32B-Instruct",
    speculative_config={
        "model": "Qwen/Qwen3-1.8B-Instruct",
        "method": "ssd",
        "num_speculative_tokens": 7,
        "ssd_async": True,
        "ssd_async_fan_out": 3,
    },
    tensor_parallel_size=4,
)

prompts = ["你好，请介绍一下你自己"]
sampling_params = SamplingParams(max_tokens=512)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

---

## 性能监控

查看服务日志，你会看到类似这样的SSD相关日志：

```
[SSD] Initialized SSDSpeculator (async=True)
[SSD] Draft model loaded successfully
[SSD] Initializing tree cache on GPU
```

---

## 故障排查

### 问题：显存不足

**解决方案**：
- 减小 `max-num-seqs`
- 减小 `num_speculative_tokens`
- 使用 `--quantization fp8` 量化
- 减小 `ssd_max_cached_sequences`

### 问题：启动时草稿模型加载失败

**解决方案**：
- 确保草稿模型已正确下载
- 检查草稿模型与目标模型的词表大小是否一致
- 尝试设置 `draft_tensor_parallel_size` 与目标模型相同

### 问题：性能没有提升

**解决方案**：
- 检查 `ssd_async` 是否设置为 `true`
- 尝试调整 `num_speculative_tokens` 和 `ssd_async_fan_out`
- 查看日志中的缓存命中率信息
