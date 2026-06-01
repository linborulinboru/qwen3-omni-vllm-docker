# Qwen3-Omni vLLM 建置錯誤排查紀錄

**日期：** 2026-04-09  
**最終狀態：** 成功啟動

---

## 錯誤一：`--rope-scaling` 參數不被識別

**症狀：** 容器啟動後立即進入重啟循環（Restarting (2)）

**錯誤訊息：**

```text
vllm: error: unrecognized arguments: --rope-scaling {"type":"linear","factor":2.0}
```

**原因：** 新版 vllm 已移除 `--rope-scaling` CLI 參數。

**修復：** 從 `docker-compose.yml` 移除以下兩行：

```yaml
- "--rope-scaling"
- '{"type":"linear","factor":2.0}'
```

---

## 錯誤二：量化方式衝突

**症狀：** 移除 `--rope-scaling` 後仍然崩潰重啟

**錯誤訊息：**

```text
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, Quantization method specified in the model config (compressed-tensors)
  does not match the quantization method specified in the `quantization` argument (awq).
```

**原因：** 模型 `cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit` 實際使用 `compressed-tensors` 格式儲存，但 `docker-compose.yml` 中指定了 `--quantization awq`，兩者衝突。

**修復：** 從 `docker-compose.yml` 移除以下兩行，讓 vllm 自動偵測：

```yaml
- "--quantization"
- "awq"
```

---

## 問題三：容器啟動時仍從 HuggingFace 下載模型

**症狀：** 模型已放入 `app/models/`，但啟動 log 仍出現：

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

**原因：** 未設定離線模式，vllm 每次啟動都嘗試連線 HF Hub 做版本檢查。

**修復：** 在 `docker-compose.yml` 的 `qwen3-omni-instruct` 服務新增環境變數：

```yaml
environment:
  - HF_HUB_OFFLINE=1
  - TRANSFORMERS_OFFLINE=1
```

---

## 問題四：HF cache 佔用 22GB 磁碟空間

**症狀：** `app/models/` 內發現 `models--cyankiwi--Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit/`（22GB）及 `xet/`（50MB），是容器之前自動下載的快取，與使用者手動放入的模型檔案重複。

**修復：**

1. 將 `docker-compose.yml` 模型路徑從 HF repo ID 改為本地路徑：

   ```yaml
   # 舊
   - "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"
   # 新
   - "/models"
   ```

2. 同步更新 `qwen3-omni-serve` 的環境變數：

   ```yaml
   - VLLM_MODEL=/models
   ```

3. 刪除 HF cache 目錄：

   ```bash
   rm -rf app/models/models--cyankiwi--Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
   rm -rf app/models/xet
   ```

---

## 錯誤五：KV cache 記憶體不足（最終重啟循環原因）

**症狀：** 模型權重載入完成後崩潰，仍在重啟循環

**錯誤訊息：**

```text
ValueError: To serve at least one request with the models's max seq len (65536),
6.0 GiB KV cache is needed, which is larger than the available KV cache memory (5.42 GiB).
Based on the available memory, the estimated maximum model length is 59152.
```

**原因：** GPU（32GB）在載入 19.21 GiB 模型權重後，剩餘可用 KV cache 記憶體僅 5.42 GiB，不足以支撐 `max_model_len=65536` 所需的 6.0 GiB。

**修復：** 將 `--max-model-len` 從 `65536` 降為 `32768`（音訊轉錄每段 60 秒，32768 tokens 已足夠）：

```yaml
- "--max-model-len"
- "32768"
```

---

## 最終成功啟動

**總啟動耗時：** 約 3 分 48 秒（06:04:24 → 06:08:12）

| 階段                                  | 耗時       |
| ------------------------------------- | ---------- |
| 模型權重載入（6 個 safetensors 分片） | 119.54 秒  |
| KV cache profile + warmup             | 76.29 秒   |

**資源使用：**

- 模型佔用：19.21 GiB GPU 記憶體
- KV cache：5.46 GiB（59,616 tokens）

**服務健康狀態：**

```json
{
  "status": "ok",
  "vllm_connected": true,
  "vllm_model": "/models",
  "vllm_url": "http://qwen3-omni-30b-a3b-instruct:8901/v1"
}
```

---

## docker-compose.yml 最終有效設定摘要

```yaml
command:
  - "/models"
  - "--dtype"
  - "float16"
  - "--port"
  - "8901"
  - "--host"
  - "0.0.0.0"
  - "--max-model-len"
  - "32768"
  - "--gpu-memory-utilization"
  - "0.8"
  - "--enforce-eager"
  - "-tp"
  - "1"
  - "--seed"
  - "42"
  - "--enable-auto-tool-choice"
  - "--tool-call-parser"
  - "hermes"
  - "--chat-template"
  - "/opt/vllm/chat-template.jinja2"
  - "--allowed-local-media-path"
  - "/data"
  - "--media-io-kwargs"
  - '{"video":{"num_frames":-1}}'
environment:
  - HF_HUB_OFFLINE=1
  - TRANSFORMERS_OFFLINE=1
```

---

# Qwen3-Omni vLLM 啟動錯誤排查紀錄（RTX 5090 / Blackwell）

**日期：** 2026-06-01
**最終狀態：** 成功啟動
**環境：** RTX 5090（SM 12.0, Blackwell）／vLLM 0.22.0 ／base image CUDA 12.8.1

> 背景：在 RTX 5090 上重新跑同一套 stack，模型權重載入成功（19.21 GiB / ~120 秒），但隨後 `EngineCore` 在 `_initialize_kv_caches` → `determine_available_memory()` → `profile_run()` 階段崩潰。日誌中持續出現 `Failed to get device capability: SM 12.x requires CUDA >= 12.9.`，是後續所有錯誤的共同根因 — base image 內的 CUDA toolkit（12.8.1）無法正確識別 Blackwell（SM 12.0），導致依賴此 capability 的 JIT 與 attention kernel 走到壞掉的分支。

---

## 錯誤六：vision encoder flash-attn 取得 CPU 上的 `cu_seqlens`

**症狀：** 模型載入成功，但 `EngineCore` 在 profile 階段崩潰，容器進入重啟循環。

**錯誤訊息：**

```text
File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 6152, in profile_run
    dummy_encoder_outputs = self.model.embed_multimodal(...)
File ".../qwen3_omni_moe_thinker.py", line 1015, in forward
    hidden_states = blk(...)
File ".../vllm/v1/attention/ops/vit_attn_wrappers.py", line 51, in flash_attn_maxseqlen_wrapper
    output = flash_attn_varlen_func(...)
RuntimeError: cu_seqlens_q must be on CUDA
```

**原因：** vLLM 的 vision encoder 自動挑了 `FLASH_ATTN` backend 給 `MMEncoderAttention`，但在 Blackwell + CUDA 12.8.1 組合下，flash-attn v2 對 multimodal encoder 的 `cu_seqlens` tensor 處理會回退到 CPU 上，被 `flash_attn_varlen_func` 拒絕。LLM 主體的 attention 不受影響，問題只在 ViT 路徑。

**修復：** 強制 vision encoder 走 PyTorch SDPA 路徑（不依賴 flash-attn kernel）。在 `docker-compose.yml` 的 `qwen3-omni-instruct` `command` 加上：

```yaml
- "--mm-encoder-attn-backend"
- "TORCH_SDPA"
```

**驗證：** 重啟後日誌出現 `Using AttentionBackendEnum.TORCH_SDPA for MMEncoderAttention.`，profile_run 通過該錯誤。LLM 主體仍使用 FLASH_ATTN。

---

## 錯誤七：FlashInfer sampler JIT 偽 `sm75` 失敗

**症狀：** 修完錯誤六後，相同的 profile 階段又崩潰於另一處。

**錯誤訊息：**

```text
File ".../flashinfer/jit/core.py", line 108, in check_cuda_arch
    raise RuntimeError("FlashInfer requires GPUs with sm75 or higher")
RuntimeError: FlashInfer requires GPUs with sm75 or higher
```

**原因：** 訊息有誤導性 — RTX 5090（sm120）遠超 sm75。實際是 FlashInfer 在 JIT-build top-k/top-p sampler 前讀取不到 GPU compute capability（CUDA 12.8.1 不識別 SM 12.0），`check_cuda_arch()` 因此拋例外。vLLM 預設用 FlashInfer 來做 sampling。

**修復：** 停用 FlashInfer sampler，並明確告訴 JIT 工具鏈這是 Blackwell。在 `docker-compose.yml` 的 `qwen3-omni-instruct` `environment` 加上：

```yaml
- VLLM_USE_FLASHINFER_SAMPLER=0
- TORCH_CUDA_ARCH_LIST=12.0
```

**驗證：** 重啟後 sampler 改走 vLLM native 路徑，profile_run 完整通過，KV cache 初始化成功。

---

## 最終成功啟動

**模型載入耗時：** 138.20 秒（6 個 safetensors 分片）

**資源使用：**

- 模型權重：19.21 GiB GPU 記憶體
- KV cache：5.46 GiB（59,664 tokens）
- 容器狀態：`Up (healthy)`

**API 煙霧測試：**

```bash
curl -s -X POST http://localhost:8901/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/models","messages":[{"role":"user","content":"say hi in one word"}],"max_tokens":16}'
```

回應正常：`{"choices":[{"message":{"role":"assistant","content":"Hello!"}, "finish_reason":"stop"}], ...}`

---

## docker-compose.yml 本次新增的設定

```yaml
command:
  # ...（其餘維持先前設定）
  - "--mm-encoder-attn-backend"
  - "TORCH_SDPA"
environment:
  - HF_HUB_OFFLINE=1
  - TRANSFORMERS_OFFLINE=1
  - VLLM_USE_FLASHINFER_SAMPLER=0
  - TORCH_CUDA_ARCH_LIST=12.0
```

---

## 共同根因與長期解法

三個旁路（TORCH_SDPA / 停用 FlashInfer sampler / 設定 arch list）都是在繞過同一件事：**base image `aimehub/pytorch-2.8.0-aime-cuda12.8.1` 內的 CUDA 12.8.1 不認識 Blackwell（SM 12.0）**。

長期解法：將 `Dockerfile` 的 base image 升級到 CUDA ≥ 12.9 的版本，屆時可以移除上述三個旁路設定，讓 vLLM 自動挑回 FLASH_ATTN 與 FlashInfer sampler。
