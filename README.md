# HF AI Studio

一個用 PyQt6 開發的桌面應用程式，可以在本機下載並執行 Hugging Face 上的語言模型，用起來就像 ChatGPT。

支援串流輸出、多輪對話、System Prompt、生成參數調整、深色/淺色主題、Markdown 渲染和硬體資源監控。

---

## 目錄

- [功能特色](#功能特色)
- [系統需求](#系統需求)
- [安裝](#安裝)
- [啟動程式](#啟動程式)
- [介面說明](#介面說明)
- [快速上手](#快速上手)
- [生成參數說明](#生成參數說明)
- [推薦模型](#推薦模型)
- [使用受限模型（Llama / Gemma）](#使用受限模型llama--gemma)
- [快捷鍵](#快捷鍵)
- [檔案與設定存放位置](#檔案與設定存放位置)
- [常見問題](#常見問題)

---

## 功能特色

- **任意模型**：從清單挑選，或直接輸入任何 Hugging Face 模型 ID
- **串流輸出**：回答一個字一個字出現，隨時可以停止
- **多輪對話記憶**：自動套用模型的 Chat Template
- **System Prompt**：自訂 AI 的角色與行為
- **參數即時調整**：Temperature、Top-p、Top-k、Max tokens、Repetition penalty
- **裝置與精度選擇**：CUDA / MPS（Apple Silicon）/ CPU，fp32 / fp16 / bf16
- **Markdown 渲染**：程式碼高亮、表格、清單
- **資源監控**：即時顯示 RAM / VRAM 用量
- **深色/淺色主題**，對話字體大小可調
- **對話匯出**：存成 Markdown 檔
- **設定自動保存**：下次開啟時恢復上次的設定

---

## 系統需求

| 項目 | 需求 |
|---|---|
| 作業系統 | Windows 10/11、macOS 12+、Linux |
| Python | 3.9 以上（建議 3.10 或 3.11） |
| 記憶體 | 至少 8 GB RAM（跑 0.5B～1.5B 小模型） |
| GPU（選配） | NVIDIA 顯示卡（CUDA）或 Apple M 系列晶片（MPS） |
| 硬碟空間 | 每個模型約 1～8 GB |

> 沒有 GPU 也能用，只是速度比較慢，建議搭配 1.5B 以下的小模型。

---

## 安裝

### 1. 建立虛擬環境（建議）

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. 安裝套件

```bash
pip install PyQt6 transformers torch markdown pygments psutil
```

| 套件 | 必要性 | 用途 |
|---|---|---|
| PyQt6 | 必要 | 圖形介面 |
| transformers | 必要 | 載入與執行模型 |
| torch | 必要 | 深度學習運算 |
| markdown | 選裝 | Markdown 渲染 |
| pygments | 選裝 | 程式碼語法高亮 |
| psutil | 選裝 | RAM 用量監控 |

### 3. NVIDIA GPU 使用者（重要）

用 `pip install torch` 預設可能會裝到 **CPU 版**。要使用 GPU，請到 [PyTorch 官網](https://pytorch.org/get-started/locally/) 選擇對應的 CUDA 版本，例如：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

用下面這行確認 GPU 有沒有被偵測到：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

輸出 `True` 就代表成功。

---

## 啟動程式

```bash
python hf_ai_studio.py
```

---

## 介面說明

```
┌──────────────────┬──────────────────────────────────────────┐
│  HF AI Studio    │  對話   模型名稱          [清除] [匯出]   │
│                  │ ┌──────────────────────────────────────┐ │
│  ▸ 模型          │ │                                      │ │
│    模型 ID       │ │            對話顯示區                │ │
│    裝置 / 精度   │ │                                      │ │
│    HF Token      │ │                                      │ │
│    [載入] [卸載] │ └──────────────────────────────────────┘ │
│                  │ ┌──────────────────────────────┐ [發送] │
│  ▸ 生成參數      │ │  輸入訊息…                   │ [停止] │
│  ▸ System Prompt │ └──────────────────────────────┘        │
│  ▸ 外觀          │                                          │
├──────────────────┴──────────────────────────────────────────┤
│ ● 模型狀態                        RAM / VRAM 用量  ▬▬▬▬     │
└─────────────────────────────────────────────────────────────┘
```

| 區域 | 說明 |
|---|---|
| **左側：模型** | 選擇模型、運算裝置、精度，載入或卸載模型 |
| **左側：生成參數** | 用滑桿調整 AI 的輸出風格 |
| **左側：System Prompt** | 設定 AI 的角色與行為規則 |
| **左側：外觀** | 切換深色/淺色主題、調整字體大小 |
| **右側：對話區** | 顯示對話內容和每則回覆的速度統計 |
| **底部狀態列** | 模型狀態（灰色＝未載入、藍色＝處理中、綠色＝就緒）和記憶體用量 |

---

## 快速上手

1. **選擇模型**：在「模型 ID」下拉選單選一個模型，第一次使用建議選 `Qwen/Qwen2.5-0.5B-Instruct`。
2. **選擇裝置**：保持「自動偵測」就好，程式會依序優先使用 CUDA → MPS → CPU。
3. **按「載入模型」**：第一次會從 Hugging Face 下載權重，依網速可能要幾分鐘。下載完成後會快取在本機，之後載入就很快。
4. **開始對話**：在下方輸入框打字，按 **Enter** 送出。
5. **需要時按「停止」**：可以隨時中斷正在生成的回答。
6. **換模型**：直接選另一個模型再按「重新載入」，舊模型會自動卸載。

### 自訂模型

模型 ID 欄位可以直接輸入任何 Hugging Face 模型名稱，格式是 `作者/模型名稱`，例如：

```
Qwen/Qwen2.5-3B-Instruct
TinyLlama/TinyLlama-1.1B-Chat-v1.0
gpt2
```

> 只支援**文字生成類**（Causal LM）模型。圖片、語音或 Embedding 模型無法使用。

---

## 生成參數說明

| 參數 | 預設值 | 說明 | 調整建議 |
|---|---|---|---|
| **Temperature** | 0.7 | 隨機性，越高越有創意、越低越穩定 | 寫程式、查資料：0.1～0.3<br>一般聊天：0.7<br>創意寫作：0.9～1.2<br>設為 **0** 時每次輸出都相同 |
| **Top-p** | 0.9 | 只從累積機率前 p 的詞彙中挑選 | 通常維持 0.9 即可 |
| **Top-k** | 50 | 只從機率最高的 k 個詞彙中挑選，0 = 不限制 | 通常維持 50 即可 |
| **Max new tokens** | 512 | 單次回覆的最大長度 | 回答被截斷時調高 |
| **Repetition penalty** | 1.1 | 懲罰重複內容，1.0 = 不懲罰 | 模型一直鬼打牆時調到 1.2～1.3 |

- 調整後**下一則訊息就會生效**，不用重新載入模型。
- 按「恢復預設值」可以全部還原。

### System Prompt 範例

```
你是一位資深 Python 工程師，請用繁體中文回答，並附上可執行的範例程式碼。
```

```
你是一位英文老師。使用者輸入中文時，請翻譯成自然的英文，並解釋重要的單字。
```

---

## 推薦模型

| 模型 | 參數量 | 約需記憶體（fp16） | 特點 |
|---|---|---|---|
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 0.36B | ~1 GB | 超輕量，CPU 也很快，以英文為主 |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ~1 GB | **新手首選**，中文能力不錯 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~3 GB | 速度和品質的平衡點 |
| `meta-llama/Llama-3.2-1B-Instruct` | 1.2B | ~2.5 GB | 需授權（見下節） |
| `google/gemma-2-2b-it` | 2.6B | ~5 GB | 需授權（見下節） |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | ~8 GB | 推理能力強，建議使用 GPU |

> 用 CPU 執行時精度會是 fp32，記憶體需求大約是表中數字的 **2 倍**。

---

## 使用受限模型（Llama / Gemma）

部分模型需要先同意授權條款才能下載：

1. 註冊並登入 [Hugging Face](https://huggingface.co/)
2. 進入模型頁面（例如 `meta-llama/Llama-3.2-1B-Instruct`），點選同意授權，等待審核通過（通常幾分鐘到幾小時）
3. 到 [Settings → Access Tokens](https://huggingface.co/settings/tokens) 建立一個 **Read** 權限的 Token
4. 把 Token（`hf_` 開頭）貼到程式的「HF Token」欄位，再按「載入模型」

> 基於安全考量，程式**不會保存** Token，每次開啟都需要重新輸入。
> 如果不想每次都輸入，可以在終端機執行一次 `huggingface-cli login`，之後欄位留空也能下載。

---

## 快捷鍵

| 按鍵 | 功能 |
|---|---|
| `Enter` | 送出訊息 |
| `Shift + Enter` | 換行 |

---

## 檔案與設定存放位置

### 模型快取

下載過的模型存放在：

| 系統 | 路徑 |
|---|---|
| Windows | `C:\Users\<使用者>\.cache\huggingface\hub` |
| macOS / Linux | `~/.cache/huggingface/hub` |

- 想釋放硬碟空間，可以直接刪除對應的模型資料夾。
- 想改存到其他磁碟，可以在啟動前設定環境變數 `HF_HOME`：

```bash
# Windows (PowerShell)
$env:HF_HOME="D:\hf_cache"

# macOS / Linux
export HF_HOME=/data/hf_cache
```

### 程式設定

主題、生成參數、System Prompt、字體大小和上次使用的模型會在關閉程式時自動保存：

| 系統 | 位置 |
|---|---|
| Windows | 登錄檔 `HKEY_CURRENT_USER\Software\HFAIStudio\App` |
| macOS | `~/Library/Preferences/` 下的 HFAIStudio 相關 plist 檔 |
| Linux | `~/.config/HFAIStudio/App.conf` |

刪除上述設定就能恢復成初始狀態。

---

## 常見問題

**Q：裝置選單沒有出現 CUDA？**
A：通常是裝到了 CPU 版的 PyTorch。請參考 [安裝](#3-nvidia-gpu-使用者重要) 第 3 步重新安裝 CUDA 版本。

**Q：載入時出現 `401` / `403` / `GatedRepoError`？**
A：這是受限模型。請確認已在模型頁面同意授權、審核已經通過，並且填入了正確的 HF Token。

**Q：出現 `CUDA out of memory`？**
A：模型太大，顯示卡記憶體不夠。可以試試：
- 改用參數量較小的模型
- 精度改選 `float16`
- 按「卸載」釋放記憶體後再重新載入
- 調低 Max new tokens

**Q：生成速度很慢？**
A：用 CPU 跑模型本來就比較慢。建議改用 1.5B 以下的模型，或改用 GPU。速度可以看每則回覆下方的 `tok/s` 數值。

**Q：回答一直重複同樣的內容？**
A：把 Repetition penalty 調高到 1.2～1.3，或把 Temperature 稍微調高。

**Q：回答品質很差，甚至自問自答？**
A：你可能載入的是 Base 模型（例如 `gpt2`）。請改用名稱中有 `Instruct`、`Chat` 或 `-it` 的版本。

**Q：對話太長之後，AI 好像忘了前面的內容？**
A：對話超過模型的上下文長度時，程式會自動刪除最舊的訊息，並在統計列顯示「已省略 N 則較舊訊息」。可以按「清除對話」重新開始。

**Q：下載到一半想關掉程式？**
A：程式會詢問是否強制離開。已經下載完成的部分會保留，下次載入時會從中斷的地方接續。

**Q：Apple Silicon（M1/M2/M3）怎麼用？**
A：裝置選「MPS」或「自動偵測」即可，不需要額外設定。

---

## 授權

本專案程式碼可自由使用與修改。各模型的使用請遵守 Hugging Face 上該模型自己的授權條款。