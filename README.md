# Mosake Face Swap

使用 [images/pic-race.png](images/pic-race.png) 的臉替換 [videos/01-1-70.mp4](videos/01-1-70.mp4) 中偵測到的人臉，輸出為新的影片。

## 安裝

```powershell
\.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果你要用系統 Python，也可以改成：

```powershell
C:/Users/User/AppData/Local/Programs/Python/Python39/python.exe -m pip install -r requirements.txt
```

## 執行

```powershell
\.\.venv\Scripts\python.exe swap-video.py
```

預設輸出：依 target 影片自動命名（例如 `videos/01-1-70.mp4` → `videos/01-1-70-swapped.mp4`）

## Batch（config 多筆）

`config-swap-video.config` 支援用空行（或一行 `---`）分隔多個區塊，每個區塊是一筆任務（job）。

如果多筆任務共用同一張參考圖，也可以在同一個區塊內把 `Source=` 寫成首行，後續每行直接追加一個影片路徑，腳本會自動展開成多筆 job。

範例：

```ini
MaxFrames=100

Reference=./images/pic-race.png
Source=./videos/1.mp4

Reference=./images/pic-race.png
Source=./videos/2.mp4
```

或是：

```ini
Reference=./images/pic-race.png
Source=./videos/1.mp4
./videos/2.mp4
./videos/3.mp4
```

執行：

```powershell
\.\.venv\Scripts\python.exe swap-video.py --config config-swap-video.config
```

若每筆 job 沒有指定 `Output`，會依該筆 `Source` 影片自動產生輸出檔名（例如 `videos/1.mp4` → `videos/1-swapped.mp4`）。

## 可用參數

- `--swap-all-faces`：每一幀替換所有偵測到的臉。
- `--max-frames 100`：只跑前 100 幀，方便快速測試。
- `--execution-provider CPUExecutionProvider`：指定 ONNX Runtime provider。

## 備註

第一次執行時，腳本會自動下載 `inswapper_128_fp16.onnx` 到 `models/`，並把 `buffalo_l` 分析模型下載到 `.insightface/`。

目前 `requirements.txt` 使用 `onnxruntime-gpu`。若機器有可用的 NVIDIA CUDA 環境，腳本會自動優先使用 `CUDAExecutionProvider`，否則退回 `CPUExecutionProvider`。