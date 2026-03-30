# Mosake Face Swap

使用 [images/pic-race.png](images/pic-race.png) 的臉替換 [videos/01-1-70.mp4](videos/01-1-70.mp4) 中偵測到的人臉，輸出為新的影片。

## 安裝

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果你要用系統 Python，也可以改成：

```powershell
C:/Users/User/AppData/Local/Programs/Python/Python39/python.exe -m pip install -r requirements.txt
```

## 執行

```powershell
.\.venv\Scripts\python.exe swap_video.py
```

預設輸出：`videos/01-1-70-swapped.mp4`

## 可用參數

- `--swap-all-faces`：每一幀替換所有偵測到的臉。
- `--max-frames 100`：只跑前 100 幀，方便快速測試。
- `--execution-provider CPUExecutionProvider`：指定 ONNX Runtime provider。

## 備註

第一次執行時，腳本會自動下載 `inswapper_128_fp16.onnx` 到 `models/`，並把 `buffalo_l` 分析模型下載到 `.insightface/`。