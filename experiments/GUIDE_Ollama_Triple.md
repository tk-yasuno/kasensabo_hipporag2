# GUIDE_Ollama_Triple: GPU対応による高速化

**作成日**: 2026-06-22  
**対象スクリプト**: `experiments/01_build_triple_index.py`

---

## 📋 概要

OpenIE で triple を抽出する際、Ollama が **GPU を使用していない（CPU-only モード）** 場合、処理が非常に遅くなります。

本ガイドでは、Ollama を GPU モードで正しく起動し、**処理時間を 60 分 → 20-30 分に短縮**する方法を説明します。

---

## 🔍 問題診断

### **症状**
- Triple 抽出に異常に時間がかかる（5-10 分/100チャンク）
- GPU メモリ使用量が 0.3 GB 程度（通常 4-6 GB）
- タスクマネージャーで GPU 使用率が 0%

### **原因**
Ollama が CPU-only モードで実行されており、GPU を利用していない。

### **確認方法**

**方法1: タスクマネージャーで確認**
1. Windows キー → 「タスクマネージャー」を開く
2. 「パフォーマンス」タブ → 「GPU」を確認
3. 専用 GPU メモリが **0.3 GB 以下** なら CPU-only 実行中

**方法2: コマンドで確認**
```powershell
# Ollama のログを確認
Get-Content "$env:LOCALAPPDATA\Ollama\logs\*.log" -Tail 50 | Select-String -Pattern "cuda|gpu|device"
```

**GPU が認識されていれば以下が表示される:**
```
INFO  backend/gpu.go:123 detected cuda device 0
INFO  backend/gpu.go:456 loaded nvidia driver
```

GPU が見つからない場合:
```
WARN  backend/gpu.go:789 no compatible gpu found
```

---

## ✅ 解決方法

### **Step 1: Ollama プロセスを確認・終了**

```powershell
# Ollama プロセス確認
Get-Process ollama

# プロセスを終了
Stop-Process -Name ollama -Force -ErrorAction SilentlyContinue

# 確実に終了したか確認
Get-Process ollama -ErrorAction SilentlyContinue
```

**期待される出力** (プロセスなし):
```
Get-Process : プロセス "ollama" が見つかりません。
```

---

### **Step 2: GPU 環境変数を設定**

#### **Windows 環境（推奨）**

```powershell
# PowerShell で環境変数設定 + Ollama 起動
$env:CUDA_VISIBLE_DEVICES="0"
$env:OLLAMA_GPU_COMPUTE_CAPABILITY="all"

# Ollama サーバー起動
ollama serve
```

#### **または、システム全体に設定**

1. **環境変数の設定**
   - Windows キー → 「環境変数」を検索
   - 「システム環境変数の編集」を開く
   - 「環境変数」ボタンをクリック
   - 「新規」をクリック
   
   以下を設定:
   ```
   変数名: CUDA_VISIBLE_DEVICES
   変数値: 0
   ```
   
   ```
   変数名: OLLAMA_GPU_COMPUTE_CAPABILITY
   変数値: all
   ```

2. **PC を再起動**
   ```powershell
   Restart-Computer
   ```

3. **Ollama を再起動**
   ```powershell
   ollama serve
   ```

---

### **Step 3: GPU 認識確認**

Ollama が起動したら、新しいターミナルで確認:

```powershell
# モデル情報確認（GPU 情報を表示）
ollama show swallow8b-lora-n4000-v09-q4 | Select-String -Pattern "parameters|tensor_split|gpu"
```

または、GPU メモリをリアルタイム監視:

```powershell
# リアルタイム GPU メモリ監視
while ($true) {
    Clear-Host
    Write-Host "=== GPU メモリ監視 ===" -ForegroundColor Cyan
    
    # GPU 情報
    Get-WmiObject -Class Win32_VideoController | Select-Object Name, @{Label='VRAM (GB)'; Expression={$_.AdapterRAM / 1GB}}
    
    # Ollama プロセス
    Write-Host "`n[Ollama プロセス]" -ForegroundColor Green
    Get-Process ollama -ErrorAction SilentlyContinue | Select-Object ProcessName, @{Label='Memory (MB)'; Expression={$_.WorkingSet / 1MB}}, @{Label='Threads'; Expression={$_.Threads}}
    
    Write-Host "`n次の更新まで 5 秒待機..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
```

**GPU モード実行時の期待される出力:**
```
Memory (MB)
-----------
   4,850    ← Swallow 8B が GPU で実行中（4-6 GB）
```

---

## 🚀 Triple 抽出の再実行

GPU 対応確認後、Triple 抽出スクリプトを再実行:

```powershell
cd i:\ACT2025.5.26-2030\MVP\kasensabo_hipporag2\experiments

python 01_build_triple_index.py
```

---

## 📊 期待される改善

### **CPU-only モード（改善前）**
| 項目 | 値 |
|------|-----|
| GPU メモリ | 0.3 GB |
| チャンク処理速度 | 5-10 分/100 チャンク |
| 全体実行時間 | 60-120 分 |

### **GPU モード（改善後）**
| 項目 | 値 |
|------|-----|
| GPU メモリ | 4-6 GB |
| チャンク処理速度 | 1-2 分/100 チャンク |
| 全体実行時間 | **20-30 分** |

**高速化倍率**: **2-3 倍**

---

## 🔧 トラブルシューティング

### **Q1: 環境変数を設定してもGPUが認識されない**

**原因**: NVIDIA ドライバが古い可能性

**解決方法**:
```powershell
# NVIDIA ドライバ確認
Get-WmiObject -Class Win32_PnPSignedDevice | Where-Object {$_.DeviceName -like "*NVIDIA*"} | Select-Object DeviceName, DriverVersion

# 最新ドライバをダウンロード
Start-Process "https://www.nvidia.com/Download/driverDetails.aspx"
```

### **Q2: Ollama が起動しない**

**確認事項**:
```powershell
# Ollama が正しくインストールされているか確認
Get-Command ollama

# インストール位置確認
(Get-Command ollama).Source

# Ollama サービス確認
Get-Service Ollama -ErrorAction SilentlyContinue | Select-Object Status, StartType
```

**解決方法**: Ollama を再インストール
```powershell
# 公式サイトからダウンロード
Start-Process "https://ollama.ai"
```

### **Q3: GPU メモリ不足エラー**

**症状**: `CUDA out of memory`

**解決方法**: VRAM 使用量を削減
```powershell
# 方法1: より軽量なモデルを使用
ollama pull swallow8b-lora-n4000-v09-q5_0  # Q4 → Q5 量子化

# 方法2: Ollama の GPU メモリ制限
$env:OLLAMA_GPU_MEMORY="4000"  # 4GB に制限
ollama serve
```

### **Q4: Triple 抽出途中で Ollama が落ちる**

**原因**: GPU メモリ不足またはドライバ不安定

**解決方法**:
```powershell
# NVIDIA ドライバ更新
nvidia-smi  # ドライババージョン確認

# Ollama ログ確認
Get-Content "$env:LOCALAPPDATA\Ollama\logs\*.log" -Tail 100 | Select-String -Pattern "ERROR|CUDA"

# 安全モードで再実行（遅いが安定）
$env:OLLAMA_NUM_GPU=0  # CPU-only フォールバック
python 01_build_triple_index.py
```

---

## 📈 パフォーマンスモニタリング

### **モニタリング用 PowerShell スクリプト**

```powershell
# GPU + Triple 抽出 の進捗監視
$scriptPath = "i:\ACT2025.5.26-2030\MVP\kasensabo_hipporag2\experiments\01_build_triple_index.py"

# バックグラウンド ジョブで実行
$job = Start-Job -ScriptBlock {
    cd i:\ACT2025.5.26-2030\MVP\kasensabo_hipporag2\experiments
    python 01_build_triple_index.py
}

# 進捗監視
while ($job.State -eq "Running") {
    Clear-Host
    Write-Host "=== Triple 抽出 進捗監視 ===" -ForegroundColor Cyan
    Write-Host "開始時刻: $(Get-Date)" -ForegroundColor Yellow
    
    # GPU 状態
    Write-Host "`n[GPU 状態]" -ForegroundColor Green
    Get-WmiObject -Class Win32_VideoController | Select-Object Name, @{Label='VRAM (GB)'; Expression={$_.AdapterRAM / 1GB}} | Format-Table
    
    # Ollama プロセス
    Write-Host "[Ollama メモリ使用量]" -ForegroundColor Green
    Get-Process ollama -ErrorAction SilentlyContinue | Select-Object ProcessName, @{Label='Memory (MB)'; Expression={$_.WorkingSet / 1MB}} | Format-Table
    
    # ジョブ出力（最後10行）
    Write-Host "[実行ログ（最新10行）]" -ForegroundColor Green
    Receive-Job -Job $job -ErrorAction SilentlyContinue | Select-Object -Last 10
    
    Start-Sleep -Seconds 10
}

# 完了
Write-Host "`n✅ Triple 抽出 完了!" -ForegroundColor Green
Receive-Job -Job $job
Remove-Job -Job $job
```

---

## 📁 生成ファイル確認

Triple 抽出完了後、以下ファイルが生成されます:

```powershell
# 生成ファイル確認
Get-ChildItem experiments/indices/triple* | Select-Object Name, @{Label='Size (MB)'; Expression={[math]::Round($_.Length/1MB, 2)}}
```

**期待される出力:**
```
Name                    Size (MB)
----                    ---------
triple_embs.npy              110
triples.json                  12
triple.index                 110
```

---

## ✨ Next Step

Triple Index 構築完了後、以下を実行:

```powershell
# Step 2: RAG 評価実行
cd experiments
python 04_eval_rag.py --model swallow --rag hipporag2
```

この時点で、triple filtering が自動で有効になります。

---

## 📚 参考資料

- **Ollama GPU 設定**: https://github.com/ollama/ollama#gpu-acceleration
- **NVIDIA CUDA**: https://developer.nvidia.com/cuda-downloads
- **Windows GPU コンピューティング**: https://docs.microsoft.com/ja-jp/windows/ai/

---

**作成者**: GitHub Copilot  
**最終更新**: 2026-06-22

