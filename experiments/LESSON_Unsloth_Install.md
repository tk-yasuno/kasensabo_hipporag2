# Unsloth インストール教訓（Windows / Python 3.12 / RTX 4060 Ti）

作成日: 2026-06-20  
対象環境: Windows 11, Python 3.12, NVIDIA GeForce RTX 4060 Ti 16GB, CUDA 12.4

---

## ✅ 最終動作確認済みパッケージ構成

```
torch              2.6.0+cu124
unsloth            2026.6.8
unsloth_zoo        2026.5.4       ← 2026.6.6 は gemma3n.py で torch.compile クラッシュ
transformers       5.5.0
torchao            0.7.0          ← 必須（0.17.0 は Float8WeightOnlyConfig 問題）
trl                1.6.0
bitsandbytes       0.49.2
pyarrow            18.1.0         ← 必須（24.0.0 は DLL 衝突でクラッシュ）
triton-windows     3.7.0.post26
sympy              1.14.0
```

---

## 🔧 インストール手順（再現手順）

### 1. torch 2.6.0+cu124 のクリーンインストール

```powershell
# 既存の壊れた torch を完全削除
$sp = ".venv-hipp\Lib\site-packages"
Remove-Item -Recurse -Force "$sp\torch"      -EA SilentlyContinue
Remove-Item -Recurse -Force "$sp\torchgen"   -EA SilentlyContinue
Get-ChildItem $sp | Where-Object { $_.Name -match "^~" } | Remove-Item -Recurse -Force -EA SilentlyContinue

# --no-deps で再インストール（依存関係の上書き防止）
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --no-deps
```

### 2. torchao を 0.7.0 にダウングレード（必須）

```powershell
pip install torchao==0.7.0 --no-deps
```

### 3. unsloth_zoo は 2026.5.4 を使用（必須）

```powershell
pip install unsloth_zoo==2026.5.4 --no-deps
```

> **理由**: 2026.6.6 以降は `temporary_patches/gemma3n.py` でモジュールレベルに
> `@torch_compile` デコレータを使用しており、triton と組み合わせると
> Windows で Access Violation クラッシュ（Exit -1073741819）が発生する。

### 4. unsloth / trl を --no-deps でインストール

```powershell
pip install unsloth --no-deps
pip install trl==1.6.0 --no-deps
```

### 5. pyarrow を 18.1.0 にダウングレード（必須）

```powershell
pip install pyarrow==18.1.0 --no-deps
```

> **理由**: pyarrow 24.0.0 は torch ロード後に DLL 衝突が発生し
> `pyarrow/__init__.py` line 71 で Access Violation クラッシュする。
> 18.1.0 ではこの問題が回避される。

### 6. triton-windows インストール

```powershell
pip install triton-windows
```

> **注意**: `pip show triton` では表示されない（パッケージ名が `triton_windows`）。
> `import triton; print(triton.__version__)` で確認する。

### 7. sympy 修復（壊れていた場合）

```powershell
pip install --force-reinstall "sympy>=1.13.1"
```

---

## 🩹 必須パッチ: `torch/_inductor/runtime/hints.py`

triton 3.7.0 では `AttrsDescriptor` クラスが削除された。
`torch 2.6.0` の `hints.py` は triton 3.2.0 を想定しており、フォールバックが不完全。

**パッチ内容**: 両方の `ImportError` が失敗した場合に `namedtuple` でフォールバックする。

```python
# 修正前（hints.py の該当箇所）
    except ImportError:
        from triton.compiler.compiler import AttrsDescriptor
        def AttrsDescriptorWrapper(...):
            ...

else:

# 修正後
    except ImportError:
        try:
            from triton.compiler.compiler import AttrsDescriptor
            def AttrsDescriptorWrapper(...):
                ...
        except ImportError:
            # triton 3.7+ では AttrsDescriptor が削除 → namedtuple でフォールバック
            AttrsDescriptorWrapper = collections.namedtuple(
                "AttrsDescriptor",
                ["divisible_by_16", "equal_to_1"],
                defaults=[(), ()],
            )

else:
```

**対象ファイル**: `.venv-hipp/Lib/site-packages/torch/_inductor/runtime/hints.py`

> **重要**: torch を再インストールするとこのパッチが消えるため、再適用が必要。

---

## ⚙️ 実行時の必須環境変数

```powershell
$env:UNSLOTH_COMPILE_DISABLE = "1"
```

または Python コード先頭に以下を追加：

```python
import os
os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'
```

> **理由**: Windows では `torch.compile`（triton JIT）が不安定なため、
> unsloth 内部の `@torch_compile` デコレータを noop に切り替える必要がある。
> この環境変数は `unsloth_zoo/temporary_patches/common.py` で認識される。

---

## 🚀 推論時の設定（batch_size=8 で 10s/問 達成）

```python
import os
os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'

from unsloth import FastLanguageModel

MODEL_PATHS = {
    "swallow": "models/swallow8b_merged_n4000_r32_d05",
    "elyza":   "models/elyza8b_merged_n4000",
}

# ロード
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATHS["swallow"],
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=None,
)
FastLanguageModel.for_inference(model)
tokenizer.padding_side = "left"   # バッチ生成には必須

# バッチ推論（8件まとめて処理）
batch_size = 8
inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

# 推論後はメモリ解放
import gc, torch
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
```

---

## ❌ 失敗パターンと原因

| 症状 | 原因 | 解決策 |
|------|------|--------|
| `ModuleNotFoundError: No module named 'sympy.core.cache'` | sympy が壊れている | `pip install --force-reinstall sympy>=1.13.1` |
| `ImportError: cannot import name 'AttrsDescriptor'` | triton 3.7.0 で API 削除 | `hints.py` にパッチを当てる |
| `Windows fatal exception: access violation` (pyarrow) | pyarrow 24.0.0 + torch の DLL 衝突 | pyarrow を 18.1.0 にダウングレード |
| `Windows fatal exception: access violation` (gemma3n) | unsloth_zoo 2026.6.6 の `@torch_compile` | unsloth_zoo を 2026.5.4 にダウングレード |
| `ModuleNotFoundError: No module named 'triton'` | triton_windows が未インストール | `pip install triton-windows` |
| torch が CPU版に上書きされる | `unsloth_zoo` が依存関係でtorchを再インストール | 全て `--no-deps` でインストールする |
| GPU 0.3GB しか使われず推論が遅い（Ollama） | Ollama の GPU 活用不足 | Unsloth FastLanguageModel に切り替える |

---

## 📋 動作確認コマンド

```powershell
.\.venv-hipp\Scripts\Activate.ps1
$env:UNSLOTH_COMPILE_DISABLE = "1"

# 基本確認
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# unsloth インポート確認
python -c "import os; os.environ['UNSLOTH_COMPILE_DISABLE']='1'; from unsloth import FastLanguageModel; print('unsloth OK')"

# bitsandbytes 4bit CUDA 確認
python -c "
import bitsandbytes as bnb, torch
lin = bnb.nn.Linear4bit(64, 64).cuda()
x = torch.randn(1, 64).cuda()
out = lin(x)
print('bnb 4bit CUDA OK, device:', out.device)
"
```
