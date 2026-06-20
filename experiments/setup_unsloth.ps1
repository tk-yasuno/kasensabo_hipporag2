# Unsloth セットアップスクリプト (教訓の手順に従う)
# 実行: .\.venv-hipp\Scripts\Activate.ps1; pwsh experiments/setup_unsloth.ps1

$pip = "i:\ACT2025.5.26-2030\MVP\kasensabo_hipporag2\.venv-hipp\Scripts\pip.exe"

Write-Host "=== Step 1: torch 2.6.0+cu124 ===" -ForegroundColor Cyan
& $pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124 --no-deps -q
& $pip install "torchvision==0.21.0+cu124" --index-url https://download.pytorch.org/whl/cu124 --no-deps -q
Write-Host "torch: $(& $pip show torch | Select-String Version)"

Write-Host "=== Step 2: torchao 0.7.0 ===" -ForegroundColor Cyan
& $pip install "torchao==0.7.0" --no-deps -q
Write-Host "torchao: $(& $pip show torchao | Select-String Version)"

Write-Host "=== Step 3: unsloth + zoo + trl (--no-deps) ===" -ForegroundColor Cyan
& $pip install unsloth --no-deps -q
& $pip install unsloth_zoo --no-deps -q
& $pip install trl --no-deps -q
& $pip install peft --no-deps -q
& $pip install bitsandbytes --no-deps -q
& $pip install accelerate --no-deps -q

Write-Host "=== Step 4: 動作確認 ===" -ForegroundColor Cyan
$py = "i:\ACT2025.5.26-2030\MVP\kasensabo_hipporag2\.venv-hipp\Scripts\python.exe"
& $py -c "import torch; print('torch:', torch.__version__, '  CUDA:', torch.cuda.is_available())"
& $py -c "from unsloth import FastLanguageModel; print('unsloth OK')"
