#!/usr/bin/env pwsh
<#
experiments/04b_run_all.ps1
──────────────────────────────────────────────────────────
Phase 4b: 全6条件（3 RAG方式 × 2モデル）を順次実行

Usage:
  cd i:\ACT2025.5.26-2030\MVP\kasensabo_hipporag2
  pwsh experiments/04b_run_all.ps1

  # dry-run（各条件10問のみ）:
  pwsh experiments/04b_run_all.ps1 -DryRun

  # 1条件のみ再実行:
  python experiments/04_eval_rag.py --model swallow --rag hipporag2
#>

param(
    [switch]$DryRun,
    [string]$Models  = "swallow,elyza",
    [string]$RagList = "naive,light,hipporag2"
)

$modelList = $Models  -split ","
$ragTypes  = $RagList -split ","

$dryFlag = if ($DryRun) { "--dry-run" } else { "" }

$total    = $modelList.Count * $ragTypes.Count
$current  = 0
$failed   = @()
$start    = Get-Date

Write-Host "=" * 60
Write-Host "  RAG 比較実験 — 全条件実行"
Write-Host "  条件数: $total  DryRun: $DryRun"
Write-Host "=" * 60

foreach ($model in $modelList) {
    foreach ($rag in $ragTypes) {
        $current++
        $cond = "${model}_${rag}"
        Write-Host ""
        Write-Host "-" * 60
        Write-Host "  [$current/$total] $cond"
        Write-Host "-" * 60

        $args_list = @(
            "experiments/04_eval_rag.py",
            "--model", $model,
            "--rag",   $rag
        )
        if ($dryFlag) { $args_list += $dryFlag }

        $t0 = Get-Date
        python @args_list
        $exit_code = $LASTEXITCODE
        $elapsed   = [math]::Round(((Get-Date) - $t0).TotalSeconds)

        if ($exit_code -ne 0) {
            Write-Host "  [FAILED] $cond  (exit=$exit_code)"
            $failed += $cond
        } else {
            Write-Host "  [OK] $cond  (${elapsed}s)"
        }
    }
}

$total_elapsed = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
Write-Host ""
Write-Host "=" * 60
Write-Host "  完了: $total 条件  失敗: $($failed.Count)  総時間: ${total_elapsed} 分"
if ($failed) {
    Write-Host "  失敗条件: $($failed -join ', ')"
}
Write-Host "=" * 60
