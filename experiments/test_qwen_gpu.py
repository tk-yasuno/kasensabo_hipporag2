"""
qwen2.5:14b GPU動作確認スクリプト (Dry Run)

Ollamaでqwen2.5:14bモデルを呼び出し、GPU使用状況を確認します。
"""

import httpx
import time
import subprocess
import sys
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5-14b-gpu"

def check_gpu_usage():
    """nvidia-smiでGPU使用状況を取得"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception as e:
        return f"nvidia-smi実行エラー: {e}"

def test_qwen_inference():
    """qwen2.5:14bで簡単な推論テストを実行"""
    print("=" * 60)
    print(f"qwen2.5:14b GPU動作確認 (Dry Run)")
    print("=" * 60)
    
    # GPU状態（推論前）
    print("\n[推論前] GPU使用状況:")
    gpu_before = check_gpu_usage()
    if gpu_before:
        print(gpu_before)
    else:
        print("  GPU情報を取得できませんでした")
    
    # テストプロンプト
    test_prompt = "河川砂防技術について、簡潔に説明してください。"
    
    print(f"\n[テストプロンプト]")
    print(f"  {test_prompt}")
    
    # Ollama API呼び出し
    print(f"\n[推論実行中] モデル: {MODEL_NAME}")
    print("  推論を開始します...")
    
    start_time = time.time()
    
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 4096,
                        "num_gpu": 99,  # 全レイヤーをGPUに
                        "temperature": 0.7,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            
            elapsed = time.time() - start_time
            
            # 結果表示
            print(f"\n[推論結果]")
            print(f"  実行時間: {elapsed:.2f}秒")
            print(f"  生成トークン数: {result.get('eval_count', 'N/A')}")
            
            if 'eval_duration' in result:
                eval_duration_sec = result['eval_duration'] / 1e9
                tokens_per_sec = result.get('eval_count', 0) / eval_duration_sec if eval_duration_sec > 0 else 0
                print(f"  推論速度: {tokens_per_sec:.1f} tokens/sec")
            
            print(f"\n[生成テキスト]")
            print(f"  {result.get('response', 'N/A')[:200]}...")
            
    except Exception as e:
        print(f"\n[エラー] 推論に失敗しました: {e}")
        sys.exit(1)
    
    # GPU状態（推論後）
    print(f"\n[推論後] GPU使用状況:")
    gpu_after = check_gpu_usage()
    if gpu_after:
        print(gpu_after)
    else:
        print("  GPU情報を取得できませんでした")
    
    # モデル情報確認
    print(f"\n[モデル詳細情報]")
    try:
        with httpx.Client(timeout=10.0) as client:
            show_response = client.post(
                f"{OLLAMA_URL}/api/show",
                json={"name": MODEL_NAME}
            )
            show_response.raise_for_status()
            model_info = show_response.json()
            
            # Modelfileのパラメータ確認
            if 'parameters' in model_info:
                print("  パラメータ設定:")
                params = model_info['parameters'].split('\n')
                for param in params:
                    if param.strip():
                        print(f"    {param}")
            
            # システム情報
            if 'system' in model_info:
                print(f"  システムプロンプト: {'設定あり' if model_info['system'] else '設定なし'}")
                
    except Exception as e:
        print(f"  モデル情報取得エラー: {e}")
    
    print("\n" + "=" * 60)
    print("✓ Dry Run 完了")
    print("=" * 60)

if __name__ == "__main__":
    test_qwen_inference()
