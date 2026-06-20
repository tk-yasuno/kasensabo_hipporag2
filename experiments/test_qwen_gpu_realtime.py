"""
GPU使用状況をリアルタイムモニタリングしながらqwen2.5-14b-gpuを実行
"""

import httpx
import time
import subprocess
import threading
import sys

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5-14b-gpu"

# GPU監視用のフラグ
monitoring = True
gpu_usage_detected = False

def monitor_gpu():
    """バックグラウンドでGPU使用状況を監視"""
    global monitoring, gpu_usage_detected
    print("\n[GPU監視スレッド起動]")
    
    while monitoring:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", 
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                output = result.stdout.strip()
                parts = output.split(", ")
                if len(parts) >= 4:
                    mem_used = int(parts[1])
                    gpu_util = int(parts[3])
                    
                    if mem_used > 100 or gpu_util > 0:  # GPU使用を検出
                        if not gpu_usage_detected:
                            print(f"  ✓ GPU使用検出: メモリ={mem_used}MiB, 使用率={gpu_util}%")
                            gpu_usage_detected = True
                        
        except Exception:
            pass
        
        time.sleep(0.5)  # 0.5秒ごとに監視

def test_with_monitoring():
    """GPU監視しながら推論テスト"""
    print("=" * 60)
    print(f"GPU使用状況リアルタイム確認: {MODEL_NAME}")
    print("=" * 60)
    
    # GPU監視スレッドを開始
    monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
    monitor_thread.start()
    
    # 短いプロンプトでテスト
    test_prompt = "河川砂防とは何ですか？30文字以内で答えてください。"
    
    print(f"\n[テストプロンプト]")
    print(f"  {test_prompt}")
    print(f"\n[推論実行] {MODEL_NAME}")
    
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
                        "num_gpu": 99,
                        "temperature": 0.7,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            
            elapsed = time.time() - start_time
            
            print(f"\n[結果]")
            print(f"  実行時間: {elapsed:.2f}秒")
            print(f"  生成トークン数: {result.get('eval_count', 'N/A')}")
            
            if 'eval_duration' in result:
                eval_duration_sec = result['eval_duration'] / 1e9
                tokens_per_sec = result.get('eval_count', 0) / eval_duration_sec if eval_duration_sec > 0 else 0
                print(f"  推論速度: {tokens_per_sec:.1f} tokens/sec")
                
                # GPU判定
                if tokens_per_sec > 10:
                    print(f"  判定: ✓ GPU推論の可能性が高い（速い）")
                else:
                    print(f"  判定: ✗ CPU推論の可能性が高い（遅い）")
            
            print(f"\n[生成テキスト]")
            print(f"  {result.get('response', 'N/A')}")
            
    except Exception as e:
        print(f"\nエラー: {e}")
        sys.exit(1)
    finally:
        # 監視を停止
        global monitoring
        monitoring = False
        time.sleep(1)
    
    print("\n" + "=" * 60)
    if gpu_usage_detected:
        print("✓ GPU使用が確認されました")
    else:
        print("✗ GPU使用が確認できませんでした（CPUモード動作の可能性）")
    print("=" * 60)

if __name__ == "__main__":
    test_with_monitoring()
