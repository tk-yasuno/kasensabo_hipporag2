"""
Ollamaモデル設定確認スクリプト
"""

import httpx
import json

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:14b"

print("=" * 60)
print(f"Ollamaモデル設定確認: {MODEL_NAME}")
print("=" * 60)

# モデル詳細情報を取得
try:
    with httpx.Client(timeout=10.0) as client:
        response = client.post(
            f"{OLLAMA_URL}/api/show",
            json={"name": MODEL_NAME}
        )
        response.raise_for_status()
        model_info = response.json()
        
        print("\n[Modelfile]")
        if 'modelfile' in model_info:
            print(model_info['modelfile'])
        else:
            print("  Modelfile情報なし")
        
        print("\n[Parameters]")
        if 'parameters' in model_info:
            print(model_info['parameters'])
        else:
            print("  パラメータ情報なし")
        
        print("\n[Details]")
        for key in ['format', 'family', 'parameter_size', 'quantization_level']:
            if key in model_info.get('details', {}):
                print(f"  {key}: {model_info['details'][key]}")
        
        # GPU関連パラメータのチェック
        print("\n[GPU設定チェック]")
        modelfile_text = model_info.get('modelfile', '')
        parameters_text = model_info.get('parameters', '')
        
        if 'num_gpu' in modelfile_text or 'num_gpu' in parameters_text:
            print("  ✓ num_gpu パラメータが設定されています")
        else:
            print("  ✗ num_gpu パラメータが見つかりません（CPUモードの可能性）")
        
        print("\n" + "=" * 60)
        
except Exception as e:
    print(f"\nエラー: {e}")
