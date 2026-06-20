"""
Unsloth/transformers/trl 互換パッチ適用スクリプト
参考: F16_vs_Q4_K_M_Lesson.md の地雷マップ
"""
import pathlib, sys

venv = pathlib.Path(__file__).parent.parent / ".venv-hipp" / "Lib" / "site-packages"
print(f"venv site-packages: {venv}")

ok = err = skip = 0

def patch(label, fpath, old, new):
    global ok, err, skip
    f = venv / fpath
    if not f.exists():
        print(f"  SKIP [{label}]: {fpath} not found")
        skip += 1
        return
    try:
        t = f.read_text(encoding="utf-8")
    except Exception as e:
        t = f.read_bytes().decode("utf-8", errors="replace")
    if old not in t:
        print(f"  SKIP [{label}]: pattern not found (already patched?)")
        skip += 1
        return
    f.write_text(t.replace(old, new), encoding="utf-8")
    print(f"  OK   [{label}]")
    ok += 1

def patch_bytes(label, fpath, old, new):
    global ok, err, skip
    f = venv / fpath
    if not f.exists():
        print(f"  SKIP [{label}]: {fpath} not found")
        skip += 1
        return
    c = f.read_bytes()
    if old not in c:
        print(f"  SKIP [{label}]: pattern not found (already patched?)")
        skip += 1
        return
    f.write_bytes(c.replace(old, new))
    print(f"  OK   [{label}]")
    ok += 1

# ── パッチ1: huggingface-hub 上限解除 ──────────────────────
patch(
    "transformers hf-hub cap",
    "transformers/dependency_versions_table.py",
    "huggingface-hub>=0.34.0,<1.0",
    "huggingface-hub>=0.34.0",
)

# ── パッチ2: Float8WeightOnlyConfig try-except ──────────────
old2 = """if is_torchao_available():
    SUPPORTED_SAFE_SERIALIZATION_CONFIGS = [
        torchao.quantization.Float8WeightOnlyConfig,
        torchao.quantization.Float8DynamicActivationFloat8WeightConfig,
    ]"""
new2 = """if is_torchao_available():
    try:
        SUPPORTED_SAFE_SERIALIZATION_CONFIGS = [
            torchao.quantization.Float8WeightOnlyConfig,
            torchao.quantization.Float8DynamicActivationFloat8WeightConfig,
        ]
    except AttributeError:
        SUPPORTED_SAFE_SERIALIZATION_CONFIGS = []"""
patch("torchao Float8Config", "transformers/quantizers/quantizer_torchao.py", old2, new2)

# ── パッチ3: trl FSDPModule try-except ─────────────────────
patch_bytes(
    "trl FSDPModule",
    "trl/models/utils.py",
    b"from torch.distributed.fsdp import FSDPModule",
    b"try:\r\n    from torch.distributed.fsdp import FSDPModule\r\nexcept ImportError:\r\n    FSDPModule = None",
)

# ── パッチ4: trl cp932 文字コード ──────────────────────────
patch(
    "trl read_text encoding",
    "trl/chat_template_utils.py",
    ".read_text()",
    '.read_text(encoding="utf-8")',
)

# ── パッチ5: hub.py list_repo_templates 404 ────────────────
patch(
    "hub.py except Exception",
    "transformers/utils/hub.py",
    "except (HTTPError, OfflineModeIsEnabled, requests.exceptions.ConnectionError):",
    "except Exception:  # EntryNotFoundError (404) も捕捉",
)

print(f"\n完了: OK={ok}  SKIP={skip}  ERROR={err}")
