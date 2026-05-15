"""
run_recbole.py — Chạy bất kỳ model nào trong RecBole từ thư mục root của project.

Cách dùng:
  python run_recbole.py <model> <dataset> [key=value ...]       # train + eval
  python run_recbole.py eval <path/to/model.pth>               # eval only

  model   : tên model RecBole built-in   → truyền string  (ví dụ: SASRec)
          : model custom của bạn          → truyền đường dẫn import module.ClassName
                                            (ví dụ: recbole.my_model.MyModel)
  dataset : beauty | sports | ml-1m | steam
  key=val : override config tùy ý (ví dụ: epochs=50 learning_rate=0.0005)

Ví dụ — train + eval:
  python run_recbole.py SASRec beauty
  python run_recbole.py GRU4Rec beauty epochs=100
  python run_recbole.py SASRec sports learning_rate=0.0005 n_layers=4

Ví dụ — eval only từ checkpoint đã lưu:
  python run_recbole.py eval recbole_output/saved/SASRec-May-12-2026_09-28-11.pth

Ví dụ — model custom (file recbole/my_model.py, class MyModel):
  python run_recbole.py recbole.my_model.MyModel beauty
  python run_recbole.py recbole.my_model.MyModel beauty epochs=50

Output:
  recbole_output/saved/   — model checkpoints (.pth)
  recbole_output/log/     — training logs

Bản sửa lỗi: RecBole 1.2.x dùng các numpy aliases đã bị xóa trong numpy 2.0.
Script này patch tất cả trước khi import RecBole.
"""

import sys
import os
import importlib

# ── Numpy 2.0 compatibility patch ────────────────────────────────────────────
# RecBole 1.2.x dùng aliases cũ đã bị xóa trong numpy 2.0. Phải patch trước
# khi import bất kỳ thứ gì từ RecBole.
import numpy as np

_np_compat = {
    "bool8":    "bool_",
    "float_":   "float64",
    "complex_": "complex128",
    "int0":     "intp",
    "uint0":    "uintp",
    "str0":     "str_",
    "bytes0":   "bytes_",
    "object0":  "object_",
    "unicode_": "str_",
}
for _old, _new in _np_compat.items():
    if not hasattr(np, _old) and hasattr(np, _new):
        setattr(np, _old, getattr(np, _new))

# Aliases cho Python builtins bị xóa từ numpy 1.24
# Cần patch ở đây vì eval-only mode dùng pickle để load config,
# bỏ qua __init__ và compatibility_settings() của Configurator.
for _old, _builtin in [("float", float), ("int", int), ("bool", bool),
                        ("complex", complex), ("object", object), ("str", str)]:
    if not hasattr(np, _old):
        setattr(np, _old, _builtin)

# ── RecBole output directory patch ───────────────────────────────────────────
# RecBole lưu checkpoint vào 'saved/' và log vào './log/' theo mặc định.
# Patch để tập trung tất cả output vào recbole_output/.
#
# checkpoint_dir: set qua config_dict khi gọi run_recbole() / load_data_and_model()
# log dir: hardcode "./log/" bên trong hàm init_logger() → patch bytecode constant.

import types as _types
import recbole.utils.logger as _rl_logger

_log_code = _rl_logger.init_logger.__code__
_new_consts = tuple(
    "./recbole_output/log/" if c == "./log/" else c
    for c in _log_code.co_consts
)
_rl_logger.init_logger.__code__ = _log_code.replace(co_consts=_new_consts)

# ── Config ────────────────────────────────────────────────────────────────────

VALID_DATASETS  = {"beauty", "sports", "ml-1m", "steam"}
DATA_PATH       = "dataset/dataset/recbole"
RECBOLE_OUT_DIR = "recbole_output"
SAVED_DIR       = f"{RECBOLE_OUT_DIR}/saved"


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_override(args: list[str]) -> dict:
    """Parse 'key=value' args thành dict config override."""
    overrides = {}
    for arg in args:
        if "=" not in arg:
            print(f"Cảnh báo: bỏ qua tham số không hợp lệ '{arg}' (cần dạng key=value)")
            continue
        k, v = arg.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
        overrides[k] = v
    return overrides


def load_model_class(model_name: str):
    """
    Load model để truyền vào run_recbole().
    - String thuần ('SASRec')        → trả về string, RecBole tự tìm trong registry
    - Có dấu chấm ('pkg.mod.Class')  → import module và trả về class
    """
    if "." not in model_name:
        return model_name

    module_path, class_name = model_name.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        print(f"Lỗi: Không thể import module '{module_path}': {e}")
        sys.exit(1)

    if not hasattr(module, class_name):
        print(f"Lỗi: Không tìm thấy class '{class_name}' trong module '{module_path}'")
        sys.exit(1)

    return getattr(module, class_name)


def find_config(model: str, dataset: str) -> str | None:
    """Tìm file yaml config nếu tồn tại, ưu tiên theo thứ tự.

    Với custom model dạng 'module.ClassName', thử thêm bằng class name ngắn
    (ví dụ: 'recbole.cmamba4rec.CMamba4Rec' → tìm 'cmamba4rec_beauty.yaml').
    """
    model_lower = model.lower()
    class_name_lower = model_lower.rsplit(".", 1)[-1]   # phần sau dấu chấm cuối
    candidates = [
        f"{DATA_PATH}/{dataset}/{model_lower}_{dataset}.yaml",
        f"{DATA_PATH}/{dataset}/{class_name_lower}_{dataset}.yaml",
        f"{DATA_PATH}/{dataset}/sasrec_{dataset}.yaml",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


# ── Mode: eval only ───────────────────────────────────────────────────────────

def run_eval_only(model_file: str):
    """Load checkpoint đã lưu và evaluate trên test set (không train lại)."""
    if not os.path.isfile(model_file):
        # Thử tìm file khớp trong recbole_output/saved/
        candidates = [
            model_file,
            os.path.join(SAVED_DIR, model_file),
        ]
        found = next((p for p in candidates if os.path.isfile(p)), None)
        if not found:
            print(f"Lỗi: Không tìm thấy checkpoint tại '{model_file}'")
            print(f"\nCác checkpoint hiện có trong {SAVED_DIR}/:")
            if os.path.isdir(SAVED_DIR):
                for f in sorted(os.listdir(SAVED_DIR)):
                    if f.endswith(".pth"):
                        print(f"  {os.path.join(SAVED_DIR, f)}")
            sys.exit(1)
        model_file = found

    print("════════════════════════════════════════════════")
    print("  RecBole — Eval Only")
    print("════════════════════════════════════════════════")
    print(f"  Checkpoint : {model_file}")
    print("════════════════════════════════════════════════")
    print("")

    from recbole.quick_start import load_data_and_model
    from recbole.utils import get_trainer
    import torch
    import functools

    # PyTorch 2.6 đổi default weights_only=True, nhưng RecBole checkpoint chứa
    # cả config và dataset metadata (không chỉ weights) nên cần weights_only=False.
    _orig_load = torch.load
    torch.load = functools.partial(_orig_load, weights_only=False)
    try:
        config, model, dataset, train_data, valid_data, test_data = \
            load_data_and_model(model_file=model_file)
    finally:
        torch.load = _orig_load

    # Override data_path phòng trường hợp chạy từ thư mục khác
    config["data_path"] = DATA_PATH

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    print("\n── Valid set ────────────────────────────────────")
    valid_result = trainer.evaluate(valid_data, load_best_model=False, show_progress=True)
    print(valid_result)

    print("\n── Test set ─────────────────────────────────────")
    test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=True)
    print(test_result)

    print("\n✓ Eval hoàn tất.")


# ── Mode: train + eval ────────────────────────────────────────────────────────

def run_train(model: str, dataset: str, extra: list[str]):
    """Train model từ đầu và evaluate."""
    if dataset not in VALID_DATASETS:
        print(f"Lỗi: Dataset '{dataset}' không hợp lệ. Chọn trong: {VALID_DATASETS}")
        sys.exit(1)

    config_dict = {"data_path": DATA_PATH, "checkpoint_dir": SAVED_DIR}
    config_dict.update(parse_override(extra))

    config_file  = find_config(model, dataset)
    config_files = [config_file] if config_file else []

    print("════════════════════════════════════════════════")
    print("  RecBole Runner")
    print("════════════════════════════════════════════════")
    print(f"  Model   : {model}")
    print(f"  Dataset : {dataset}")
    print(f"  Config  : {config_file or '(default — không tìm thấy yaml)'}")
    for k, v in config_dict.items():
        if k != "data_path":
            print(f"  Override: {k} = {v}")
    print("════════════════════════════════════════════════")
    print("")

    model_obj = load_model_class(model)
    if not isinstance(model_obj, str):
        print(f"  → Loaded custom class: {model_obj}\n")

    from recbole.quick_start import run_recbole
    run_recbole(
        model=model_obj,
        config_file_list=config_files,
        config_dict=config_dict,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # Mode: eval only
    if len(sys.argv) >= 3 and sys.argv[1] == "eval":
        run_eval_only(sys.argv[2])
        return

    # Mode: train
    if len(sys.argv) < 3:
        print("Cách dùng:")
        print("  python run_recbole.py <model> <dataset> [key=value ...]  # train + eval")
        print("  python run_recbole.py eval <checkpoint.pth>              # eval only")
        print("")
        print("Ví dụ:")
        print("  python run_recbole.py SASRec beauty")
        print("  python run_recbole.py SASRec beauty epochs=50")
        print("  python run_recbole.py eval recbole_output/saved/SASRec-May-12-2026_09-28-11.pth")
        sys.exit(1)

    run_train(model=sys.argv[1], dataset=sys.argv[2], extra=sys.argv[3:])


if __name__ == "__main__":
    main()
