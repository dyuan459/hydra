import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
import torch
from torch import Tensor

#!/usr/bin/env python3
"""
Quick diagnosis for a PyTorch YOLOv3 checkpoint (or any .pth/.pt).
Prints file info, top-level keys, state_dict summary, optimizer info, and attempts to infer num_classes.
"""



def sizeof_fmt(num: float, suffix: str = "B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def sha256sum(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def is_state_dict(obj: Any) -> bool:
    if not isinstance(obj, Mapping):
        return False
    if not obj:
        return False
    # accept mappings where values are Tensors or (Tensor-like)
    ok = 0
    for v in obj.values():
        if isinstance(v, Tensor):
            ok += 1
        elif isinstance(v, Mapping) and "shape" in v and "dtype" in v:
            # some safeloaders may wrap metadata
            ok += 1
        else:
            # allow buffers like None? but typically tensors dominate
            continue
    return ok > 0


def try_torch_load(path: str) -> Tuple[Any, str]:
    # Attempt safest load first (weights_only available in torch >=2.0)
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)  # type: ignore
        return obj, "torch.load(weights_only=True)"
    except TypeError:
        pass
    except Exception as e:
        # If weights_only failed for other reasons, fall back
        pass
    # Fallback to standard torch.load
    obj = torch.load(path, map_location="cpu")
    return obj, "torch.load()"


def extract_state_dict(obj: Any) -> Optional[Mapping[str, Tensor]]:
    # Common patterns
    if is_state_dict(obj):
        return obj  # type: ignore
    if isinstance(obj, Mapping):
        # Direct keys
        for k in ("state_dict", "model_state_dict", "weights", "params"):
            if k in obj and is_state_dict(obj[k]):
                return obj[k]  # type: ignore
        # Some checkpoints store model as an OrderedDict state_dict
        if "model" in obj:
            if is_state_dict(obj["model"]):
                return obj["model"]  # type: ignore
            # If it's a nn.Module (when fully unpickled), try .state_dict()
            try:
                import torch.nn as nn  # lazy
                if isinstance(obj["model"], nn.Module):
                    return obj["model"].state_dict()
            except Exception:
                pass
        # Ultralytics older: 'ema' is a model/module with state dict
        if "ema" in obj:
            try:
                import torch.nn as nn  # lazy
                if isinstance(obj["ema"], nn.Module):
                    return obj["ema"].state_dict()
            except Exception:
                pass
            if is_state_dict(obj["ema"]):
                return obj["ema"]  # type: ignore
    return None


def tensor_summary(t: Tensor) -> str:
    shape = list(t.shape)
    return f"shape={tuple(shape)}, dtype={str(t.dtype).replace('torch.', '')}, numel={t.numel()}"


def summarize_state_dict(sd: Mapping[str, Tensor], max_list: int = 50) -> Dict[str, Any]:
    total_params = 0
    dtypes = Counter()
    shapes = Counter()
    prefixes = Counter()
    param_list = []

    for name, tensor in sd.items():
        if not isinstance(tensor, Tensor):
            continue
        total_params += tensor.numel()
        dtypes[str(tensor.dtype)] += 1
        shapes[str(tuple(tensor.shape))] += 1
        prefix = name.split(".", 1)[0]
        prefixes[prefix] += 1
        if len(param_list) < max_list:
            param_list.append({
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "numel": int(tensor.numel()),
            })

    # Heuristic to infer num_classes and anchors_per_head (YOLO-like)
    candidates = []
    one_d_bias_sizes = Counter()
    for name, tensor in sd.items():
        if isinstance(tensor, Tensor) and tensor.ndim == 1:
            one_d_bias_sizes[int(tensor.numel())] += 1

    inferred_nc = None
    inferred_anchors = None
    for k, count in one_d_bias_sizes.items():
        # YOLOv3 usually has 3 heads, so count >= 3 for detection head biases
        if count >= 3 and k % 3 == 0:
            nc = k // 3 - 5
            if 0 < nc <= 1000:
                candidates.append((nc, 3, k, count))
    if candidates:
        # pick the most frequent candidate
        freq = Counter((nc, ah) for nc, ah, _, _ in candidates)
        (inferred_nc, inferred_anchors), _ = freq.most_common(1)[0]

    return {
        "num_tensors": len(sd),
        "total_parameters": int(total_params),
        "dtype_counts": dict(dtypes),
        "unique_shape_counts": dict(shapes.most_common(10)),
        "top_level_prefix_counts": dict(prefixes.most_common(10)),
        "sample_parameters": param_list,
        "yolo_inferred": {
            "num_classes": inferred_nc,
            "anchors_per_head": inferred_anchors,
        },
    }


def summarize_optimizer(opt_obj: Any) -> Dict[str, Any]:
    try:
        if isinstance(opt_obj, Mapping) and "state" in opt_obj and "param_groups" in opt_obj:
            state = opt_obj.get("state", {})
            param_groups = opt_obj.get("param_groups", [])
            # Count tensors in state
            tensor_count = 0
            for st in state.values():
                if isinstance(st, Mapping):
                    for v in st.values():
                        if isinstance(v, Tensor):
                            tensor_count += 1
            return {
                "format": "state_dict",
                "num_param_groups": len(param_groups),
                "num_state_entries": len(state),
                "num_state_tensors": tensor_count,
            }
        # If fully unpickled optimizer object
        cls = type(opt_obj).__name__
        return {"format": "object", "class": cls}
    except Exception:
        return {"format": "unknown"}


def pick_metadata(obj: Mapping[str, Any]) -> Dict[str, Any]:
    keys_of_interest = [
        "epoch", "best", "best_fitness", "best_loss", "results", "training_results", "date",
        "epochs", "lr", "lr0", "lrf", "momentum", "weight_decay",
        "nc", "names", "class_names", "anchors", "hyp", "hparams", "hyperparameters",
        "img_size", "imgsz", "model_yaml", "cfg", "git", "wandb_id", "torch_version",
    ]
    out = {}
    for k in keys_of_interest:
        if k in obj:
            v = obj[k]
            # Avoid dumping huge structures
            if isinstance(v, (list, tuple)) and len(v) > 50:
                out[k] = f"{type(v).__name__}(len={len(v)})"
            elif isinstance(v, Mapping) and len(v) > 50:
                out[k] = f"{type(v).__name__}(keys={len(v)})"
            else:
                try:
                    json.dumps(v)  # test serializable
                    out[k] = v
                except Exception:
                    out[k] = str(v)
    return out


def main():
    p = argparse.ArgumentParser(description="Diagnose a PyTorch checkpoint (.pth/.pt)")
    p.add_argument("path", nargs="?", default="yolov3_ckpt_best.pth", help="Checkpoint path")
    p.add_argument("--max-params", type=int, default=50, help="Max sample parameters to list")
    p.add_argument("--json", type=str, default=None, help="Optional path to save JSON summary")
    args = p.parse_args()

    path = args.path
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    print("File")
    print(f"  path: {os.path.abspath(path)}")
    size = os.path.getsize(path)
    print(f"  size: {size} bytes ({sizeof_fmt(size)})")
    try:
        h = sha256sum(path)
        print(f"  sha256: {h}")
    except Exception as e:
        print(f"  sha256: error: {e}")

    load_start = time.time()
    try:
        obj, how = try_torch_load(path)
    except Exception as e:
        print(f"\nLoad error: {type(e).__name__}: {e}")
        print("Tip: If this is a pickled model object, try loading with the original repo's codebase.")
        sys.exit(1)
    load_ms = (time.time() - load_start) * 1000.0
    print("\nLoad")
    print(f"  method: {how}")
    print(f"  time_ms: {load_ms:.1f}")
    print(f"  top_type: {type(obj).__name__}")

    summary: Dict[str, Any] = {
        "file": {
            "path": os.path.abspath(path),
            "size_bytes": size,
            "sha256": h if 'h' in locals() else None,
        },
        "load": {
            "method": how,
            "time_ms": load_ms,
            "top_type": type(obj).__name__,
        },
    }

    if isinstance(obj, Mapping):
        keys = list(obj.keys())
        print("\nTop-level keys")
        for k in keys:
            v = obj[k]
            vt = type(v).__name__
            if isinstance(v, Tensor):
                info = tensor_summary(v)
                print(f"  {k}: Tensor({info})")
            elif isinstance(v, Mapping):
                print(f"  {k}: {vt}(keys={len(v)})")
            elif isinstance(v, (list, tuple)):
                print(f"  {k}: {vt}(len={len(v)})")
            else:
                vs = str(v)
                if len(vs) > 120:
                    vs = vs[:117] + "..."
                print(f"  {k}: {vt} = {vs}")
        summary["top_level_keys"] = {k: type(obj[k]).__name__ for k in keys}
        # Metadata of interest
        meta = pick_metadata(obj)
        if meta:
            print("\nMetadata")
            for k, v in meta.items():
                print(f"  {k}: {v}")
            summary["metadata"] = meta

        # Optimizer
        for ok in ("optimizer", "optimizer_state_dict"):
            if ok in obj:
                opt_sum = summarize_optimizer(obj[ok])
                print("\nOptimizer")
                for k, v in opt_sum.items():
                    print(f"  {k}: {v}")
                summary["optimizer"] = opt_sum
                break

    # State dict extraction and summary
    sd = extract_state_dict(obj)
    if sd is None and isinstance(obj, Mapping):
        # Try common nested keys
        for k, v in obj.items():
            if is_state_dict(v):
                sd = v  # type: ignore
                break

    if sd is not None:
        print("\nState dict")
        sd_sum = summarize_state_dict(sd, max_list=args.max_params)
        print(f"  num_tensors: {sd_sum['num_tensors']}")
        print(f"  total_parameters: {sd_sum['total_parameters']}")
        if sd_sum["dtype_counts"]:
            print(f"  dtypes: {sd_sum['dtype_counts']}")
        if sd_sum["unique_shape_counts"]:
            print(f"  top_shapes: {sd_sum['unique_shape_counts']}")
        if sd_sum["top_level_prefix_counts"]:
            print(f"  top_level_prefix_counts: {sd_sum['top_level_prefix_counts']}")
        yi = sd_sum.get("yolo_inferred") or {}
        if yi.get("num_classes") is not None:
            print(f"  inferred_num_classes: {yi.get('num_classes')}")
            if yi.get("anchors_per_head") is not None:
                print(f"  anchors_per_head: {yi.get('anchors_per_head')}")
        print("  sample_parameters:")
        for pinfo in sd_sum["sample_parameters"]:
            name = pinfo["name"]
            shape = tuple(pinfo["shape"])
            dtype = pinfo["dtype"]
            numel = pinfo["numel"]
            print(f"    - {name}: shape={shape}, dtype={dtype}, numel={numel}")
        summary["state_dict"] = sd_sum
    else:
        print("\nState dict")
        print("  Not found or could not be extracted safely.")
        summary["state_dict"] = {"available": False}

    # Save JSON if requested
    if args.json:
        try:
            with open(args.json, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary JSON written to {args.json}")
        except Exception as e:
            print(f"\nFailed to write JSON: {e}")


if __name__ == "__main__":
    main()