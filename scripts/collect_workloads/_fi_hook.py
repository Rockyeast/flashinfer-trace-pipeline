# FlashInfer Python-level hook for workload capture.
# Loaded at SGLang process startup via sitecustomize.py (PYTHONPATH injection).
# Installs hooks on FlashInfer wrapper classes / functions.
# Captures are written to FI_CAPTURE_DIR as .pt files (torch.save).
#
# Configuration (read from environment variables):
#   FI_CAPTURE_DIR  — directory for .pt capture files (default: /tmp/fi_captures)
#   FI_HOOK_SPECS   — JSON dict from collect_workloads.hook_specs.parse_def_specs()
#   FI_PLAN_MAP     — JSON dict: plan() kwarg name → definition input name mapping

# ── 标准库 import（尽量少，保持启动速度）────────────────────────────────────────
import builtins as _builtins
import json as _json
import os as _os
import sys as _sys
import threading as _threading
from pathlib import Path as _Path

# ── 从环境变量读取配置（由 sitecustomize.py 在启动前注入）────────────────────────
_CAPTURE_DIR = _Path(_os.environ.get("FI_CAPTURE_DIR", "/tmp/fi_captures"))
# _SPECS: 要 patch 哪些 FlashInfer API，格式见 collect_workloads.hook_specs.parse_def_specs()
_SPECS       = _json.loads(_os.environ.get("FI_HOOK_SPECS", "{}"))
# _PLAN_MAP: plan() 的实际 kwarg 名 → definition 里的 input 名，用于版本兼容
_PLAN_MAP    = _json.loads(_os.environ.get("FI_PLAN_MAP",  "{}"))
# Verbose hook logs are useful when debugging missing captures, but too noisy
# for normal collection runs.
_VERBOSE     = _os.environ.get("FI_HOOK_VERBOSE", "0").lower() in {"1", "true", "yes", "on"}
# int tensor 的 dtype 集合，用于识别 kv_indptr / kv_indices 等结构性 tensor
_INT_DTYPES  = {"torch.int32", "torch.int64"}

# 全局写文件锁，防止多线程并发写出相同序号的文件
_lock          = _threading.Lock()
# 全局递增序号，每次 save 加一，保证文件名唯一
_seq           = [0]
# 已安装 patch 的 fi_api 集合，防止重复 patch
_hooked: set   = set()
# 原始函数 id → 已 hook 版本的映射，用于替换 sglang 缓存的旧引用
_orig_to_hooked: dict = {}
# (fi_api, thread_id) or (fi_api, None) → latest plan/begin_forward tensors.
# Some FlashInfer wrappers can prepare on one Python wrapper object and run on
# another, sometimes on another worker thread. Keep a fallback so run() can
# still pair with recent structural tensors after shape validation.
_latest_plan_data: dict = {}
# Prevent duplicate captures when a deprecated forward() wrapper simply calls
# the already-hooked run() method underneath.
_exec_guard = _threading.local()
# First-observed debug lines for attention prefill wrapper selection. These are
# intentionally tiny and one-shot so Modal logs show which wrapper actually ran
# without flooding long inference runs.
_seen_plan_logs: set = set()
_seen_exec_logs: set = set()
_seen_plan_warnings: set = set()

_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


# ── tensor 辅助函数 ────────────────────────────────────────────────────────────

def _summarize(v, force_full=False):
    """把一个参数值压缩成可序列化的摘要。

    - int32/int64 tensor → 保存完整数据（kv_indptr 等结构性 tensor 需要完整值）
    - 被 dump policy 选中的 tensor → 保存完整数据
    - 其他 tensor → 只保存 shape + dtype
    - 标量 → 直接记录值
    - tensor 列表/tuple → 只记录各自的 shape
    - 其他类型 → 返回 None（不记录）
    """
    import torch as _torch
    if isinstance(v, _torch.Tensor):
        if force_full or str(v.dtype) in _INT_DTYPES:
            # int tensor 或 policy 选中的 tensor：保存完整数据
            return {"type": "full", "tensor": v.detach().cpu().contiguous()}
        else:
            # 未选中的 tensor：只记录形状，不保存数值（节省空间，bench 会自己随机生成）
            return {"type": "shape",
                    "shape": list(v.shape),
                    "dtype": str(v.dtype).replace("torch.", "")}
    elif isinstance(v, (int, float)):
        # 标量（如 sm_scale）：直接记录值
        return {"type": "scalar", "value": float(v)}
    elif isinstance(v, (list, tuple)):
        import torch as _torch2
        if v and all(isinstance(x, _torch2.Tensor) for x in v):
            # tensor 列表：记录每个 tensor 的形状
            return {"type": "tuple_shapes",
                    "shapes": [list(x.shape) for x in v],
                    "dtypes": [str(x.dtype).replace("torch.", "") for x in v]}
    return None  # 其他类型（如字符串、None）不记录


def _full_tensor_names(info):
    """Union of definition inputs that should keep full tensor values."""
    names = set()
    for definition in info.get("definitions", ()):
        names.update(definition.get("full_tensor_inputs", ()))
    return names


def _definition_input_names(info):
    """Best-effort fallback parameter names from the first definition."""
    definitions = info.get("definitions", ())
    if not definitions:
        return []
    inputs = definitions[0].get("inputs", {})
    return list(inputs.keys())


def _capture_call(args, kwargs, full_tensor_names=None, param_names=()):
    """把一次函数调用的所有参数压缩成 {参数名: 摘要} 字典。

    位置参数命名为 arg_0、arg_1、...（名字未知，只知道顺序）
    关键字参数直接用原始 key 命名（如 kwarg_q、kwarg_k）
    返回的 dict 就是 workload entry 的原始数据，后续写入 .pt 文件。
    """
    full_tensor_names = full_tensor_names or set()
    captured = {}
    for i, a in enumerate(args):
        param_name = param_names[i] if i < len(param_names) else None
        s = _summarize(a, force_full=param_name in full_tensor_names)
        if s:
            captured[f"arg_{i}"] = s  # 位置参数，名字只能用序号
    for k, v in kwargs.items():
        s = _summarize(v, force_full=k in full_tensor_names)
        if s:
            captured[f"kwarg_{k}"] = s  # 关键字参数，保留原始名
    return captured


def _tensor_shape_debug(v):
    """Return a compact tensor-ish shape summary for debug logs."""
    if hasattr(v, "shape") and hasattr(v, "dtype"):
        try:
            return {
                "shape": [int(x) for x in list(v.shape)],
                "dtype": str(v.dtype).replace("torch.", ""),
            }
        except Exception:
            return {"shape": "?", "dtype": str(getattr(v, "dtype", "?"))}
    return None


def _call_shape_debug(args, kwargs):
    """Summarize tensor shapes from one call without copying data."""
    out = {}
    for i, value in enumerate(args[:4]):
        summary = _tensor_shape_debug(value)
        if summary:
            out[f"arg_{i}"] = summary
    for key, value in list(kwargs.items())[:6]:
        summary = _tensor_shape_debug(value)
        if summary:
            out[f"kwarg_{key}"] = summary
    return out


def _wrapper_attr_debug(obj):
    """Capture a few wrapper attributes that influence paged/ragged selection."""
    attrs = {}
    for name in (
        "use_ragged",
        "page_size",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "kv_layout",
        "_kv_layout",
    ):
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if isinstance(value, (bool, int, float, str)):
            attrs[name] = value
    return attrs


def _first_dim(summary):
    """Return the first tensor dimension from a captured shape summary."""
    if not summary:
        return None
    if summary.get("type") == "shape":
        shape = summary.get("shape") or []
    elif summary.get("type") == "full":
        tensor = summary.get("tensor")
        shape = list(getattr(tensor, "shape", []))
    else:
        return None
    return int(shape[0]) if shape else None


def _latest_plan_matches_run(plan_data, captured):
    """Validate latest-plan fallback against the current run() tensor shapes."""
    checked = False

    q_total = _first_dim(captured.get("kwarg_q") or captured.get("arg_0"))
    if q_total is not None and "qo_indptr" in plan_data:
        checked = True
        try:
            if int(plan_data["qo_indptr"][-1].item()) != q_total:
                return False
        except Exception:
            return False

    k_total = _first_dim(captured.get("kwarg_k") or captured.get("arg_1"))
    # Ragged prefill has k as a real tensor and no kv_indices. Paged attention
    # uses cache pages, so kv_indptr[-1] is not comparable to arg_1.shape[0].
    if k_total is not None and "kv_indptr" in plan_data and "kv_indices" not in plan_data:
        checked = True
        try:
            if int(plan_data["kv_indptr"][-1].item()) != k_total:
                return False
        except Exception:
            return False

    return checked


def _extract_plan_tensors(args, kwargs, plan_param_names=()):
    """从 plan() 调用里提取 int32/int64 结构性 tensor，并归一化为 definition 里的 input 名。

    plan() 的参数会因 FlashInfer 版本不同而有不同的名字（如 paged_kv_indptr vs kv_indptr），
    _PLAN_MAP 负责把各种历史名字映射到 definition 里统一的名字（如 kv_indptr）。

    plan_param_names：由 _hook_wrapper 在安装 patch 时通过 inspect.signature(orig_plan)
    读取的真实参数名列表（去掉 self），用于将位置参数映射到正确的语义名。
    若为空（inspect 失败的兜底情况），位置参数会被跳过，只依赖 kwargs 路径。
    """
    import torch as _torch
    result = {}

    # 处理位置参数：只记录 int32/int64 tensor（其他类型不是结构性 tensor，跳过）
    for i, a in enumerate(args):
        if isinstance(a, _torch.Tensor) and str(a.dtype) in _INT_DTYPES:
            # 用从 inspect.signature() 读到的真实参数名，再经 _PLAN_MAP 映射到 definition 的 input 名
            if i >= len(plan_param_names):
                continue  # 超出已知参数范围，跳过（避免错误命名）
            raw_name = plan_param_names[i]
            def_name = _PLAN_MAP.get(raw_name, raw_name)
            if def_name not in result:  # 避免位置参数和 kwargs 重复
                result[def_name] = a.detach().cpu().contiguous()

    # 处理关键字参数：经 _PLAN_MAP 映射后存入
    for k, v in kwargs.items():
        if isinstance(v, _torch.Tensor) and str(v.dtype) in _INT_DTYPES:
            def_name = _PLAN_MAP.get(k, k)  # 没有映射则保持原名
            if def_name not in result:
                result[def_name] = v.detach().cpu().contiguous()

    # SGLang 的 KV cache pool 是预分配的，kv_indices 可能比实际用到的多
    # kv_indptr[-1] 记录了真正用到的 kv_indices 数量，裁剪掉多余的部分
    if "kv_indices" in result and "kv_indptr" in result:
        valid = int(result["kv_indptr"][-1].item())
        result["kv_indices"] = result["kv_indices"][:valid].clone()

    return result


# ── 写文件 ─────────────────────────────────────────────────────────────────────

def _save(record):
    """把一次捕获记录写入 .pt 文件（torch.save 格式）。

    文件名格式：pid{进程id}_seq{全局序号:06d}.pt
    多进程安全：pid 不同则文件名不同；同进程内用 _lock 保证序号递增不冲突。
    """
    import torch as _torch
    with _lock:
        seq = _seq[0]
        _seq[0] += 1
    pid  = _os.getpid()
    path = _CAPTURE_DIR / f"pid{pid}_seq{seq:06d}.pt"
    _torch.save(record, str(path))
    if _VERBOSE and seq < 3:
        print(f"[_fi_hook] SAVED: {path.name} fi_api={record.get('fi_api','?')}", flush=True)


def _capture_wrapper_execution(self, args, kwargs, fi_api, call_type, info, param_names=()):
    """Capture one wrapper execution method call (run/forward)."""
    # 捕获执行入口的参数（主要是 q/k/v 等 float tensor 的 shape）
    captured = _capture_call(
        args,
        kwargs,
        full_tensor_names=_full_tensor_names(info),
        param_names=param_names,
    )
    # 取出 plan()/begin_forward() 里暂存的 int tensor（如果有）
    plan_data = dict(getattr(self, "_fi_plan_data", {}))
    if not plan_data:
        for key in ((fi_api, _threading.get_ident()), (fi_api, None)):
            latest_plan = dict(_latest_plan_data.get(key, {}))
            if latest_plan and _latest_plan_matches_run(latest_plan, captured):
                plan_data = latest_plan
                break

    # 备用路径：如果 plan() 的 int tensor 没被捕获到，
    # 尝试直接从 wrapper 实例属性里读。
    _INDPTR_ATTRS = (
        "_paged_kv_indptr_buf", "paged_kv_indptr",
        "kv_indptr", "_kv_indptr_buf",
        "_indptr_buf",
    )
    _INDICES_ATTRS = (
        "_paged_kv_indices_buf", "paged_kv_indices",
        "kv_indices", "_kv_indices_buf",
        "_indices_buf",
    )
    if "kv_indptr" not in plan_data:
        for _attr in _INDPTR_ATTRS:
            _t = getattr(self, _attr, None)
            if _t is not None and hasattr(_t, "dtype") and str(_t.dtype) in _INT_DTYPES:
                plan_data["kv_indptr"] = _t.detach().cpu().contiguous()
                break
    if "kv_indices" not in plan_data:
        for _attr in _INDICES_ATTRS:
            _t = getattr(self, _attr, None)
            if _t is not None and hasattr(_t, "dtype") and str(_t.dtype) in _INT_DTYPES:
                plan_data["kv_indices"] = _t.detach().cpu().contiguous()
                break

    # 把 plan 数据合并进 captured，统一写入 .pt
    for def_key, t in plan_data.items():
        if def_key == "_sm_scale":
            captured["kwarg_sm_scale"] = {"type": "scalar", "value": t}
        else:
            # int tensor 用 full 类型保存完整数据
            captured[f"kwarg_{def_key}"] = {"type": "full", "tensor": t}
    if _VERBOSE and "prefill" in fi_api.lower():
        log_key = (fi_api, call_type)
        if log_key not in _seen_exec_logs:
            _seen_exec_logs.add(log_key)
            print(
                "[_fi_hook] OBSERVED_EXEC "
                f"fi_api={fi_api} call_type={call_type} "
                f"attrs={_wrapper_attr_debug(self)} "
                f"plan_keys={sorted(plan_data.keys())} "
                f"shapes={_call_shape_debug(args, kwargs)}",
                flush=True,
            )
    # 校验：definition 期望的 int tensor 是否都被捕获到
    _incomplete = False
    expected_int_inputs = set()
    for defn in info.get("definitions", []):
        for inp_name, inp_spec in defn.get("inputs", {}).items():
            if isinstance(inp_spec, dict) and inp_spec.get("dtype") in ("int32", "int64"):
                expected_int_inputs.add(inp_name)
    if expected_int_inputs:
        captured_keys = {k.removeprefix("kwarg_") for k in captured if k.startswith("kwarg_")}
        missing = expected_int_inputs - captured_keys - {"_sm_scale"}
        if missing:
            _incomplete = True
            warn_key = (fi_api, frozenset(missing))
            if warn_key not in _seen_plan_warnings:
                _seen_plan_warnings.add(warn_key)
                print(
                    f"[_fi_hook] WARNING: missing int tensor fields for {fi_api}: "
                    f"{sorted(missing)}. "
                    f"PLAN_KWARG_MAP may need updating for this FlashInfer version. "
                    f"captured_keys={sorted(captured_keys)} "
                    f"plan_keys={sorted(plan_data.keys())}",
                    flush=True,
                )
    _save({"fi_api": fi_api, "call_type": call_type, "captured": captured,
           "_incomplete": _incomplete})


# ── patch 安装 ────────────────────────────────────────────────────────────────

def _hook_wrapper(mod, cls_name, cls, fi_api, info):
    """为 Wrapper 类（如 BatchDecodeWithPagedKVCacheWrapper）安装 patch。

    Wrapper 类有两类关键方法：
    - plan()/begin_forward()：接收结构性 int tensor（kv_indptr、kv_indices 等）
    - run()/forward()：接收 float tensor（q、kv 等），是真正要 capture 的执行入口

    patch 策略：
    1. 如果 needs_plan=True，patch plan()/begin_forward()，暂存结构性 tensor
    2. patch run()/forward()，捕获执行 tensor，合并 plan 数据，写入 .pt 文件
    """
    import functools as _functools
    import inspect as _inspect

    # 找 plan/begin_forward（不同版本 FlashInfer 用不同名字）。
    # 有些 wrapper 同时暴露二者，但真实推理只走 begin_forward；这里两个都尝试 patch。
    plan_methods = [
        (name, getattr(cls, name, None))
        for name in ("plan", "begin_forward")
        if getattr(cls, name, None) is not None
    ]
    exec_methods = [
        (name, getattr(cls, name, None))
        for name in ("run", "forward", "forward_return_lse")
        if getattr(cls, name, None) is not None
    ]

    # ── patch plan()/begin_forward() ──
    if info.get("needs_plan"):
        for plan_name, orig_plan in plan_methods:
            if getattr(orig_plan, "_fi_fast_hooked", False):
                continue
            # 用 inspect.signature() 读真实参数名（去掉 self），作为闭包变量传给 _plan()。
            # 这样无论哪个 Wrapper 被 patch，都用该类自己的真实签名，不依赖硬编码顺序。
            try:
                _plan_param_names = [
                    p for p in _inspect.signature(orig_plan).parameters if p != "self"
                ]
            except (ValueError, TypeError):
                _plan_param_names = []  # inspect 失败时退化为空列表，位置参数将被跳过

            def _make_plan_hook(_orig_plan, _plan_name, _param_names):
                @_functools.wraps(_orig_plan)
                def _plan(self, *a, **kw):
                    try:
                        # 把 plan()/begin_forward() 里的 int tensor 提取出来，暂存到实例属性。
                        # run() 执行时再来取（准备阶段和 run 是同一个 wrapper 实例）。
                        plan_data = _extract_plan_tensors(a, kw, _param_names)
                        sm = kw.get("sm_scale") or kw.get("scale")
                        if sm is not None:
                            plan_data["_sm_scale"] = float(sm)
                        if plan_data:
                            self._fi_plan_data = plan_data
                            tid = _threading.get_ident()
                            _latest_plan_data[(fi_api, tid)] = plan_data
                            _latest_plan_data[(fi_api, None)] = plan_data
                        if _VERBOSE and "prefill" in fi_api.lower():
                            log_key = (fi_api, _plan_name)
                            if log_key not in _seen_plan_logs:
                                _seen_plan_logs.add(log_key)
                                print(
                                    "[_fi_hook] OBSERVED_PLAN "
                                    f"fi_api={fi_api} method={_plan_name} "
                                    f"attrs={_wrapper_attr_debug(self)} "
                                    f"param_names={list(_param_names)} "
                                    f"plan_keys={sorted(plan_data.keys())} "
                                    f"shapes={_call_shape_debug(a, kw)}",
                                    flush=True,
                                )
                    except Exception:
                        pass  # patch 出错不能影响正常推理，静默忽略
                    return _orig_plan(self, *a, **kw)  # 调用原始方法，保持正常流程
                _plan._fi_fast_hooked = True
                return _plan

            setattr(cls, plan_name, _make_plan_hook(orig_plan, plan_name, _plan_param_names))

    # ── patch run()/forward() ──
    for exec_name, orig_exec in exec_methods:
        if getattr(orig_exec, "_fi_fast_hooked", False):
            continue

        try:
            _exec_sig = _inspect.signature(orig_exec)
            _exec_param_names = [
                name
                for name, param in _exec_sig.parameters.items()
                if name != "self"
                and param.kind
                not in (_inspect.Parameter.VAR_POSITIONAL, _inspect.Parameter.VAR_KEYWORD)
            ]
            if not _exec_param_names or any(
                p.kind == _inspect.Parameter.VAR_POSITIONAL
                for p in _exec_sig.parameters.values()
            ):
                _exec_param_names = _definition_input_names(info)
        except (ValueError, TypeError):
            _exec_param_names = _definition_input_names(info)

        def _make_exec_hook(_orig_exec, _exec_name, _param_names):
            @_functools.wraps(_orig_exec)
            def _exec(self, *a, **kw):
                if getattr(_exec_guard, "active", False):
                    return _orig_exec(self, *a, **kw)

                captured_ok = False
                try:
                    _capture_wrapper_execution(self, a, kw, fi_api, _exec_name, info, _param_names)
                    captured_ok = True
                except Exception:
                    pass  # patch 出错不能影响正常推理

                # If forward() calls run() internally, suppress the nested run
                # capture only when the outer forward capture was saved.
                suppress_nested = _exec_name in {"forward", "forward_return_lse"} and captured_ok
                if suppress_nested:
                    _exec_guard.active = True
                try:
                    return _orig_exec(self, *a, **kw)
                finally:
                    if suppress_nested:
                        _exec_guard.active = False
            _exec._fi_fast_hooked = True
            return _exec

        setattr(cls, exec_name, _make_exec_hook(orig_exec, exec_name, _exec_param_names))


def _hook_function(mod, func_name, func, fi_api, info):
    """为普通函数（如 top_k_top_p_sampling_from_probs）安装 patch。

    普通函数没有 plan/run 的区别，直接 patch 函数本身。
    捕获所有参数（args + kwargs），写入 .pt 文件。
    """
    import functools as _functools
    import inspect as _inspect
    if getattr(func, "_fi_fast_hooked", False):
        return  # 已经 patch 过了，跳过
    try:
        _sig = _inspect.signature(func)
        _param_names = [
            name
            for name, param in _sig.parameters.items()
            if param.kind
            not in (_inspect.Parameter.VAR_POSITIONAL, _inspect.Parameter.VAR_KEYWORD)
        ]
        if not _param_names or any(
            p.kind == _inspect.Parameter.VAR_POSITIONAL for p in _sig.parameters.values()
        ):
            _param_names = _definition_input_names(info)
    except (ValueError, TypeError):
        _param_names = _definition_input_names(info)
    _names_to_dump = _full_tensor_names(info)

    @_functools.wraps(func)
    def _hooked(*a, **kw):
        try:
            captured = _capture_call(
                a,
                kw,
                full_tensor_names=_names_to_dump,
                param_names=_param_names,
            )
            _save({"fi_api": fi_api, "call_type": "function", "captured": captured})
        except Exception:
            pass
        return func(*a, **kw)
    _hooked._fi_fast_hooked = True
    # 记录原始函数 id → hooked 版本，供后续替换 sglang 缓存的旧引用
    _orig_to_hooked[id(func)] = _hooked
    setattr(mod, func_name, _hooked)


def _patch_sglang_refs(mod):
    """替换 sglang 模块里已缓存的旧 flashinfer 函数引用。

    问题背景：SGLang 在 import 时会做：
        from flashinfer.sampling import top_k_top_p_sampling_from_probs
    这行代码把函数对象缓存在 sglang 自己的命名空间里。
    之后我们给 flashinfer.sampling 模块上的函数装 patch，
    但 sglang 模块里缓存的还是旧的（未 patch）引用，patch 不生效。

    这个函数扫描 sglang 模块的所有属性，把旧引用替换成 hooked 版本。
    """
    if not _orig_to_hooked:
        return  # 还没有任何 hooked 函数，不用处理
    try:
        for attr_name in list(vars(mod)):
            val = getattr(mod, attr_name, None)
            if val is None:
                continue
            hooked = _orig_to_hooked.get(id(val))  # 查这个引用是否是某个原始函数
            if hooked is not None:
                try:
                    setattr(mod, attr_name, hooked)  # 替换为 hooked 版本
                except Exception:
                    pass
    except Exception:
        pass


def _try_install():
    """遍历 _SPECS，对所有已加载的 FlashInfer 模块尝试安装 patch。

    之所以要"尝试"，是因为 patch 只能在模块已经 import 之后才能安装。
    FlashInfer 是懒加载的，某些子模块在 SGLang 启动时还没有 import，
    所以这个函数会被 _wrapped_import 反复调用，每次有新模块加载就再试一次。
    """
    for fi_api, info in _SPECS.items():
        if fi_api in _hooked:
            continue  # 已经 patch 过了
        # fi_api 格式：flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper
        # 拆分为模块路径（flashinfer.decode）和目标名（BatchDecodeWithPagedKVCacheWrapper）
        parts = fi_api.rsplit(".", 1)
        if len(parts) != 2:
            continue
        mod_path, attr = parts
        mod = _sys.modules.get(mod_path)  # 查模块是否已加载
        if mod is None:
            continue  # 还没加载，等下次
        target = getattr(mod, attr, None)
        if target is None:
            continue
        _hooked.add(fi_api)
        if _VERBOSE:
            print(f"[_fi_hook] PATCHED: {fi_api} (is_wrapper={info['is_wrapper']})", flush=True)
        if info["is_wrapper"]:
            _hook_wrapper(mod, attr, target, fi_api, info)
        else:
            _hook_function(mod, attr, target, fi_api, info)


# ── import hook：拦截所有 import，在新模块加载后尝试安装 patch ─────────────────
_orig_import = _builtins.__import__

def _wrapped_import(name, *a, **kw):
    """替换内置 __import__，在每次 import 后触发 patch 安装。

    为什么要这样做：
    - sitecustomize.py 在 SGLang 最早的 import 之前执行，
      此时 flashinfer 还没有被加载，无法直接 patch。
    - 通过替换 __import__，可以在 flashinfer 任何子模块被 import 的瞬间
      立即安装 patch，确保不会漏掉。
    """
    mod = _orig_import(name, *a, **kw)
    if any(name == p or name.startswith(p + ".") for p in ("flashinfer",)):
        # flashinfer 的任意子模块被 import → 尝试安装 patch
        _try_install()
    if any(name == p or name.startswith(p + ".") for p in ("sglang",)):
        # sglang 模块被 import → 替换其中缓存的旧 flashinfer 引用
        _patch_sglang_refs(mod)
    return mod

_builtins.__import__ = _wrapped_import

if _VERBOSE:
    print(f"[_fi_hook] LOADED OK pid={_os.getpid()} specs={list(_SPECS.keys())} capture_dir={_CAPTURE_DIR}", flush=True)

# 如果 flashinfer 在 sitecustomize.py 执行之前就已经加载了（极少见），
# 立即尝试一次 patch 安装
_try_install()

# 同理，修复已加载的 sglang 模块里的旧引用
for _mod_name, _mod in list(_sys.modules.items()):
    if _mod is not None and (
        _mod_name == "sglang" or _mod_name.startswith("sglang.")
    ):
        _patch_sglang_refs(_mod)
