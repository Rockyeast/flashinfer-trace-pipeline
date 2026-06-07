"""Worker-side probe runtime — injected via sitecustomize on Modal worker.

Loaded as worker_trace_runtime.py in the sitecustomize bundle.
Exposes install_from_env() which is called by sitecustomize.py at process startup.

工作原理概述：
  sitecustomize.py 在每个 Python 进程启动时自动执行，调用 install_from_env()。
  install_from_env() 创建 ProbeRuntime 实例并调用 install()（幂等，已安装则跳过），
  install() 做两件事：
    1. 对已加载的目标模块打 monkey-patch（wrap 每个函数/方法）
    2. 安装 import hook（替换 builtins.__import__，确保后续动态导入的模块也被 patch）
  每次目标函数被调用，wrapper 把调用信息（参数 shape 等）写入 raw/pid-{pid}.jsonl。
"""

from __future__ import annotations

import atexit
import builtins
import functools
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# 只对这些模块前缀的函数/类打 patch，其余模块完全不碰
MODULE_PREFIXES = ("sgl_kernel", "flashinfer", "sglang.srt.layers")

# 即使命中 MODULE_PREFIXES，也不 patch 这些前缀的模块（工具/编译模块，不含 kernel）
# flashinfer.jit：JIT 编译模块，参数包含完整 CUDA C++ 源码字符串，patch 后 signature
# 体积极大（每条几百行），对 kernel 发现无任何贡献
EXCLUDE_PREFIXES = ("flashinfer.jit",)

# 需要在安装 import hook 之前强制完整加载的模块列表。
# 原因：Python 在模块体执行前就把模块名加入 sys.modules（CPython 规范行为）。
# 若 _wrapped_import 在某模块半加载时触发（由该模块内部的 sub-import 引起），
# _patch_loaded_modules() 会扫到该半加载模块，_patch_module_classes() 找不到类，
# 随后 patched_modules.add() 把它标记为"已完成"，类方法永远漏掉。
# 解法：在 hook 安装前调用 _preimport_modules()，保证这些模块被完整加载，
# 之后 _patch_loaded_modules() 能正确 patch 到所有类和方法。
PREIMPORT_MODULES = (
    "sglang.srt.layers.layernorm",
    "sglang.srt.layers.linear",
    "sglang.srt.layers.logits_processor",
    "sglang.srt.layers.sampler",
    "sglang.srt.layers.rotary_embedding",
    "sglang.srt.layers.vocab_parallel_embedding",
    "sglang.srt.layers.quantization.unquant",
    "sglang.srt.layers.quantization.fp8_kernel",
    "sglang.srt.layers.quantization.fp8_utils",
)

# sglang.srt.layers 下已知的非 kernel 工具类，patch 它们只会产生噪声。
# 采用黑名单策略：默认 patch 所有类，只排除已知无关的。
# 新 kernel 类默认被捕获，发现噪声时加到这里即可。
EXCLUDE_CLASS_NAMES = frozenset({
    # 按需添加，例如：
    # "SomeUtilityClass",
})


def _callable_trace_id(obj: Any) -> str:
    """Return the official-style dotted path for a callable."""
    if isinstance(obj, functools.partial):
        obj = obj.func
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if qualname is None:
        qualname = getattr(obj, "__name__", None)
    if qualname is None:
        cls = type(obj)
        module = module or getattr(cls, "__module__", None)
        qualname = getattr(cls, "__qualname__", cls.__name__)
    return f"{module}.{qualname}" if module else qualname




# 从 self（第一个参数，即 PyTorch Module 实例）提取的属性名列表。
# 这些属性描述 kernel 的"配置"，如 hidden_size、num_heads 等。
# 用于 signature 中的 "self" 字段，帮助区分不同尺寸的同类 kernel。
SELF_ATTR_KEYS = (
    "layer_id",
    "hidden_size",
    "head_size",
    "head_dim",
    "v_head_dim",
    "tp_q_head_num",
    "tp_k_head_num",
    "num_heads",
    "num_kv_heads",
    "num_local_heads",
    "input_size",
    "input_size_per_partition",
    "output_size",
    "output_size_per_partition",
    "intermediate_size",
    "intermediate_size_per_partition",
    "num_experts",
    "num_local_experts",
    "page_size",
    "vocab_size",
    "top_k",
    "variance_epsilon",
    "is_neox_style",
    "use_fallback_kernel",
)

# 从非 tensor/scalar 参数对象（如 ForwardBatch、SamplingBatchInfo 等）提取的属性。
# 这些对象描述单次推理请求的"运行时状态"，如序列长度、batch 大小等。
OBJECT_ATTR_KEYS = (
    "forward_mode",
    "seq_lens",
    "seq_lens_sum",
    "extend_seq_lens",
    "extend_num_tokens",
    "positions",
    "out_cache_loc",
    "top_ps",
    "top_ks",
    "min_ps",
    "temperatures",
    "sampling_seed",
    "is_all_greedy",
    "need_top_p_sampling",
    "need_top_k_sampling",
    "need_min_p_sampling",
)

# LinearMethod.apply(layer, x, ...) passes the actual SGLang Linear layer as a
# non-self positional object. Keep only dimension-like attrs so GEMM discovery
# can recover N/K without expanding arbitrary runtime objects.
LINEAR_OBJECT_ATTR_KEYS = (
    "input_size",
    "input_size_per_partition",
    "output_size",
    "output_size_per_partition",
    "gather_output",
    "skip_bias_add",
)

# 全局单例：每个进程只有一个 ProbeRuntime 实例
_RUNTIME: "ProbeRuntime | None" = None


def _module_is_interesting(name: str) -> bool:
    """判断模块名是否属于需要 patch 的目标模块前缀（且不在排除黑名单里）。"""
    if any(name == ex or name.startswith(ex + ".") for ex in EXCLUDE_PREFIXES):
        return False
    return any(name == prefix or name.startswith(prefix + ".") for prefix in MODULE_PREFIXES)


def _is_patchable_callable(value: Any) -> bool:
    """判断一个对象是否可以被 monkey-patch（排除类、模块、Triton JIT 函数等）。"""
    if inspect.isclass(value) or inspect.ismodule(value):
        return False
    # Triton JITFunction 不能被 wrap（会破坏其 CUDA kernel 编译机制）
    if hasattr(value, "__class__") and value.__class__.__name__ == "JITFunction":
        return False
    return callable(value)


class ProbeRuntime:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"    # 存放每个进程的 .jsonl 原始事件文件
        self.meta_dir = self.output_dir / "meta"  # 存放进程元数据和分类信息
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.pid = os.getpid()
        self.seq = 0                              # 事件序号，单调递增（用于排序）
        self.patched_modules: set[str] = set()    # 已 patch 的模块名，避免重复 patch
        self.seen_trace_ids: set[str] = set()     # 已见过的 trace_id（首次出现时额外记录调用栈）
        self.seen_prefill_logs: set[str] = set()  # 已打印过的 prefill wrapper 选择日志
        self.original_import = builtins.__import__  # 保存原始 __import__，用于恢复和调用

        # 每个进程写一个独立的 .jsonl 文件，行是 JSON 对象（一行一个事件）
        self.raw_path = self.raw_dir / f"pid-{self.pid}.jsonl"
        self.raw_fp = self.raw_path.open("a", encoding="utf-8", buffering=1)  # 行缓冲（每行立即写盘）

        self._write_metadata()  # 进程启动时写一次元数据（pid / argv / 环境变量等）

    def install(self) -> "ProbeRuntime":
        """安装 probe runtime：预加载目标模块 → patch 已加载模块 → 安装 import hook。"""
        self._preimport_modules()      # 强制完整加载目标模块，避免半加载竞争条件导致类方法漏 patch
        self._patch_loaded_modules()   # 对 sys.modules 里已有的目标模块打 patch
        builtins.__import__ = self._wrapped_import  # 确保后续动态 import 的目标模块也被 patch
        atexit.register(self.close)    # 进程退出时自动 flush 并写分类文件
        print(
            f"[worker-trace] pid={self.pid} probe runtime installed output_dir={self.output_dir}",
            file=sys.stderr,
            flush=True,
        )
        return self

    def _preimport_modules(self) -> None:
        """在安装 import hook 之前，强制完整加载 PREIMPORT_MODULES 中列出的目标模块。

        必要性——半加载竞争条件（CPython 规范行为）：
          当 import sglang.srt.layers.layernorm 执行时，CPython 在执行 layernorm.py
          文件体之前就把空模块对象注册到 sys.modules。若 layernorm.py 内部有子 import，
          子 import 会触发已安装的 _wrapped_import，后者调用 _patch_loaded_modules()，
          此时 layernorm 模块已在 sys.modules 但仍处于半加载状态——顶层函数可能已定义，
          但 class RMSNorm 等尚未执行。_patch_module_classes() 找不到类，
          随后 patched_modules.add() 标记为"已完成"，类方法永远漏掉。

          解法：在 builtins.__import__ = _wrapped_import 之前调用此函数，
          使用原始 import（无任何 hook）强制完整加载目标模块，之后
          _patch_loaded_modules() 能看到全量类并正确 patch。
        """
        for module_name in PREIMPORT_MODULES:
            try:
                __import__(module_name)
            except Exception:
                pass  # 模块不存在或加载失败时跳过（不同 sglang 版本的路径可能有差异）

    def close(self) -> None:
        """进程退出时的清理：flush/close raw 文件，写分类 JSON，恢复 __import__。"""
        try:
            self.raw_fp.flush()
            self.raw_fp.close()
        except Exception:
            pass
        # 恢复原始 __import__（防止影响其他在同一进程内继续运行的代码）
        if builtins.__import__ is self._wrapped_import:
            builtins.__import__ = self.original_import

    def _write_metadata(self) -> None:
        """在进程启动时写入 meta/pid-{pid}.json，记录进程基本信息和相关环境变量。"""
        env_keys = (
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "SGLANG_DP_RANK",
            "SGLANG_WORKER_TRACE_DIR",
        )
        metadata = {
            "pid": self.pid,
            "ppid": os.getppid(),       # 父进程 pid（用于理解进程树结构）
            "argv": sys.argv,           # 启动命令行参数
            "cwd": os.getcwd(),         # 工作目录
            "python": sys.executable,   # Python 解释器路径
            "time": time.time(),        # 进程启动时间戳
            "env": {key: os.environ.get(key) for key in env_keys if key in os.environ},
        }
        meta_path = self.meta_dir / f"pid-{self.pid}.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


    def _wrapped_import(
        self,
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ):
        """替换 builtins.__import__ 的 hook：先执行真正的 import，再检查是否需要 patch 新模块。"""
        # 先执行正常的 import，得到模块对象
        module = self.original_import(name, globals, locals, fromlist, level)
        try:
            # 判断被导入的模块是否属于目标前缀
            is_interesting = _module_is_interesting(name)
            # from flashinfer.norm import xxx 这类 import，name="flashinfer.norm"，fromlist=["xxx"]
            # 需要额外检查 fromlist 里的名字是否命中前缀
            if not is_interesting and fromlist:
                for imported_name in fromlist:
                    if _module_is_interesting(f"{name}.{imported_name}"):
                        is_interesting = True
                        break
            if is_interesting:
                # 有新的目标模块被导入，重新扫描 sys.modules 并 patch 未处理的模块
                self._patch_loaded_modules()
        except Exception:
            pass  # import hook 内不允许抛异常，否则会破坏正常的 import 流程
        return module

    def _patch_loaded_modules(self) -> None:
        """扫描 sys.modules，对所有未处理的目标模块打 patch。"""

        for module_name, module in list(sys.modules.items()):
        # 遍历当前进程已加载的所有模块（sys.modules 是个字典：模块名→模块对象）
        # 用 list() 包一层是为了防止遍历过程中 sys.modules 被修改导致报错

            if module is None or module_name in self.patched_modules:
                continue  # None 是占位符；已 patch 的不重复处理
            # module is None：Python 有时用 None 作占位符（包加载过程中的内部状态），跳过
            # module_name in self.patched_modules：这个模块已经被 patch 过了，不要重复处理

            if not _module_is_interesting(module_name):
                continue
            # 检查模块名是否属于我们关心的前缀（sgl_kernel、flashinfer、sglang.srt.layers）
            # 不是的话直接跳过，不动它

            self._patch_module_functions(module_name, module)
            # 把这个模块里所有顶层的可调用函数（非类、非私有）替换成 wrapper
            # wrapper 会在函数被调用时把参数 shape 等信息写进 .jsonl 文件

            if module_name.startswith("sglang.srt.layers") or module_name.startswith("flashinfer"):
                self._patch_module_classes(module_name, module)
            # 只有 sglang.srt.layers.* 和 flashinfer.* 这两个包，才额外 patch 类的方法
            # （sgl_kernel 只有顶层函数，没有需要 patch 的类，所以排除在外）

            self.patched_modules.add(module_name)  # 标记为已 patch
            # 把这个模块名加入"已处理集合"，下次再调用 _patch_loaded_modules() 时跳过它
            # 避免同一个模块被重复 patch（每次 _wrapped_import 触发时都会调用这个函数）


    def _patch_module_functions(self, module_name: str, module: Any) -> None:
        """将模块顶层的可调用函数（非类、非私有、非 typing 辅助）替换为 wrapper。"""
        for attr_name, attr in vars(module).items():
            if attr_name.startswith("_") or inspect.isclass(attr):
                continue  # 跳过私有名和类（类方法单独处理）
            # 跳过 typing 模块的辅助符号（Optional、Union、Any 等），它们不是真正的函数
            if attr_name in (
                "dataclass", "runtime_checkable", "NamedTuple", "Protocol",
                "Optional", "Union", "Any", "Dict", "List", "Set", "Tuple",
                "Callable", "Type", "TypeVar", "Generic", "Iterator",
                "Iterable", "Mapping", "Sequence", "Literal", "cast", "overload",
            ):
                continue
            if not _is_patchable_callable(attr):
                continue
            if getattr(attr, "__sglang_worker_trace_wrapped__", False):
                continue  # 已经被 wrap 过，跳过（防止双重 wrap）
            trace_id = _callable_trace_id(attr)
            wrapped = self._wrap_callable("module_function", trace_id, attr)
            try:
                setattr(module, attr_name, wrapped)
            except Exception:
                continue  # 某些模块属性是只读的，跳过

    def _patch_module_classes(self, module_name: str, module: Any) -> None:
        """对模块内满足条件的类，patch 其特定方法（forward / run / apply 等）。
        黑名单策略：默认 patch 所有目标模块的类，仅排除 EXCLUDE_CLASS_NAMES 中已知无关类。
        """
        for _, cls in vars(module).items():
            if not inspect.isclass(cls):
                continue
            if cls.__name__ in EXCLUDE_CLASS_NAMES:
                continue
            for method_name in self._method_names_for_class(module_name, cls.__name__):
                self._patch_class_method(cls, method_name)

    def _method_names_for_class(self, module_name: str, cls_name: str) -> set[str]:
        """根据类名返回需要 patch 的方法名集合。
        基础方法（forward / apply）对所有类都 patch；
        特殊类（Attention / MoE / Sampler 等）额外 patch 专有方法。
        """
        # 所有类都 patch 的基础方法
        methods = {
            "forward", "forward_cuda", "forward_cpu", "forward_xpu",
            "forward_npu", "_forward_impl", "apply",
        }
        if "Backend" in cls_name or "Attention" in cls_name or "Wrapper" in cls_name:
            # Attention backend 类：额外 patch prefill/decode 入口和 run/plan
            methods.update({"forward_extend", "forward_decode", "run", "plan"})
        if "MoE" in cls_name or cls_name == "TopK" or ".moe." in module_name:
            # MoE 类：额外 patch 核心计算方法
            methods.update({"forward_impl", "run_moe_core"})
        if cls_name == "Sampler":
            # Sampler 类：patch 采样方法
            methods.update({"_sample_from_probs", "_sample_from_logprobs"})
        if cls_name == "LogitsProcessor":
            # LogitsProcessor 类：patch logits 计算方法
            methods.update({"_compute_lm_head", "_get_logits"})
        if module_name.startswith("flashinfer"):
            # flashinfer 的类：额外 patch run / plan / begin_forward / end_forward
            methods.update({"run", "plan", "begin_forward", "end_forward"})
        return methods

    def _patch_class_method(self, cls: type, method_name: str) -> None:
        """将类的指定方法替换为 wrapper，处理 classmethod / staticmethod 的特殊情况。"""
        descriptor = cls.__dict__.get(method_name)
        if descriptor is None:
            return  # 该类没有这个方法，跳过

        is_classmethod = isinstance(descriptor, classmethod)
        is_staticmethod = isinstance(descriptor, staticmethod)
        # 取出真正的函数对象（classmethod/staticmethod 包了一层）
        func = descriptor.__func__ if (is_classmethod or is_staticmethod) else descriptor

        if not callable(func):
            return
        if getattr(func, "__sglang_worker_trace_wrapped__", False):
            return  # 已 wrap 过，跳过

        trace_id = _callable_trace_id(func)
        wrapped = self._wrap_callable("class_method", trace_id, func)
        # 保持原来的 classmethod / staticmethod 装饰器
        replacement = wrapped
        if is_classmethod:
            replacement = classmethod(wrapped)
        elif is_staticmethod:
            replacement = staticmethod(wrapped)

        try:
            setattr(cls, method_name, replacement)
        except Exception:
            return  # 某些类的方法是只读的（C 扩展类等），忽略

    def _wrap_callable(self, category: str, trace_id: str, func: Any):
        """为函数/方法创建 wrapper：调用时记录事件到 .jsonl 文件，然后调用原始函数。

        category: "module_function" 或 "class_method"
        trace_id: 如 "flashinfer.norm.RMSNormKernel.forward"
        func:     原始函数对象
        """
        # 在 patch 时提取原始函数的参数名列表（一次性开销，不在每次调用时执行）。
        # 解决 args 匿名问题：probe 记录的 args[0], args[1] 没有参数名，
        # 但 inspect.signature() 可以拿到函数定义时的参数名（如 qo_indptr, paged_kv_indptr）。
        # 只在首次见到该 trace_id 时写入 raw JSONL（和 stack 相同策略，减少文件体积）。
        param_names: list[str] | None = None
        try:
            sig = inspect.signature(func)
            param_names = [
                p.name for p in sig.parameters.values()
                if p.name != "self"  # 排除 self（类方法的第一个参数）
            ]
        except (ValueError, TypeError):
            pass  # C 扩展函数、内置函数等无法 inspect，跳过

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self._record_event(category, trace_id, args, kwargs, param_names=param_names)
            return func(*args, **kwargs)  # 调用原始函数，不影响其返回值

        wrapper.__sglang_worker_trace_wrapped__ = True  # 标记已 wrap，防止二次 wrap
        return wrapper

    def _record_event(
        self,
        category: str,
        trace_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        param_names: list[str] | None = None,
    ) -> None:
        """将一次函数调用记录为 JSON 事件，追加写入 raw/pid-{pid}.jsonl。

        事件字段：
          seq          — 单调递增序号（同一进程内全局唯一）
          time         — 调用时间戳（Unix epoch）
          pid          — 进程 id
          category     — "module_function" 或 "class_method"
          trace_id     — 函数全限定名（如 flashinfer.norm.RMSNormKernel.forward）
          signature    — 参数摘要（self 属性 + args 的 tensor shape/dtype + kwargs）
          stack        — 调用栈（仅首次见到该 trace_id 时才记录，减少文件大小）
          param_names  — 函数参数名列表（仅首次见到该 trace_id 时记录），
                         用于将匿名的 args[0], args[1] 映射回真实参数名
                         （如 ["qo_indptr", "paged_kv_indptr", "paged_kv_indices", ...]）
        """
        self.seq += 1
        event = {
            "seq": self.seq,
            "time": time.time(),
            "pid": self.pid,
            "category": category,
            "trace_id": trace_id,
            "signature": self._summarize_call(args, kwargs),  # 参数形状摘要
        }
        if (
            "flashinfer.prefill.BatchPrefill" in trace_id
            and trace_id not in self.seen_prefill_logs
        ):
            self.seen_prefill_logs.add(trace_id)
            summary = json.dumps(event["signature"], ensure_ascii=False)[:1200]
            print(
                f"[probe-runtime] OBSERVED_PREFILL trace_id={trace_id} "
                f"signature={summary}",
                flush=True,
            )
        # 每个 trace_id 只在首次出现时记录调用栈和参数名（减少文件大小）
        if trace_id not in self.seen_trace_ids:
            self.seen_trace_ids.add(trace_id)
            event["stack"] = traceback.format_stack(limit=10)[:-1]  # 去掉最后一帧（就是这里本身）
            # 参数名列表：将 args[0] 对应到 "qo_indptr" 等真实参数名，
            # 下游 parse 阶段用于生成 definition inputs 字段名，替代硬编码模板
            if param_names:
                event["param_names"] = param_names

        self.raw_fp.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.raw_fp.flush()  # 行缓冲模式下每行自动 flush，确保崩溃时数据不丢失

    def _summarize_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """将函数调用的参数整理为结构化摘要。

        如果 args[0] 不是 tensor/scalar（即它是一个 nn.Module 实例），
        则单独提取其属性（hidden_size 等）作为 "self" 字段；
        其余位置参数取前 6 个；kwargs 取前 8 个。
        """
        self_summary = None
        positional_args = args
        if args and not self._is_likely_tensor_or_scalar(args[0]):
            # args[0] 是 self（nn.Module 实例），提取其配置属性
            # is_module=True：用递归 vars() 方式收集所有基本类型属性（含嵌套子对象）
            self_summary = self._summarize_object(args[0], include_attrs=True, is_module=True)
            positional_args = args[1:]  # 去掉 self，剩余才是真正的位置参数
        return {
            "self": self_summary,                                                    # Module 配置属性
            "args": [self._summarize(value) for value in positional_args[:6]],      # 位置参数（最多6个）
            "kwargs": {key: self._summarize(value) for key, value in list(kwargs.items())[:8]},  # 关键字参数（最多8个）
        }

    def _summarize(self, value: Any, depth: int = 0) -> Any:
        """递归摘要任意 Python 对象：
          - tensor → shape/dtype/numel（叶节点，depth 截断前优先处理）
          - None/bool/int/float/str → 原值（叶节点，同上）
          - list/tuple → 类型 + 长度 + 前4个元素的摘要
          - dict → 类型 + 长度 + 前6个键值对的摘要
          - 其他对象 → 类名 + 属性摘要
          - depth > 2 时对容器/对象截断（避免无限递归）
          注：tensor 和标量是叶节点，不会继续递归，无需受 depth 截断约束。
        """
        # 叶节点优先：tensor 和标量不会继续递归，无论在第几层都安全记录，
        # 必须放在 depth 截断之前，否则深层 tensor 会被 max_depth 静默丢失。
        tensor_summary = self._summarize_tensor(value)
        if tensor_summary is not None:
            return tensor_summary
        if value is None or isinstance(value, (bool, int, float, str)):
            return value  # 标量直接返回原值

        # 只对"还需要继续递归展开"的容器/对象做 depth 截断，防止循环引用死循环
        if depth > 2:
            return {"type": "max_depth", "class": type(value).__name__}
        if isinstance(value, (list, tuple)):
            return {
                "type": type(value).__name__,
                "len": len(value),
                "items": [self._summarize(item, depth + 1) for item in value[:4]],  # 最多4个元素
            }
        if isinstance(value, dict):
            return {
                "type": "dict",
                "len": len(value),
                "items": {
                    str(key): self._summarize(item, depth + 1)
                    for key, item in list(value.items())[:6]  # 最多6个键值对
                },
            }
        return self._summarize_object(value, include_attrs=True, depth=depth)

    def _summarize_tensor(self, value: Any) -> dict[str, Any] | None:
        """如果 value 是 torch.Tensor，返回 shape/dtype/numel 摘要；否则返回 None。

        注意：torch 通过 sys.modules 动态查找，不在模块级 import，
        避免在 torch 未安装的环境下报错。
        """
        torch_mod = sys.modules.get("torch")
        if torch_mod is None:
            return None
        tensor_type = getattr(torch_mod, "Tensor", None)
        if tensor_type is None or not isinstance(value, tensor_type):
            return None
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),  # 去掉 "torch." 前缀，更简洁
            # 注意：不记录 device（cuda:0 / cuda:1），避免多 GPU 时同一 kernel 被分为不同 signature
            # 注意：不记录 numel（可从 shape 乘积算出，下游 parse 不读它，去掉减少文件体积）
        }

    def _collect_module_attrs(
        self, obj: Any, depth: int = 0, max_depth: int = 3, _seen: set | None = None
    ) -> dict[str, Any]:
        """递归遍历对象的 __dict__（及其子对象的 __dict__），收集所有基本类型（int/float/str/bool）属性。

        设计目标：替代静态 SELF_ATTR_KEYS 白名单，自动发现 nn.Module 及其嵌套子对象
        里的所有标量配置属性（如 hidden_size、head_dim，乃至深层的 n_group 等）。

        规则：
          - 只收录 int/float/str/bool 类型的叶节点值，容器和复杂对象只用于递归展开
          - 跳过以 '_' 开头的私有属性（nn.Module 内部簿记属性）
          - 用 id() 集合避免循环引用死循环
          - 同名冲突时子对象的值覆盖父对象（实际极少发生）
          - max_depth 限制递归层数，防止极端情况下的性能问题
        """
        if _seen is None:
            _seen = set()
        obj_id = id(obj)
        if obj_id in _seen or depth > max_depth:
            return {}
        _seen.add(obj_id)

        result: dict[str, Any] = {}
        try:
            obj_dict = vars(obj)
        except TypeError:
            return result  # 内置类型等没有 __dict__，跳过

        for k, v in obj_dict.items():
            if k.startswith("_"):
                continue  # 跳过私有/内部属性（_version、_modules、_parameters 等）
            if isinstance(v, (int, float, str, bool)):
                result[k] = v  # 基本类型叶节点，直接收录
            elif v is not None and hasattr(v, "__dict__") and not isinstance(v, type):
                # 子对象：递归进去继续找基本类型属性
                sub = self._collect_module_attrs(v, depth + 1, max_depth, _seen)
                result.update(sub)  # 合并（子对象的属性名优先覆盖，极少冲突）

        return result

    def _summarize_object(
        self, value: Any, include_attrs: bool = False, depth: int = 0,
        is_module: bool = False,
    ) -> dict[str, Any]:
        """摘要任意对象：记录类名，可选地提取属性值。

        目的：把一个 Python 对象变成一个 JSON 可序列化的字典，
             只保留我们关心的属性，丢掉不关心的内容。

        参数：
            value        — 要摘要的对象（可以是 Module 实例、ForwardBatch、任何东西）
            include_attrs — 是否提取属性；False 时只记类名，不读任何属性
            depth        — 当前递归深度（防止无限递归）
            is_module    — True 表示 value 是 nn.Module（方法的 self 参数），
                           此时用递归 vars() 方式收集所有基本类型属性（含嵌套子对象）；
                           False 时用 OBJECT_ATTR_KEYS 白名单（适用于 ForwardBatch 等运行时对象）
        """
        # ── 第一步：永远记录对象的"类名" ──
        # 例如 value 是一个 GemmaRMSNorm 实例，就记成：
        #   {"type": "object", "class": "sglang.srt.layers.layernorm.GemmaRMSNorm"}
        # 这样 parse 阶段能知道这是哪种类型的对象，即使没有任何属性也有参考价值。
        summary = {
            "type": "object",
            "class": f"{value.__class__.__module__}.{value.__class__.__name__}",  # 全限定类名，格式：模块路径.类名
        }

        # ── 第二步：如果调用者不需要属性，直接返回只有类名的摘要 ──
        # 某些场景（如递归处理嵌套对象）不需要深挖属性，避免信息爆炸。
        if not include_attrs:
            return summary

        attrs: dict[str, Any] = {}

        if is_module:
            # ── 第三步 A（nn.Module 路径）：递归 vars() 收集所有基本类型属性 ──
            # 替代静态 SELF_ATTR_KEYS 白名单：自动覆盖所有当前和未来的 int/float/str/bool
            # 属性，包括深层嵌套子对象（如 quant_method.quant_config.n_group）。
            raw_attrs = self._collect_module_attrs(value)
            # 将基本类型值直接存入 attrs（不再 _summarize，已经是叶节点）
            attrs = raw_attrs
        else:
            class_path = summary["class"]
            if (
                class_path.startswith("sglang.srt.layers.linear.")
                and "Linear" in class_path.rsplit(".", 1)[-1]
            ):
                raw_attrs = self._collect_module_attrs(value, max_depth=2)
                attrs.update({
                    key: raw_attrs[key]
                    for key in LINEAR_OBJECT_ATTR_KEYS
                    if key in raw_attrs
                })

            # ── 第三步 B（运行时对象路径）：保留 OBJECT_ATTR_KEYS 白名单 ──
            # ForwardBatch、SamplingBatchInfo 等运行时对象的 __dict__ 包含大量
            # tensor/list 字段，全量递归会引入过多噪音；白名单精准控制要记录的字段。
            for attr_name in OBJECT_ATTR_KEYS:
                if not hasattr(value, attr_name):
                    continue
                try:
                    attr_value = getattr(value, attr_name)
                except Exception:
                    continue
                attrs[attr_name] = self._summarize(attr_value, depth + 1)

        # ── 第四步：把收集到的属性塞进摘要字典 ──
        # 只有真的读到了至少一个属性才加 "attrs" 字段，避免空字典污染输出
        if attrs:
            summary["attrs"] = attrs

        # 最终返回的 summary 长这样（以 GemmaRMSNorm 为例）：
        # {
        #   "type": "object",
        #   "class": "sglang.srt.layers.layernorm.GemmaRMSNorm",
        #   "attrs": {
        #     "hidden_size": 2048,
        #     "variance_epsilon": 1e-06
        #   }
        # }
        return summary

    def _is_likely_tensor_or_scalar(self, value: Any) -> bool:
        """判断 args[0] 是否是 tensor 或标量（而非 nn.Module 实例）。
        用于区分"类方法的 self 参数"和"普通位置参数"。
        """
        if value is None or isinstance(value, (bool, int, float, str)):
            return True
        return self._summarize_tensor(value) is not None


# ──────────────────────────────────────────────────────────────────────────────
# sitecustomize.py 调用的入口函数
# ──────────────────────────────────────────────────────────────────────────────

def install_from_env() -> ProbeRuntime:
    """sitecustomize.py 在每个 Python 进程启动时调用此函数。幂等：若已安装则直接返回已有实例。

    从 SGLANG_WORKER_TRACE_DIR 读取输出目录，安装 ProbeRuntime 单例。
    """
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    trace_dir = os.environ.get("SGLANG_WORKER_TRACE_DIR", "/tmp/probe-output")
    _RUNTIME = ProbeRuntime(output_dir=trace_dir).install()
    return _RUNTIME


if __name__ == "__main__":
    print("probe runtime imports successfully")
