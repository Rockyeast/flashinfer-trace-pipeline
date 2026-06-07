import ast
import importlib
import os
import sys
from pathlib import Path


# 进程级状态：同一个 Python 进程里 attach 过一次后，后续重复调用直接跳过。
_ATTACHED = False


def _log(message):
    """统一输出 runtime patch 日志，方便在 Modal 日志里 grep。"""
    print(f"[fi-trace-runtime-patch] {message}", file=sys.stderr, flush=True)


def _module_name_for_file(root, path):
    """把 FlashInfer 源码文件路径转换成 import module 名。

    例子：
      /root/local_flashinfer_patch/flashinfer/sampling.py
      -> flashinfer.sampling

      /root/local_flashinfer_patch/flashinfer/trace/templates/__init__.py
      -> flashinfer.trace.templates
    """
    rel = list(path.relative_to(root).with_suffix("").parts)
    if rel and rel[-1] == "__init__":
        rel = rel[:-1]
    return "flashinfer" + (("." + ".".join(rel)) if rel else "")


def _current_package_parts(root, path):
    """返回当前文件所在 package 的相对 parts，用于解析相对 import。"""
    rel = list(path.relative_to(root).with_suffix("").parts)
    if rel and rel[-1] == "__init__":
        return rel[:-1]
    return rel[:-1]


def _resolve_import_module(root, path, level, module):
    """把 ast.ImportFrom 的相对 import 解析成绝对 module 名。

    例如在 flashinfer/sampling.py 里：
      from .trace.templates.sampling import foo
    会被解析成：
      flashinfer.trace.templates.sampling
    """
    if level:
        current_pkg = _current_package_parts(root, path)
        keep = max(0, len(current_pkg) - (level - 1))
        parts = ["flashinfer", *current_pkg[:keep]]
        if module:
            parts.extend(module.split("."))
        return ".".join(parts)
    return module or ""


def _trace_name_from_decorator(decorator):
    """从 @flashinfer_api(trace=xxx) 里提取 xxx。

    返回值只是 template 变量名，例如 top_k_top_p_sampling_from_probs；
    template 变量来自哪个 module，要结合当前文件的 import 表判断。
    """
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    if isinstance(func, ast.Name):
        is_flashinfer_api = func.id == "flashinfer_api"
    elif isinstance(func, ast.Attribute):
        is_flashinfer_api = func.attr == "flashinfer_api"
    else:
        is_flashinfer_api = False
    if not is_flashinfer_api:
        return None
    for keyword in decorator.keywords:
        if keyword.arg != "trace":
            continue
        value = keyword.value
        if isinstance(value, ast.Name):
            return value.id
        if isinstance(value, ast.Attribute):
            return value.attr
    return None


def _discover_targets_from_source():
    """从 pinned FlashInfer source 里发现需要检查/attach 的目标 API。

    输出 target tuple：
      function: ("function", api_module, "", function_name, template_module, template_name)
      method:   ("method", api_module, class_name, method_name, template_module, template_name)

    这一步只静态扫描源码，不 import 当前容器里的 flashinfer，也不做 attach。
    """
    root = Path(os.environ.get("PROBE_LOCAL_FLASHINFER_PATCH_DIR", "/root/local_flashinfer_patch/flashinfer"))
    if not root.is_dir():
        return [], f"no_source_root:{root}"

    targets = []
    for path in sorted(root.rglob("*.py")):
        try:
            # 用 AST 看结构，比字符串 grep 更稳：能明确识别函数、类、decorator、import。
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        # 收集当前文件里的 "template 名 -> template module" 映射。
        # 看到 @flashinfer_api(trace=foo) 时，需要靠它知道 foo 来自哪个 module。
        imports = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            module_name = _resolve_import_module(root, path, node.level, node.module)
            for alias in node.names:
                imports[alias.asname or alias.name] = module_name

        api_module = _module_name_for_file(root, path)

        def visit(node, class_stack):
            # class_stack 非空表示当前正在扫描类里的 method。
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    visit(child, class_stack + [node.name])
                return
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                trace_name = None
                for decorator in node.decorator_list:
                    # 找 @flashinfer_api(trace=xxx)，拿到 xxx。
                    trace_name = _trace_name_from_decorator(decorator)
                    if trace_name:
                        break
                if trace_name:
                    # xxx 必须能在 import 表里找到来源，否则不知道该挂哪个 template。
                    template_module = imports.get(trace_name)
                    if template_module:
                        if class_stack:
                            targets.append((
                                "method",
                                api_module,
                                class_stack[-1],
                                node.name,
                                template_module,
                                trace_name,
                            ))
                        else:
                            targets.append((
                                "function",
                                api_module,
                                "",
                                node.name,
                                template_module,
                                trace_name,
                            ))
                for child in node.body:
                    visit(child, class_stack)

        for node in tree.body:
            visit(node, [])

    if not targets:
        return [], f"no_discovered_targets:{root}"
    return targets, f"discovered:{len(targets)}:{root}"


def _target_label(kind, module_name, class_name, attr_name, template_module, template_name):
    """把 target tuple 渲染成人能读的 API 路径，用于日志和 summary。"""
    if kind == "method":
        return f"{module_name}.{class_name}.{attr_name}"
    return f"{module_name}.{attr_name}"


def _get_target_object(target):
    """从当前容器实际安装的 flashinfer 包里取出 target 对象。

    注意：这里检查的是 runtime 里的真实对象，不是 pinned source 里的源码对象。
    """
    kind, module_name, class_name, attr_name, _template_module, _template_name = target
    mod = importlib.import_module(module_name)
    if kind == "method":
        cls = getattr(mod, class_name, None)
        if cls is None:
            return None
        return getattr(cls, attr_name, None)
    return getattr(mod, attr_name, None)


def _inspect_target(target):
    """检查单个 target 在当前 runtime 里的状态。

    返回状态：
      attached    API 存在，并且已经有 callable .fi_trace
      no_fi_trace API 存在，但没有 .fi_trace，需要 runtime attach
      missing     pinned source 里有这个 API，但当前容器的 flashinfer 没有
      failed      import/检查过程报错
    """
    label = _target_label(*target)
    try:
        obj = _get_target_object(target)
        if obj is None:
            return label, "missing", None
        if callable(getattr(obj, "fi_trace", None)):
            return label, "attached", None
        return label, "no_fi_trace", None
    except Exception as exc:
        return label, "failed", f"{type(exc).__name__}: {exc}"


def _inspect_existing_targets():
    """检查所有 source-discovered targets 在当前 runtime 里的 .fi_trace 覆盖。"""
    targets, discovery_status = _discover_targets_from_source()
    inspected = [_inspect_target(target) for target in targets]
    attached = [label for label, status, _ in inspected if status == "attached"]
    no_fi_trace = [label for label, status, _ in inspected if status == "no_fi_trace"]
    missing = [label for label, status, _ in inspected if status == "missing"]
    failed = [(label, err) for label, status, err in inspected if status == "failed"]
    return {
        "discovery_status": discovery_status,
        "total": len(targets),
        "attached": attached,
        "no_fi_trace": no_fi_trace,
        "missing": missing,
        "failed": failed,
    }


def _attach_function(module_name, attr_name, template_module, template_name):
    """给普通函数 API 挂 official .fi_trace。

    关键点：
      tpl 是 official TraceTemplate（生成 definition 的规则）
      _attach_fi_trace 是 FlashInfer 提供的底层绑定函数
    """
    api_logging = importlib.import_module("flashinfer.api_logging")
    mod = importlib.import_module(module_name)
    tpl = getattr(importlib.import_module(template_module), template_name)
    original = getattr(mod, attr_name, None)
    if original is None:
        return False
    wrapped = api_logging._attach_fi_trace(original, original, tpl)
    setattr(mod, attr_name, wrapped)
    return callable(getattr(wrapped, "fi_trace", None))


def _attach_method(module_name, class_name, method_name, template_module, template_name):
    """给 class method API 挂 official .fi_trace。"""
    api_logging = importlib.import_module("flashinfer.api_logging")
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name, None)
    if cls is None:
        return False
    tpl = getattr(importlib.import_module(template_module), template_name)
    original = getattr(cls, method_name, None)
    if original is None:
        return False
    wrapped = api_logging._attach_fi_trace(original, original, tpl)
    setattr(cls, method_name, wrapped)
    return callable(getattr(wrapped, "fi_trace", None))


def _attach_target(target):
    """根据 target 类型分发到 function attach 或 method attach。"""
    kind, module_name, class_name, attr_name, template_module, template_name = target
    label = _target_label(*target)
    try:
        if kind == "method":
            return label, _attach_method(
                module_name,
                class_name,
                attr_name,
                template_module,
                template_name,
            ), None
        return label, _attach_function(
            module_name,
            attr_name,
            template_module,
            template_name,
        ), None
    except Exception as exc:
        return label, False, f"{type(exc).__name__}: {exc}"


def _attach_all_targets():
    """对所有 source-discovered targets 尝试 runtime attach，并汇总结果。"""
    targets, discovery_status = _discover_targets_from_source()
    results = [_attach_target(target) for target in targets]
    attached = [label for label, ok, _ in results if ok]
    failed = [(label, err) for label, ok, err in results if not ok and err]
    missing = [label for label, ok, err in results if not ok and not err]
    return {
        "discovery_status": discovery_status,
        "total": len(targets),
        "attached": attached,
        "missing": missing,
        "failed": failed,
    }


def _with_count_fields(summary):
    """给 summary 增加 *_count 字段，方便日志和 aggregated_summary 展示。"""
    return {
        **summary,
        "attached_count": len(summary.get("attached", [])),
        "no_fi_trace_count": len(summary.get("no_fi_trace", [])),
        "missing_count": len(summary.get("missing", [])),
        "failed_count": len(summary.get("failed", [])),
    }


def _log_summary(summary):
    """打印 attach/preflight 摘要；列表过长时只打 preview。"""
    no_fi_trace = summary.get("no_fi_trace", [])
    _log(
        f"summary {summary['discovery_status']} total={summary['total']} "
        f"attached={len(summary['attached'])} "
        f"no_fi_trace={len(no_fi_trace)} "
        f"missing={len(summary['missing'])} failed={len(summary['failed'])}"
    )
    attached = summary["attached"]
    failed = summary["failed"]
    missing = summary["missing"]
    if attached:
        preview = ", ".join(attached[:40])
        suffix = "" if len(attached) <= 40 else f", ... +{len(attached) - 40}"
        _log(f"attached {preview}{suffix}")
    if failed:
        preview = "; ".join(f"{label} -> {err}" for label, err in failed[:20])
        suffix = "" if len(failed) <= 20 else f"; ... +{len(failed) - 20}"
        _log(f"failed {preview}{suffix}")
    if no_fi_trace:
        preview = ", ".join(no_fi_trace[:40])
        suffix = "" if len(no_fi_trace) <= 40 else f", ... +{len(no_fi_trace) - 40}"
        _log(f"no_fi_trace {preview}{suffix}")
    if missing:
        preview = ", ".join(missing[:40])
        suffix = "" if len(missing) <= 40 else f", ... +{len(missing) - 40}"
        _log(f"missing {preview}{suffix}")


def preflight_fi_trace_targets():
    """给 fi_trace_integration.py 调用的 preflight 入口。

    目的：在加载大模型前确认 official .fi_trace 是否可用。

    mode:
      existing      当前 runtime 已经有足够 .fi_trace，未做 runtime attach
      runtime_patch 当前 runtime 覆盖不足，已尝试按 pinned source 补 attach
    """
    global _ATTACHED
    existing = _inspect_existing_targets()
    if existing["attached"] and not existing["no_fi_trace"]:
        _ATTACHED = True
        return {
            **_with_count_fields(existing),
            "mode": "existing",
            "existing": _with_count_fields(existing),
            "final": _with_count_fields(existing),
        }

    summary = _attach_all_targets()
    _ATTACHED = bool(summary["attached"])
    return {
        **_with_count_fields(summary),
        "mode": "runtime_patch",
        "existing": _with_count_fields(existing),
        "final": _with_count_fields(summary),
    }


def attach_fi_trace():
    """给 sitecustomize.py 调用的 worker 启动入口。

    Python worker 启动后会自动 import sitecustomize.py；sitecustomize.py 再调用这里。
    这里会检查当前 runtime 的 .fi_trace 覆盖，不足时补 attach。
    """
    global _ATTACHED
    if _ATTACHED:
        return True
    existing = _inspect_existing_targets()
    if existing["attached"] and not existing["no_fi_trace"]:
        _ATTACHED = True
        _log("existing fi_trace coverage is sufficient; skipping runtime attach")
        _log_summary(existing)
        return True

    summary = _attach_all_targets()
    _ATTACHED = bool(summary["attached"])
    _log_summary(summary)
    return _ATTACHED
