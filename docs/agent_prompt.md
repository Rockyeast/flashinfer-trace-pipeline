# Onboard Model Proposal Prompt

这个文件只提供启动 `onboard-model-proposal` skill 的通用 prompt 模板。规则的唯一权威来源是 `.agents/skills/onboard-model-proposal/SKILL.md`。

使用前替换尖括号里的内容。不要保留尖括号占位符。

```text
使用 onboard-model-proposal skill，为 <MODEL_NAME> 生成 review-only proposal。

工作目录：
<REPO_ROOT>

输入：
- HF config: <HF_CONFIG_PATH_OR_OMIT>
- SGLang source root: <SGLANG_SOURCE_ROOT>
- SGLang model implementation hint: <SGLANG_MODEL_FILE_RELATIVE_TO_ROOT>
- FlashInfer source root: <FLASHINFER_SOURCE_ROOT_OR_OMIT>
- sgl-cookbook root: <SGL_COOKBOOK_ROOT_OR_OMIT>
- diagnostics:
  - first-pass: omit
  - repair-pass: <RUN_DIR>/proposal/repair_prompt.md
- run dir: <RUN_DIR>

输出限制：
- 只允许生成或修改：
  - <RUN_DIR>/proposal/architecture.md
  - <RUN_DIR>/proposal/candidate_targets.json
  - <RUN_DIR>/proposal/review_checklist.md
  - <RUN_DIR>/proposal/definitions/... only for non-FI definition_source=agent drafts
  - <RUN_DIR>/proposal/definition_hints/... only for non-FI definition_source=agent drafts
  - <RUN_DIR>/config/run_config.json
- 不要 apply。
- 不要生成正式 approved_targets.json。
- 不要生成正式 definitions/、workloads/、blob/。
- 不要运行 Modal。
- 不要运行正式 probe/collect/validate。
- 不要提交 git commit。
- 直接写 <RUN_DIR>/config/run_config.json；同时在 review_checklist.md 里说明 GPU、TP、image、CUDA graph、batch_sizes、max_new_tokens、supplemental_runs、max_captures_per_target 的证据或不确定性。
- 正式 collect 固定读取仓库根目录 sharegpt_100.json 作为 prompt source；缺少 reviewed batch_sizes、max_new_tokens、supplemental_runs、max_captures_per_target 应视为 proposal 未完成。
- 如果 diagnostics 是 repair_prompt.md，进入 repair-pass：只修 proposal 草案，不要修改 config/、output/、reports/。

执行要求：
- 严格读取并遵循 .agents/skills/onboard-model-proposal/SKILL.md。
- 需要 definition/workload 标准时，读取该 skill 下的 references/。
- 生成 proposal 后，运行：
  python3 -B -m tools.proposal_tools agent-loop --proposal-dir <RUN_DIR>/proposal --hf-config <HF_CONFIG_PATH> --flashinfer-root <FLASHINFER_SOURCE_ROOT>
- 如果 `agent_feedback.md` 显示 `FIX_REQUIRED`，先修正 proposal 文件，再重复运行同一个 `agent-loop` 命令。
- 只有 `ready for human review: True` 后才交付；warnings 可以保留，但必须在 review_checklist.md 里解释。
```

Reviewer 仍需人工检查 `proposal_check.json`、`agent_feedback.md` 和 `review_checklist.md`。agent-loop 自检不是 approval。

如果已经跑过一次 pipeline 并生成了 `reports/run_report.json`，先运行：

```bash
python3 -B -m tools.proposal_tools repair-loop \
  --run <RUN_DIR> \
  --hf-config <HF_CONFIG_PATH> \
  --flashinfer-root <FLASHINFER_SOURCE_ROOT>
```

然后把生成的 `<RUN_DIR>/proposal/repair_prompt.md` 作为 diagnostics 输入给同一个 skill。repair-pass 成功标准是：

```text
diagnostics ok: True
ready for human review: True
```
