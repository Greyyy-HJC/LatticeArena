"""Write REPORT.md: a code-generated top-3 table plus LLM commentary."""
import json
from pathlib import Path

from agents.llm import chat

PROMPT = """You are interpreting a lattice-QCD benchmark result. Numbers below were computed by code; do not invent numbers.

Reference status: {status}
Failure reasons: {reasons}

Top candidates (risk = uncertainty term + departure term, lower is better; ratio is risk_local/risk_candidate):
{table}

Write 3-5 short bullets in Chinese explaining: (1) where the candidates differ, (2) the likely physics (excited-state contamination, smearing, signal-to-noise), (3) what to try next. Do not claim SOTA or a discovery. Return JSON only: {{"markdown": "..."}}"""


def run(output, model):
    output = Path(output)
    report = json.loads((output / 'validation_results.json').read_text())
    rows = [(name, value) for name, value in report['results'].items() if 'risk' in value]
    rows.sort(key=lambda row: row[1]['risk'])
    table = '\n'.join(
        f"| {i + 1} | {name} | {value['risk']:.6g} | {value['variance']:.4g} | "
        f"{value['departure_squared']:.4g} | {value['diagnostic_loss_ratio_vs_local']:.3g} | "
        f"{value.get('risk_95_interval')} |"
        for i, (name, value) in enumerate(rows[:3]))
    text = json.loads(chat(PROMPT.format(status=report['reference_passed'],
                                         reasons=report['reference_failure_reasons'],
                                         table=table), model))['markdown']

    extra = output / 'agent' / 'best.json'
    post = ''
    if extra.exists():
        best = json.loads(extra.read_text())
        llm = best.get('llm_best') or {}
        post = (f"\n## 附：LLM 作答（post-hoc，非官方提交）\n\n"
                f"- 模型：`{best['model']}`，{best['rounds']} 轮 × 3 个提案\n"
                f"- LLM 最好提案（round {llm.get('round')}）：开发 risk {llm.get('risk', float('nan')):.6g}，"
                f"归一化系数 u {llm.get('u')}\n"
                f"- 同一条提案的留出诊断 risk：{best.get('validation_risk', 'invalid')}"
                f"（相对 local 的 loss 比 {best.get('validation_ratio_vs_local', float('nan')):.3g}）\n"
                f"- 参考地板（含基线 `H^32`）：{best['risk']:.6g}\n"
                f"- **说明**：这是事后诊断，没有走冻结/提交流程，不是官方分数。\n")

    (output / 'REPORT.md').write_text(
        f"# 结果解读\n\n"
        f"> 数字由代码从 `validation_results.json` 生成；下面的「LLM 解读」是模型叙述，不是证据。\n\n"
        f"## 参考状态\n\n- reference_passed: `{report['reference_passed']}`\n"
        f"- 失败原因：{report['reference_failure_reasons']}\n"
        f"- 分数状态：`{report['score_status']}`（参考未通过时所有 score 为 null）\n\n"
        f"## 前三名（按 risk 排序）\n\n"
        f"| 排名 | 候选 | risk | 方差项 | 偏离项 | 相对 local | 95% 区间 |\n"
        f"| --- | --- | ---: | ---: | ---: | ---: | --- |\n{table}\n\n"
        f"## LLM 解读（非权威）\n\n{text}\n{post}")
    print(f'wrote {output / "REPORT.md"}')
