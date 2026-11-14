# -*- coding: utf-8 -*-
"""
================================================================================
生成式人工智能系统应用员初赛实操题 —— 基于上下文工程的思维链（CoT）推理答题程序
================================================================================
运行方式：
    1) pip install openai
    2) 修改下面【一、配置区】里的 API_KEY（BASE_URL / MODEL 按你的服务商按需改）
    3) 把赛题 jsonl 放在本脚本同一目录，PyCharm 里直接点运行即可

输入：生成式人工智能系统应用员初赛实操题.jsonl
输出：生成式人工智能系统应用员初赛实操题-a1.jsonl
日志：run.log（运行过程留痕，便于人工复核）

解题流程（与赛题三个步骤一一对应）：
    第一步  调大模型判断题目类型 / 题型细分 / 是否可解；
            信息缺失时在控制台打印题目 id 与判断原因，不强行猜测作答
    第二步  按题目类型拼装专用提示词（专家人设 + 分模式推理规范 + JSON 输出约束）
    第三步  分模式输出：reasoning（先原理/公式，再详细步骤）+ answer（严格按 answer_format）

工程细节：
    - 断点续跑：输出文件里已有的 id 自动跳过（RESUME=True）
    - 并发处理：ThreadPoolExecutor，结果仍按原题顺序写回
    - 稳健解析：兼容 markdown 代码块、字段名漂移、非 JSON 输出
    - 失败隔离：单题失败不影响整批，结束时列出失败 id，重跑即自动补做
================================================================================
"""

import json
import logging
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:  # 依赖缺失时给出明确提示，而不是抛一堆栈
    print("缺少依赖，请先在命令行执行： pip install openai")
    sys.exit(1)


# ============================================================================
# 一、配置区（通常只需要改 API_KEY）
# ============================================================================
API_KEY = "sk-ws-H.YYMEYRI.GoK2.MEYCIQC-EXczLvaE935t9JRzvhE8nODMNwmVJgUesju7IvWSbAIhAJ9hvrD4zRaaMwN1-_0whJz84oPrtnkf_gao4gj-2a20"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"   # OpenAI 兼容模式接口
MODEL = "qwen3.8-27b"                                            # 赛题指定模型（大小写与网关模型名严格一致）

INPUT_FILE = "生成式人工智能系统应用员初赛实操题.jsonl"
OUTPUT_FILE = "生成式人工智能系统应用员初赛实操题-a2.jsonl"
LOG_FILE = "run.log"

MAX_WORKERS = 4            # 并发线程数：接口限流时报 429 就调小到 1~2
MAX_RETRIES = 3            # 单题最多尝试次数（含格式纠错重试）
REQUEST_TIMEOUT = 180      # 单次请求超时（秒）
MAX_TOKENS = 6000          # 单次输出上限，推理过程较长，别给太小
TEMPERATURE_CLASSIFY = 0.0  # 题型判断要稳定，用 0
TEMPERATURE_SOLVE = 0.3     # 推理答题略带多样性，0.2~0.5 之间较稳

ENABLE_THINKING: Optional[bool] = None   # None=不下发该参数；True/False=强制开启/关闭思考模式
RESUME = True              # True=断点续跑（想整批重跑：删掉输出文件，或改 False）
SOLVE_WHEN_UNSOLVABLE = False  # False=按赛题要求不硬猜；True=条件不足也尝试作答
UNSOLVABLE_ANSWER = "无法确定"  # 条件不足时 answer 字段的取值
MAX_QUESTIONS = 0          # 只跑前 N 题用于试水，0=全部
PREFLIGHT = True           # 启动时先发一个极小请求，验证 API_KEY / 模型名 / 网络


# ============================================================================
# 二、路径与日志
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / INPUT_FILE
OUTPUT_PATH = BASE_DIR / OUTPUT_FILE
LOG_PATH = BASE_DIR / LOG_FILE

try:  # Windows 控制台中文乱码兜底
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logger = logging.getLogger("cot")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
_fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
logger.addHandler(_fh)
logger.addHandler(_sh)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=0)
_write_lock = threading.Lock()


# ============================================================================
# 三、第一步：题目类型判断（上下文工程 · 分类器）
# ============================================================================
CLASSIFY_SYSTEM = (
    "你是赛题分析助手。你的任务只有一件：判断题目类型与可解性，绝对不要动手解题。\n"
    "只输出一个 JSON 对象，不要输出 markdown 代码块、不要输出任何解释文字。"
)

CLASSIFY_USER_TMPL = """请阅读下面这道题，完成三件事：
1) 判断题目类型，从这些类别里选最贴切的一个：数学、物理、化学、生物、地理、历史、政治、经济、医学、文学、常识、逻辑推理、其他。
2) 判断题型细分，从这些里选一个：计算题、应用题、概念题、推理题、判断说明题、其他。
3) 判断题目是否信息完整、条件充分、能够唯一求解；若不能，逐条列出缺少哪些条件。

题目：
\"\"\"
{question}
\"\"\"

答案格式要求：{answer_format}

严格按此 JSON 结构输出（字段名不可改）：
{{"type": "数学", "subtype": "应用题", "solvable": true, "missing": [], "reason": "一句话说明判断依据"}}"""


def _to_bool(value: Any, default: bool = True) -> bool:
    """模型可能把 true 写成字符串 'true'/'是'，统一转换。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "是", "可解", "完整", "可以"}
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def classify(qid: str, question: str, answer_format: str) -> Dict[str, Any]:
    """第 1 步：让大模型判断题型与可解性。"""
    if not question:
        return {
            "type": "其他", "subtype": "未知", "solvable": False,
            "missing": ["题干为空"], "reason": "question 字段为空，无法判断考察方向",
        }

    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": CLASSIFY_USER_TMPL.format(question=question, answer_format=answer_format or "无")},
    ]
    obj: Dict[str, Any] = {}
    try:
        text, _ = call_llm(messages, temperature=TEMPERATURE_CLASSIFY, max_tokens=800)
        obj = parse_model_json(text) or {}
    except Exception as exc:  # 分类失败不阻断主流程，降级为“其他”
        logger.warning("%s 题型判断调用失败：%s，降级按“其他”处理", qid, exc)

    qtype = str(obj.get("type") or "其他").strip()
    subtype = str(obj.get("subtype") or "其他").strip()
    missing = obj.get("missing") or []
    if isinstance(missing, str):
        missing = [missing]
    if not isinstance(missing, (list, tuple)):
        missing = []
    missing = [str(m).strip() for m in missing if str(m).strip()]

    return {
        "type": qtype or "其他",
        "subtype": subtype or "其他",
        "solvable": _to_bool(obj.get("solvable"), default=True),
        "missing": missing,
        "reason": str(obj.get("reason") or "").strip() or "模型未给出判断依据",
    }


# ============================================================================
# 四、第二步：按题型生成提示词（上下文工程 · 人设 + 分模式输出规范）
# ============================================================================
PERSONA = {
    "calc": "你是一名数学、物理、化学计算题专家，解题过程写得像教科书例题一样规范、可逐行复核。",
    "concept": "你是一名理科基础课教师，熟悉教材标准表述，讲概念时严格区分定义、性质与特例。",
    "logic": "你是一名逻辑推理与约束满足问题专家，擅长用条件编号 + 假设 + 排除法，把推理链完整摊开。",
    "human": "你是一名人文社科研究者，熟悉主流教材与学界共识，能清楚区分史实与观点。",
    "medicine": "你是一名临床医学教师，按教科书标准结论作答，措辞严谨、边界清楚。",
    "common": "你是一名严谨的常识问答专家，回答讲究依据来源，对无共识内容主动标注风险。",
    "general": "你是一名通用推理助手，擅长把复杂问题拆成清晰步骤，逐条推进直到结论。",
}

RULE_CALC = """【本题为计算 / 应用题，按“已知 → 求 → 公式 → 计算 → 作答”输出】
1. 先用一两句话点明本题用到的公式、定理或知识点（公式先行、符号规范）；
2. 明确列出已知量与待求量，统一单位，推导全过程中单位带全；
3. 代入与计算逐步展开，每一步写清依据，禁止跳步；
4. 数值按题目要求的精度保留，无要求时保留合理有效数字；
5. 结论单独成句，与 answer 字段完全一致。"""

RULE_LOGIC = """【本题为逻辑推理题，按“条件梳理 → 假设 → 排除 → 推导 → 汇总”输出】
1. 把题目给出的条件逐条编号列出（C1、C2……），位置类条件要写明“第几位是哪个元素”；
2. 明确推理目标，同时说明 answer_format 对答案形态（分隔符、排序、去重）的约束；
3. 先固定死的条件（如某元素位置已确定），再对剩余元素做假设与分类讨论；
4. 每一种被排除的情形，都要说明是被哪一条条件排除的；
5. 得到全部可行解后，逐一回代校验全部条件，再按题目要求排序、去重、拼接；
6. 禁止只给结论不给过程；多解时必须全部列出，不得遗漏。"""

RULE_CONCEPT = """【本题为理科概念题，按“定义 → 性质 → 特例/边界”输出】
1. 用课本标准表述给出定义，不要自造术语；
2. 分点写出主要性质，必要时给出适用条件；
3. 指出易混点、特例或常见误区；
4. 不确定的内容如实写明“教材未涉及/存在不同表述”，不要似是而非。"""

RULE_HUMAN = """【本题为人文社科类题目，按“背景 → 史实/事实 → 分析 → 结论”输出】
1. 依托主流教材与学界共识陈述事实，时间、人物、事件要准确；
2. 严格区分“史实/客观事实”与“评价/观点”，评价部分说明依据；
3. 存在学术争议的，注明“存在不同观点”并简述分歧；
4. 禁止编造史料、数据、条文与引文。"""

RULE_MEDICINE = """【本题为医学类题目，按“机制/定义 → 临床表现或判断依据 → 结论”输出】
1. 以教科书标准结论为准，术语规范；
2. 需要鉴别时写出鉴别要点；
3. 推理结束后，在 reasoning 末尾固定写上：【仅供学习参考，不能替代临床诊疗】。"""

RULE_COMMON = """【本题为常识类题目，按“依据 → 推理 → 结论”输出】
1. 先说明判断依据（生活常识、行业惯例、公开可查的通用规则）；
2. 冷门、无统一共识的内容，标注“该说法缺乏统一权威来源，存在来源风险”；
3. 不要用编造的“数据/研究”为结论背书。"""

RULE_GENERAL = """【通用推理规范】
1. 先复述题目要求与已知条件，确认没有遗漏约束；
2. 把问题拆成若干子步骤，逐步推进，每步写清依据；
3. 关键处做一次回代/自检，确认答案满足题目的全部条件；
4. 若条件确实不足，明确说明缺什么，不臆造前提。"""


def pick_persona_and_rule(info: Dict[str, Any]) -> Tuple[str, str]:
    """根据题型选择人设与对应的输出规范（这就是分模式输出的核心）。"""
    qtype = str(info.get("type", ""))
    subtype = str(info.get("subtype", ""))
    text = qtype + subtype

    if "医学" in text or "医" == qtype:
        return PERSONA["medicine"], RULE_MEDICINE
    if "逻辑" in text:
        return PERSONA["logic"], RULE_LOGIC
    if qtype in {"数学", "物理", "化学"} and ("计算" in subtype or "应用" in subtype or subtype in {"其他", ""}):
        return PERSONA["calc"], RULE_CALC
    if qtype in {"数学", "物理", "化学", "生物"}:
        return PERSONA["concept"], RULE_CONCEPT
    if qtype in {"历史", "政治", "经济", "地理", "文学"}:
        return PERSONA["human"], RULE_HUMAN
    if "常识" in text:
        return PERSONA["common"], RULE_COMMON
    return PERSONA["general"], RULE_GENERAL


SYSTEM_HEAD = """{persona}

你正在参加“生成式人工智能系统应用员”赛事的思维链推理环节。评分只看两点：
① reasoning 是否完整、可被人工逐步复核；② answer 是否严格符合给定格式。

{rule}

【学术诚信 · 硬性约束】
- 禁止编造知识点、公式、史料、数据、文献；不确定就如实写明“不确定 / 条件不足”，不要似是而非。
- 只使用题目给出的条件；确需额外假设时必须显式声明“此处假设……”。

【输出格式 · 必须严格遵守】
只输出一个 JSON 对象，不要 markdown 代码块，不要任何 JSON 之外的文字：
{{"reasoning": "完整的推理过程", "answer": "最终答案"}}

- reasoning：字符串。先用一两句话简洁说清原理 / 公式 / 知识点，再给出详细推导步骤；
  用 \\n 表示换行；不得为空，不得只写结论。
- answer：字符串。只放最终答案本身 —— 不要“答案：”前缀、不要解释、不要单位说明之外的文字，
  严格按下面的答案格式要求书写（分隔符、排序、去重都要照做）。
  答案格式要求：{answer_format}"""

USER_TMPL = """题目编号：{qid}

题目：
{question}

题型判断：{qtype} / {subtype}
答案格式要求：{answer_format}

请先在 reasoning 中完成完整推理，再给出严格符合格式的 answer，最后只输出那个 JSON 对象。"""


def build_system_prompt(info: Dict[str, Any], answer_format: str) -> str:
    persona, rule = pick_persona_and_rule(info)
    return SYSTEM_HEAD.format(persona=persona, rule=rule, answer_format=answer_format or "题目未给出，按最简形式作答")


def build_user_prompt(qid: str, question: str, info: Dict[str, Any], answer_format: str) -> str:
    return USER_TMPL.format(
        qid=qid,
        question=question,
        qtype=info.get("type", "其他"),
        subtype=info.get("subtype", "其他"),
        answer_format=answer_format or "题目未给出，按最简形式作答",
    )


# ============================================================================
# 五、模型调用与稳健解析
# ============================================================================
def call_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = MAX_TOKENS,
) -> Tuple[str, str]:
    """调用大模型，返回 (正文, 思考内容)。带退避重试 + 可选参数自动降级。"""
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        kwargs: Dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        if ENABLE_THINKING is not None:
            kwargs["extra_body"] = {"enable_thinking": ENABLE_THINKING}
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_err = exc
            # 某些兼容层不支持 response_format / extra_body，去掉后再试一次
            if "response_format" in kwargs or "extra_body" in kwargs:
                kwargs.pop("response_format", None)
                kwargs.pop("extra_body", None)
                try:
                    resp = client.chat.completions.create(**kwargs)
                except Exception as exc2:
                    last_err = exc2
                    wait = min(2 ** attempt + random.random(), 20)
                    logger.warning("模型调用失败（第 %d 次）：%s，%.1fs 后重试", attempt, exc2, wait)
                    time.sleep(wait)
                    continue
            else:
                wait = min(2 ** attempt + random.random(), 20)
                logger.warning("模型调用失败（第 %d 次）：%s，%.1fs 后重试", attempt, last_err, wait)
                time.sleep(wait)
                continue

        if not resp or not getattr(resp, "choices", None):
            last_err = RuntimeError("返回结果为空")
            time.sleep(1.5)
            continue

        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        think = (getattr(msg, "reasoning_content", None) or "").strip()
        if content or think:
            return content, think
        last_err = RuntimeError("模型返回内容为空")
        time.sleep(1.5)

    raise RuntimeError(f"模型调用失败，已重试 {MAX_RETRIES} 次：{last_err}")


def parse_model_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出里抽取 JSON 对象：兼容 ```json 包裹、前后废话、多个花括号片段。"""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*```(?:json|JSON)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    decoder = json.JSONDecoder()
    candidates: List[Dict[str, Any]] = []
    for match in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned[match.start():])
        except Exception:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)

    for obj in candidates:
        if "answer" in obj or "reasoning" in obj or "推理过程" in obj or "答案" in obj:
            return obj
    return candidates[0] if candidates else None


def fallback_extract(text: str) -> Tuple[str, str]:
    """没有 JSON 时的兜底：全文当推理，正则抓最后一条“答案：xxx”。"""
    reasoning = (text or "").strip()
    answer = ""
    hits = re.findall(r"(?:最终答案|答案|answer)\s*[:：]\s*(.+)", reasoning)
    if hits:
        answer = hits[-1].strip().split("\n")[0]
    return reasoning, answer


def normalize_answer(answer: Any, answer_format: str) -> str:
    """按 answer_format 做轻量规范化：去前缀、统一分隔符、必要时排序去重。"""
    if answer is None:
        return ""
    if isinstance(answer, (list, tuple)):
        answer = ";".join(str(a) for a in answer)
    s = str(answer).strip()
    if not s:
        return ""

    s = s.replace("；", ";").replace("：", ":")
    s = re.sub(r"^\s*(最终答案|答案|answer)\s*[:：]\s*", "", s, flags=re.IGNORECASE)
    s = s.strip().strip("\"'`“”‘’")
    s = s.rstrip("。．.，,；; ")

    # 形如 P-Q-R-S-T 的字母序列：去掉空格、统一大写
    if "字母" in answer_format or "大写" in answer_format:
        s = re.sub(r"\s+", "", s).upper()

    # answer_format 明确要求升序时，按分隔符拆分后排序去重
    if any(k in answer_format for k in ("升序", "从小到大", "递增", "字典序")):
        parts = [p.strip() for p in re.split(r"[;]", s) if p.strip()]
        uniq: List[str] = []
        for p in parts:
            if p not in uniq:
                uniq.append(p)
        if len(uniq) > 1:
            s = ";".join(sorted(uniq))
    return s


# ============================================================================
# 六、第三步：单题求解（分类 → 提示词 → 分模式输出 → 校验）
# ============================================================================
def solve_one(item: Dict[str, Any], idx: int, total: int) -> Dict[str, Any]:
    qid = str(item.get("id") or f"q{idx:03d}").strip()
    question = str(item.get("question") or "").strip()
    answer_format = str(item.get("answer_format") or "").strip()
    started = time.time()

    # ---------- 第一步：题型判断 ----------
    info = classify(qid, question, answer_format)
    qtype, subtype = info["type"], info["subtype"]

    # 类型不清 / 条件不足：按赛题要求打印 id 与原因，不强行作答
    type_unclear = qtype in {"其他", "未知", ""} and subtype in {"其他", "未知", ""}
    if not info["solvable"] or type_unclear:
        missing = info["missing"] or ["题干未说明缺失项"]
        reason = info["reason"]
        tip = "条件不足" if not info["solvable"] else "题型无法判定"
        print(f"\n【{tip}】题目 id = {qid}")
        print(f"          缺少条件：{'、'.join(missing)}")
        print(f"          判断原因：{reason}")
        logger.warning("%s 跳过强行作答（%s）：缺少=%s；原因=%s", qid, tip, missing, reason)

        if not SOLVE_WHEN_UNSOLVABLE:
            reasoning = (
                "【条件不足，不作猜测】\n"
                f"经题型分析，本题属于{qtype}类（{subtype}），但现有题干不足以唯一确定答案。\n"
                f"缺少的条件：{'、'.join(missing)}。\n"
                f"判断依据：{reason}\n"
                "在补充上述条件之前，任何具体取值都只是猜测，因此不作答。"
            )
            return {
                "id": qid, "question": question,
                "reasoning": reasoning, "answer": UNSOLVABLE_ANSWER,
            }

    # ---------- 第二步：按题型生成提示词 ----------
    system_prompt = build_system_prompt(info, answer_format)
    user_prompt = build_user_prompt(qid, question, info, answer_format)

    # ---------- 第三步：分模式输出 + 格式纠错重试 ----------
    reasoning, answer, raw_text = "", "", ""
    for attempt in range(1, MAX_RETRIES + 1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if attempt > 1:
            messages.append({
                "role": "user",
                "content": (
                    "上一次输出不合格（answer 为空或 reasoning 过短，或不是合法 JSON）。"
                    "请重新作答：reasoning 必须完整展示推理链，answer 必须严格符合格式，"
                    "并且只输出一个 JSON 对象。"
                ),
            })

        raw_text, think_text = call_llm(messages, temperature=TEMPERATURE_SOLVE)

        obj = parse_model_json(raw_text)
        if obj:
            reasoning = str(obj.get("reasoning") or obj.get("reasoning_process") or obj.get("推理过程") or "").strip()
            answer = obj.get("answer")
            if answer is None:
                answer = obj.get("final_answer") or obj.get("答案") or ""
        else:
            reasoning, answer = fallback_extract(raw_text)

        # 模型若单独返回了思考内容（如 Qwen 的 reasoning_content），并入 reasoning 开头
        if think_text:
            reasoning = f"{think_text}\n\n{reasoning}".strip()

        answer = normalize_answer(answer, answer_format)

        if answer and len(reasoning) >= 15:
            break
        logger.warning("%s 第 %d 次输出不完整（answer=%r，reasoning 长度=%d），准备重试",
                       qid, attempt, answer, len(reasoning))

    if not answer:
        raise RuntimeError(f"重试 {MAX_RETRIES} 次仍未拿到有效答案；最后原始输出片段：{raw_text[:200]!r}")

    cost = time.time() - started
    preview = answer if len(answer) <= 46 else answer[:46] + "..."
    logger.info("[%d/%d] %s  %s/%s  %.1fs  答案：%s", idx, total, qid, qtype, subtype, cost, preview)

    return {
        "id": qid,
        "question": question,
        "reasoning": reasoning,
        "answer": answer,
    }


# ============================================================================
# 七、文件读写（全部写在代码里，不依赖命令行参数）
# ============================================================================
def load_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到赛题文件：{path}\n请把赛题 jsonl 与本脚本放在同一目录。")
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:   # utf-8-sig 兼容带 BOM 的文件
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("第 %d 行不是合法 JSON，已跳过：%s", line_no, exc)
                continue
            if isinstance(obj, dict) and obj.get("question"):
                items.append(obj)
    if MAX_QUESTIONS > 0:
        items = items[:MAX_QUESTIONS]
    return items


def load_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    """读取已有输出，用于断点续跑。"""
    done: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("id") and str(obj.get("answer") or "").strip():
                done[str(obj["id"])] = obj
    return done


def write_output(items: List[Dict[str, Any]], results: Dict[str, Dict[str, Any]]) -> int:
    """按原题顺序原子写入（先写临时文件再替换，避免中途崩溃写出半截文件）。"""
    lines: List[str] = []
    for item in items:
        qid = str(item.get("id") or "").strip()
        record = results.get(qid)
        if record:
            lines.append(json.dumps(record, ensure_ascii=False))
    tmp = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    tmp.replace(OUTPUT_PATH)
    return len(lines)


# ============================================================================
# 八、主流程
# ============================================================================
def preflight() -> bool:
    """开跑前先花一个 token 验证连通性，避免整批题都因配置写错而白白失败。"""
    if not PREFLIGHT:
        return True
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
            temperature=0,
        )
        _ = resp.choices[0].message.content
        print(f"[自检] 模型 {MODEL} 调用正常，开始批量作答。\n")
        return True
    except Exception as exc:
        msg = str(exc)
        low = msg.lower()
        print(f"\n[自检失败] 无法调用模型 {MODEL}：{msg[:300]}\n")
        if "model" in low and any(k in low for k in ("not", "exist", "invalid", "unsupported", "unknown")):
            print("  → 多半是模型名不对：请对照网关文档里的模型标识（大小写、连字符）改脚本顶部 MODEL。")
        if "401" in msg or "invalid_api_key" in low or "auth" in low or "api key" in low:
            print("  → 多半是 API_KEY 无效：检查是否已替换、有无多余空格/引号。")
        if "404" in msg or "connection" in low or "url" in low or "timeout" in low:
            print("  → 请检查 BASE_URL 是否为可访问的 OpenAI 兼容接口（通常以 /v1 结尾），以及网络/代理。")
        print("  配置无误后可把 PREFLIGHT 设为 False 跳过自检。")
        return False


def main() -> None:
    print("=" * 78)
    print("生成式人工智能系统应用员初赛实操题 —— 思维链推理答题程序")
    print(f"模型：{MODEL}    接口：{BASE_URL}")
    print(f"输入：{INPUT_PATH.name}")
    print(f"输出：{OUTPUT_PATH.name}")
    print("=" * 78)

    if "请在这里替换" in API_KEY:
        print("\n[提示] 还没填 API_KEY，请先修改脚本顶部的 API_KEY 再运行。\n")
        return

    if not preflight():
        return

    try:
        items = load_questions(INPUT_PATH)
    except FileNotFoundError as exc:
        print(f"\n[错误] {exc}")
        return

    total = len(items)
    if total == 0:
        print("\n[错误] 赛题文件里没有读到有效题目。")
        return
    logger.info("共读取 %d 道题目", total)

    results: Dict[str, Dict[str, Any]] = load_existing(OUTPUT_PATH) if RESUME else {}
    if results:
        logger.info("断点续跑：已有 %d 道题完成，将跳过", len(results))

    todo = [it for it in items if str(it.get("id") or "").strip() not in results]
    logger.info("本次需要作答 %d 道题，并发数 %d", len(todo), MAX_WORKERS)

    failures: List[Tuple[str, str]] = []
    started = time.time()

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, MAX_WORKERS)) as pool:
            future_map = {
                pool.submit(solve_one, item, i + 1, len(todo)): item
                for i, item in enumerate(todo)
            }
            for future in as_completed(future_map):
                item = future_map[future]
                qid = str(item.get("id") or "")
                try:
                    record = future.result()
                except Exception as exc:
                    failures.append((qid, str(exc)))
                    logger.error("%s 处理失败：%s", qid, exc)
                    continue
                with _write_lock:
                    results[qid] = record
                    write_output(items, results)   # 每完成一题就落盘，随时可中断

    written = write_output(items, results)
    cost = time.time() - started

    print("\n" + "-" * 78)
    print(f"完成：{written}/{total} 道题已写入 {OUTPUT_PATH.name}，耗时 {cost:.1f}s")
    print(f"输出文件：{OUTPUT_PATH}")
    print(f"运行日志：{LOG_PATH}")
    if failures:
        print(f"\n[警告] 有 {len(failures)} 道题失败，未写入输出文件：")
        for qid, err in failures:
            print(f"  - {qid}：{err[:120]}")
        print("  直接再次运行本脚本即可自动补做这些题目（RESUME=True 会跳过已完成项）。")
    else:
        print("全部题目处理成功。")
    print("-" * 78)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已手动中断，已完成的题目已保存，重跑可继续。")
