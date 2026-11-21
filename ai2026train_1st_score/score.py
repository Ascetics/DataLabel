# -*- coding: utf-8 -*-
"""
================================================================================
人工智能训练师竞赛 · 实操题
任务：高质量数据集质量评测（文本数据四维打分 + 质量判定）
================================================================================

赛事要求与实现对应关系
----------------------
1) 三个大模型全部参与，每题都调用：
     阿里云百炼        Qwen3.8-27B
     硅基流动          Qwen3-32B      （同时担任仲裁者）
     讯飞星火          Spark-X2-Flash
2) 四个维度分数：三个模型各自给出分数后，**每个维度分别取平均值**。
3) quality 判定：三个模型各自给出 high/low 结论，**少数服从多数**（3:0 或 2:1）；
   若三家答案全都不一致（无法形成多数），**直接以硅基流动 Qwen3-32B 的答案为准**。
4) 综合得分 score_avg = 四个维度最终得分的平均值。
5) 阈值由选手自主设置（见 THRESHOLDS），用于要求模型给出判定结论。

运行方式
--------
PyCharm 中直接点绿色三角运行，不需要命令行参数、不需要 pip 安装第三方包
（仅使用 Python 标准库 urllib / json / threading / concurrent.futures）。

首次运行只改两处：CONFIG 区的 3 个 API_KEY、INPUT_FILE / OUTPUT_FILE 文件名。
================================================================================
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ==============================================================================
# ============================ CONFIG 配置区（按需修改）=========================
# ==============================================================================

# ---------- 1. 文件路径（默认脚本同级目录，改成你自己的文件名/路径即可）----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "人工智能训练师初赛实操题.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "人工智能训练师初赛实操题-a1.jsonl")

# ---------- 2. 并发与超时 ----------
MAX_WORKERS = 4            # 【并发数】默认同同时处理 4 道题，可自行调大/调小
MAX_INFLIGHT_CALLS = 12    # 全局同时在飞的大模型请求上限（4 题 × 3 模型 = 12）
REQUEST_TIMEOUT = 120      # 单次 HTTP 请求超时（秒）
MAX_RETRIES = 3            # 单次模型调用失败后的最大尝试次数
RETRY_BACKOFF = 2.0        # 重试退避基数（秒），第 n 次失败等待 RETRY_BACKOFF * n
PREFLIGHT_CHECK = True     # 开跑前做一次极小的连通性自检，提前暴露 Key 问题

# ---------- 3. 质量判定阈值（赛题允许选手自主设置，四项均"高于"阈值且综合高于阈值才为 high）----------
THRESHOLDS: Dict[str, float] = {
    "completeness": 0.65,   # 文本完整程度阈值
    "density": 0.55,        # 知识信息密度阈值
    "coherence": 0.65,      # 文本连贯程度阈值
    "purity": 0.60,         # 文本纯净程度阈值
    "avg": 0.65,            # 综合得分阈值
}

# ---------- 4. 判定口径开关 ----------
# False（默认）：严格按赛题"少数服从多数"取三家投票结果；
# True         ：强制用"平均值 + 阈值"重新推导 quality，保证分数与结论绝对自洽。
FORCE_THRESHOLD_CONSISTENCY = False

# ---------- 5. 提示词长度控制（长文本只取头尾，兼顾"开头"与"截断常发生在结尾"）----------
PROMPT_TEXT_HEAD = 4000    # 保留原文开头字符数
PROMPT_TEXT_TAIL = 2000    # 保留原文结尾字符数

# ---------- 6. 断点保护：每完成 N 题就落盘一次，防止中途崩溃白跑 ----------
SAVE_EVERY = 5


@dataclass
class ModelConfig:
    """一个大模型的接入配置（三家均为 OpenAI 兼容的 chat/completions 协议）。"""
    short: str                      # 控制台短名：百炼 / 硅基 / 讯飞
    name: str                       # 完整名称，用于失败提示
    api_key: str
    base_url: str
    model: str
    is_arbiter: bool = False        # 是否为仲裁者（硅基流动 Qwen3-32B）
    temperature: float = 0.2
    max_tokens: int = 1200
    extra_body: Dict[str, Any] = field(default_factory=dict)
    timeout: int = REQUEST_TIMEOUT


# ==============================================================================
# ============================ 三个大模型的接入配置 =============================
# ==============================================================================
# 【替换 API_KEY】把下面三处 "sk-请替换..." 换成你自己的 Key 即可。
# ==============================================================================

MODEL_BAILIAN = ModelConfig(
    short="百炼",
    name="阿里云百炼-Qwen3.8-27B",
    api_key="sk-ws-H.YYMEYRI.GoK2.MEYCIQC-EXczLvaE935t9JRzvhE8nODMNwmVJgUesju7IvWSbAIhAJ9hvrD4zRaaMwN1-_0whJz84oPrtnkf_gao4gj-2a20",
    # 百炼 OpenAI 兼容模式（北京地域）；国际站改为
    # https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.8-27b",
)

MODEL_SILICONFLOW = ModelConfig(
    short="硅基",
    name="硅基流动-Qwen3-32B（仲裁者）",
    api_key="sk-sxonjhtlsbketbarbwjifpwalapsxjjtbfskqeyrryovphdz",
    base_url="https://api.siliconflow.cn/v1",
    model="Qwen/Qwen3-32B",
    is_arbiter=True,                     # 无法形成多数时，以它的答案为准
    # Qwen3 系列默认开启思考模式，耗时高、容易把 JSON 挤到截断。
    # 这里关闭思考链；若平台报 400 参数错误，脚本会自动剥离该参数重试。
    extra_body={"enable_thinking": False},
)

MODEL_XFYUN = ModelConfig(
    short="讯飞",
    name="讯飞星火-Spark-X2-Flash",
    api_key="ak-ade8de781bebccd1d1ba428592c34ea8",
    # 讯飞星辰 MaaS 平台 OpenAI 兼容接口（官方文档：model=xsparkx2flash）
    # OpenAI_Chat: https://maas-api.cn-huabei-1.xf-yun.com/v2/chat/completions
    base_url="https://maas-api.cn-huabei-1.xf-yun.com/v2",
    model="xsparkx2flash",
)

MODELS: List[ModelConfig] = [MODEL_BAILIAN, MODEL_SILICONFLOW, MODEL_XFYUN]
ARBITER: ModelConfig = next(m for m in MODELS if m.is_arbiter)

# 四个维度：内部键 -> 输出字段名 -> 中文名
DIMENSIONS: List[Tuple[str, str, str]] = [
    ("completeness", "score_completeness", "完整程度"),
    ("density", "score_density", "知识信息密度"),
    ("coherence", "score_coherence", "连贯程度"),
    ("purity", "score_purity", "纯净程度"),
]
DIM_KEYS = [d[0] for d in DIMENSIONS]
DIM_OUT_KEY = {d[0]: d[1] for d in DIMENSIONS}
DIM_LABEL = {d[0]: d[2] for d in DIMENSIONS}

# ==============================================================================
# ============================== 控制台输出工具 ================================
# ==============================================================================

PRINT_LOCK = threading.RLock()


def _safe_write(text: str) -> None:
    """线程安全 + 编码安全的控制台输出（Windows 控制台偶发 GBK 编码报错）。"""
    with PRINT_LOCK:
        try:
            print(text, flush=True)
        except UnicodeEncodeError:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            print(text.encode(enc, "replace").decode(enc, "replace"), flush=True)


def log(text: str = "") -> None:
    _safe_write(text)


def log_block(text: str) -> None:
    """整块输出，避免多线程下不同题目的日志互相穿插。"""
    with PRINT_LOCK:
        _safe_write(text)


def hr(ch: str = "─", n: int = 78) -> str:
    return ch * n


def fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:.0f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h{m}m{s:.0f}s"


# ==============================================================================
# ============================ 大模型调用层（HTTP）=============================
# ==============================================================================

INFLIGHT_SEMAPHORE = threading.Semaphore(MAX_INFLIGHT_CALLS)


class ModelCallError(Exception):
    """带"模型名 + 原因 + 是否可重试"的调用异常。"""

    def __init__(self, model_name: str, reason: str, retryable: bool = True, status: Optional[int] = None):
        super().__init__(reason)
        self.model_name = model_name
        self.reason = reason
        self.retryable = retryable
        self.status = status


def _extract_error_message(body: str) -> str:
    """从各家返回的错误体里尽量抠出人类可读的原因。"""
    if not body:
        return "（响应体为空）"
    try:
        obj = json.loads(body)
        if isinstance(obj, dict):
            err = obj.get("error")
            if isinstance(err, dict):
                for k in ("message", "msg", "detail", "code"):
                    if err.get(k):
                        return f"{err.get('code', '')} {err.get(k)}".strip()
            if isinstance(err, str) and err:
                return err
            for k in ("message", "msg", "error_msg", "detail"):
                if obj.get(k):
                    return str(obj[k])
    except Exception:
        pass
    return body.strip()[:300]


def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int) -> Tuple[int, str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return getattr(resp, "status", 200), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            pass
        return e.code, body
    except urllib.error.URLError as e:
        raise ModelCallError("", f"网络连接失败（URLError）：{e.reason}", retryable=True)
    except TimeoutError:
        raise ModelCallError("", f"请求超时（>{timeout}s）", retryable=True)
    except OSError as e:
        raise ModelCallError("", f"底层网络错误：{e}", retryable=True)


def _chat_once(cfg: ModelConfig, messages: List[Dict[str, str]], temperature: float) -> str:
    """发起一次 chat/completions 请求，返回文本内容。失败抛 ModelCallError。"""
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    base_payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": round(temperature, 3),
        "max_tokens": cfg.max_tokens,
        "stream": False,
    }
    payload: Dict[str, Any] = dict(base_payload)
    if cfg.extra_body:
        payload.update(cfg.extra_body)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
    }

    with INFLIGHT_SEMAPHORE:
        status, body = _post_json(url, headers, payload, cfg.timeout)
        # 兼容性兜底：个别平台不认识 enable_thinking 这类扩展参数会直接 400，
        # 此时去掉扩展参数重试一次，避免"仲裁者"整体掉线。
        if status == 400 and cfg.extra_body:
            log(f"    [兼容性重试] 模型={cfg.name} 拒绝扩展参数 {list(cfg.extra_body)}，"
                f"已改为不发送该参数重试")
            status, body = _post_json(url, headers, base_payload, cfg.timeout)

    if status != 200:
        reason = f"HTTP {status}：{_extract_error_message(body)}"
        retryable = status in (408, 409, 425, 429) or status >= 500
        raise ModelCallError(cfg.name, reason, retryable=retryable, status=status)

    try:
        obj = json.loads(body)
    except Exception:
        raise ModelCallError(cfg.name, f"响应不是合法 JSON：{body[:200]}", retryable=True)

    choices = obj.get("choices") or []
    if not choices:
        err = _extract_error_message(body)  # 有些平台把错误塞在 200 响应里
        raise ModelCallError(cfg.name, f"响应中没有 choices 字段：{err}", retryable=True)

    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):  # 少数实现返回分段结构
        content = "".join(seg.get("text", "") for seg in content if isinstance(seg, dict))
    if not content:
        content = msg.get("reasoning_content") or ""
    if not str(content).strip():
        raise ModelCallError(cfg.name, "模型返回内容为空", retryable=True)
    return str(content)


# ------------------------------ JSON 稳健解析 ------------------------------

def _balanced_objects(s: str):
    """从任意文本中扫描出所有"括号配平"的 JSON 对象片段。"""
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield s[start:i + 1]
                        break
        start = s.find("{", start + 1)


def extract_json(text: str) -> Optional[dict]:
    """模型输出 -> dict。兼容 markdown 代码块、前后废话、数组包对象等情况。"""
    if not text or not text.strip():
        return None
    s = text.strip()
    fence = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)```", s)
    if fence:
        s = fence.group(1).strip()
    near = ""
    if "{" in s and "}" in s:
        near = s[s.find("{"):s.rfind("}") + 1]
    for cand in (s, near):
        if not cand:
            continue
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v[0]
        except Exception:
            pass
    for cand in _balanced_objects(s):
        try:
            v = json.loads(cand)
            if isinstance(v, dict):
                return v
        except Exception:
            continue
    return None


def _num(x: Any) -> Optional[float]:
    """把各种写法的分数转成 float（兼容 "0.7"、"70%"、0.7）。"""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        t = x.strip().replace("％", "%")
        try:
            if t.endswith("%"):
                return float(t[:-1]) / 100.0
            return float(t)
        except ValueError:
            return None
    return None


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


_HIGH_WORDS = {"high", "h", "高", "高质量", "1", "true", "yes", "y", "是"}
_LOW_WORDS = {"low", "l", "低", "低质量", "0", "false", "no", "n", "否"}


def normalize_quality(x: Any) -> Optional[str]:
    """把模型给出的质量结论规范成 "high" / "low"。"""
    if isinstance(x, bool):
        return "high" if x else "low"
    if isinstance(x, (int, float)) and x in (0, 1):
        return "high" if int(x) == 1 else "low"
    if isinstance(x, str):
        t = x.strip().lower().replace(" ", "")
        if t in _HIGH_WORDS:
            return "high"
        if t in _LOW_WORDS:
            return "low"
        # 兼容 "high（高质量）" 这类带修饰的写法
        if "high" in t or "高" in t:
            return "high"
        if "low" in t or "低" in t:
            return "low"
    return None


# 模型对同一字段可能有不同写法（输出字段名 / 中文名 / 简写），统一做别名兼容，
# 避免因键名不一致导致"取不到分"而退化到规则兜底。
KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "completeness": ("score_completeness", "completeness", "完整程度", "完整性", "完整度"),
    "density": ("score_density", "density", "知识信息密度", "知识密度", "信息密度"),
    "coherence": ("score_coherence", "coherence", "连贯程度", "连贯性", "连贯度"),
    "purity": ("score_purity", "purity", "纯净程度", "纯净度", "纯净性"),
    "avg": ("score_avg", "avg", "average", "综合得分", "综合分", "平均分"),
    "quality": ("quality", "label", "is_high", "result", "质量", "质量判定", "判定", "结论"),
    "reasoning": ("reasoning", "analysis", "分析", "推理"),
}


def lookup_value(data: Any, key: str) -> Tuple[bool, Any]:
    """在模型返回的 dict 里按别名查找某个键。"""
    if not isinstance(data, dict):
        return False, None
    for alias in KEY_ALIASES.get(key, (key,)):
        if alias in data:
            return True, data[alias]
    return False, None


# ==============================================================================
# ============================ 上下文工程：评测规范 ============================
# ==============================================================================

# 规范前置注入：三家模型看到完全一致的标准、阈值、示例，减少评分口径漂移
RUBRIC = """【评测维度定义】（参照全国数标委《高质量数据集 质量评测规范》设计思路）
1. 文本完整程度（completeness）
   文本内容表达完整、不存在缺失或中途截断的程度。
   关注：是否半句话收尾、是否句中突然停止、是否缺结论/后半段、是否只有标题没有正文。
2. 知识信息密度（density）
   文本中包含真实且有效信息的丰富程度。
   关注：信息点是否密集、是否有实质概念/事实/方法；空话套话、重复表述、广告与导航文字越多，密度越低。
3. 文本连贯程度（coherence）
   主题与内容保持一致，且行文流畅、无异常截断和错别字的程度。
   关注：是否前后主题漂移、逻辑是否衔接、语序是否混乱、有无明显错别字与病句。
4. 文本纯净程度（purity）
   正文内容清晰整洁，不受广告、页码、特殊符号和乱码等干扰的程度。
   关注：广告位招租、推广链接、版权声明、页码页眉、OCR 噪声、乱码（□□、※※※）、多余符号等。

【评分标尺】（建议分数落在 0.05 的整数倍，取值范围 0~1）
1.0      完全符合该维度要求，无任何问题
0.85-0.95  基本符合，仅有极轻微瑕疵
0.65-0.80  主体可用，但存在明显但可容忍的问题
0.45-0.60  问题较严重，需要较多人工修整才能使用
0.20-0.40  问题很严重，大部分内容不可用
0.0-0.15   该维度完全不达标（如整篇乱码）

【纪律要求】
- 一切判断必须以原文可见证据为依据，不得揣测原文没写的内容；不要因为文本短就一律打低分，
  短但表达完整的文本，完整程度也可以是高分。
- 四个维度相互独立评分，不要互相牵制（例如有广告只扣纯净程度，不要连带把完整程度也打低）。
- 严格只输出约定的 JSON 对象，不要输出解释性文字，不要使用 markdown 代码块。"""


def build_system_prompt() -> str:
    """把角色、规范、阈值、判定规则、两个示例一次性注入 system，形成稳定的"上下文"。"""
    th = THRESHOLDS
    return f"""你是一名资深的大模型训练数据工程师，负责对训练语料做系统化质量评测。
你的评分会直接决定该条数据能否进入训练集，请严谨、克制、以证据说话。

{RUBRIC}

【本次比赛的判定阈值（由选手设定，必须严格遵守）】
- 完整程度阈值：{th['completeness']}
- 知识信息密度阈值：{th['density']}
- 连贯程度阈值：{th['coherence']}
- 纯净程度阈值：{th['purity']}
- 综合得分阈值：{th['avg']}
判定规则：四项维度得分**均严格高于**各自阈值，且综合得分（四项平均值）**严格高于**综合阈值时，
quality 才为 "high"；否则为 "low"。

【工作流程（必须按顺序思考，再把结论写进 reasoning）】
第1步：通读全文，判断主题、体裁与用途。
第2步：逐维度找证据（完整性看结尾是否残缺；密度数有效信息点；连贯看主题漂移与病句；纯净看广告/页码/乱码占比）。
第3步：对照评分标尺，给四个维度分别打分。
第4步：自己算一遍四项平均分，对照上面阈值，得出 quality。
第5步：在 reasoning 中用 3~6 句话说明判断依据，然后给出四项分数与 quality。

【示例一（高质量）】
文本：光合作用是绿色植物利用光能，将二氧化碳和水转化为有机物并释放氧气的过程。该过程主要在叶绿体中进行，分为光反应与暗反应两个阶段：光反应在类囊体薄膜上完成水的光解并产生 ATP 和 NADPH；暗反应在叶绿体基质中通过卡尔文循环固定二氧化碳。
输出：{{"reasoning": "首尾完整无截断；包含定义、场所、两阶段机制等密集知识点；主题统一、行文流畅、无错别字；无广告页码乱码，纯净。四项平均约0.94，高于全部阈值，判定为高质量。", "score_completeness": 0.95, "score_density": 0.90, "score_coherence": 0.95, "score_purity": 1.0, "quality": "high"}}

【示例二（低质量）】
文本：深度学习是机器学习的一个分支，通过多层神经网络对数据进行特征提取和模式识别……广告位招租※※※在图像识别、自然语言处理等领域取得了
输出：{{"reasoning": "结尾停在“取得了”处，句子未完成，属明显截断；正文中插入“广告位招租※※※”，与主题无关，纯净度差；有效信息只有一句半，密度偏低；主题基本一致但被广告打断，连贯性一般。四项平均约0.36，低于阈值，判定为低质量。", "score_completeness": 0.30, "score_density": 0.45, "score_coherence": 0.50, "score_purity": 0.20, "quality": "low"}}

【输出格式】只输出一个 JSON 对象，字段与顺序如下：
{{"reasoning": "3~6句话的分析", "score_completeness": 0.00, "score_density": 0.00, "score_coherence": 0.00, "score_purity": 0.00, "quality": "high 或 low"}}"""


SYSTEM_PROMPT = build_system_prompt()


def clip_text(text: str) -> Tuple[str, bool, int]:
    """长文本截取头尾；返回 (用于提示词的文本, 是否被截断, 原文长度)。"""
    raw_len = len(text)
    if raw_len <= PROMPT_TEXT_HEAD + PROMPT_TEXT_TAIL:
        return text, False, raw_len
    omitted = raw_len - PROMPT_TEXT_HEAD - PROMPT_TEXT_TAIL
    clipped = (
        text[:PROMPT_TEXT_HEAD]
        + f"\n……（此处省略中间 {omitted} 字，省略部分不影响对整体质量的判断）……\n"
        + text[-PROMPT_TEXT_TAIL:]
    )
    return clipped, True, raw_len


def build_user_prompt(text: str, qid: str) -> str:
    clipped, was_clipped, raw_len = clip_text(text)
    return "\n".join([
        f"【题目编号】{qid}",
        f"【待评测文本】原文共 {raw_len} 字"
        + ("（因长度限制，提示词中只保留开头与结尾）" if was_clipped else ""),
        "<<<TEXT_BEGIN>>>",
        clipped,
        "<<<TEXT_END>>>",
        "",
        "请按系统提示中的五个步骤逐维度分析并打分，只输出一个 JSON 对象。",
    ])


# ==============================================================================
# ============================= 单模型调用（含重试）==========================
# ==============================================================================

def _fail_notice(cfg: ModelConfig, qid: str, step: str, attempt: int, reason: str) -> None:
    """【要求6】大模型调用失败时，明确打印是哪个模型失败、失败原因。"""
    log(f"    [模型调用失败] 模型={cfg.name} | 题目id={qid} | 步骤={step} | "
        f"第{attempt}/{MAX_RETRIES}次尝试 | 原因={reason}")


def call_model_json(
    cfg: ModelConfig, system_prompt: str, user_prompt: str, qid: str, step: str
) -> Tuple[bool, Optional[dict], str]:
    """带重试的调用 + JSON 解析。返回 (成功?, 解析结果, 错误原因)。"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_reason = ""
    for attempt in range(1, MAX_RETRIES + 1):
        temp = max(0.0, min(1.0, cfg.temperature + (attempt - 1) * 0.2))
        try:
            content = _chat_once(cfg, messages, temp)
        except ModelCallError as e:
            last_reason = e.reason or str(e)
            _fail_notice(cfg, qid, step, attempt, last_reason)
            if not e.retryable:
                return False, None, last_reason
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
            continue

        data = extract_json(content)
        if data is not None:
            return True, data, ""

        last_reason = f"输出无法解析为 JSON（原文前120字：{content[:120]!r}）"
        _fail_notice(cfg, qid, step, attempt, last_reason)
        # 解析失败也重试，并追加一条纠正指令
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": content[:2000]},
            {"role": "user", "content": "上面的输出不是合法 JSON。请严格只输出一个 JSON 对象，"
                                        "不要任何解释文字，不要 markdown 代码块。"},
        ]
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF)
    return False, None, last_reason


def ask_all_models(qid: str, text: str) -> Dict[str, Tuple[bool, Optional[dict], str]]:
    """一道题同时分发给 3 个模型（并行），返回 {模型名: (成功?, 数据, 错误原因)}。"""
    user_prompt = build_user_prompt(text, qid)
    results: Dict[str, Tuple[bool, Optional[dict], str]] = {}
    lock = threading.Lock()

    def worker(cfg: ModelConfig) -> None:
        try:
            out = call_model_json(cfg, SYSTEM_PROMPT, user_prompt, qid, "四维打分")
        except Exception as e:  # noqa: BLE001 任何意外都不能让该票静默消失
            out = (False, None, f"{type(e).__name__}: {e}")
            _fail_notice(cfg, qid, "四维打分", 1, out[2])
        with lock:
            results[cfg.name] = out

    threads = [threading.Thread(target=worker, args=(m,), daemon=True) for m in MODELS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# ==============================================================================
# ====================== 规则兜底（全模型失败时保证有输出）====================
# ==============================================================================

AD_REGEXES = [
    r"广告", r"招租", r"推广", r"点击(进入|查看|链接)", r"关注(公众号|我们)", r"扫码", r"二维码",
    r"微信(号|搜索)", r"QQ\s*群", r"https?://", r"www\.", r"版权所有", r"转载(请|自)", r"免责声明",
    r"Copyright", r"All\s+Rights\s+Reserved", r"第\s*\d+\s*页", r"共\s*\d+\s*页", r"编辑[:：]", r"责任编辑",
    r"下载(APP|客户端)", r"立即(购买|抢购)", r"优惠(券)?", r"客服(电话|微信)",
]
AD_RE = re.compile("|".join(AD_REGEXES), re.IGNORECASE)
GARBLE_RE = re.compile(
    r"[\ufffd\u25a1\u25a0\u203b\u25c7\u25c6\u2605\u2606]"
    r"|(?:[^\w\s\u4e00-\u9fff，。！？；：、“”‘’（）《》…—\-\.,!?;:'\"()]{4,})"
)
REPEAT_RE = re.compile(r"(.)\1{3,}")


def heuristic_scores(text: str) -> Tuple[Dict[str, float], List[str]]:
    """纯规则的保守兜底打分：只在模型全部不可用时使用，保证输出文件完整。"""
    t = (text or "").strip()
    notes: List[str] = []
    s = {"completeness": 0.85, "density": 0.75, "coherence": 0.80, "purity": 0.85}

    terminal = "。！？!?…”’）)》】.~～"
    if (not t) or (t[-1] not in terminal):
        s["completeness"] -= 0.35
        notes.append("结尾非终止标点→疑似截断")
    if AD_RE.search(t):
        s["purity"] -= 0.35
        s["density"] -= 0.10
        notes.append("命中广告/页码特征")
    if GARBLE_RE.search(t):
        s["purity"] -= 0.30
        s["coherence"] -= 0.20
        notes.append("命中乱码/异常符号特征")
    if REPEAT_RE.search(t):
        s["coherence"] -= 0.20
        s["completeness"] -= 0.10
        notes.append("存在同字符连续重复")
    if len(t) < 30 or (len(set(t)) / max(1, len(t)) < 0.08):
        s["density"] -= 0.25
        s["coherence"] -= 0.10
        notes.append("文本过短或字符重复率过高")
    if len(t) < 10:
        s["completeness"] -= 0.20
        s["density"] -= 0.20
        notes.append("长度过短")

    return {k: round(clamp01(v), 2) for k, v in s.items()}, notes


def decide_quality_by_threshold(avg: float, scores: Dict[str, float]) -> str:
    """阈值规则：四项均严格高于各自阈值，且综合得分严格高于综合阈值 -> high。"""
    ok = all(scores[k] > THRESHOLDS[k] for k in DIM_KEYS) and avg > THRESHOLDS["avg"]
    return "high" if ok else "low"


# ==============================================================================
# ============================== 单题完整流程 =================================
# ==============================================================================

def vote_quality(
    votes: Dict[str, Optional[str]],
) -> Tuple[str, str, str]:
    """
    质量结论投票：少数服从多数。
      3:0 / 2:1      -> 多数结论
      无多数(如 1:1) -> 以硅基流动 Qwen3-32B（仲裁者）的答案为准
      仲裁者也无答案 -> 标记为 tie，由调用方按阈值规则兜底
    返回 (结论, 票型描述, 决策方式)
    """
    valid = {k: v for k, v in votes.items() if v}
    if not valid:
        return "", "三家均无有效结论", "no_vote"

    counter = Counter(valid.values())
    total = len(valid)
    desc = " ".join(f"{short}={votes.get(name) or '失败'}"
                    for name, short in ((m.name, m.short) for m in MODELS))

    if len(counter) == 1:
        only = next(iter(counter))
        return only, f"{desc} → {total}:0", "一致"

    top, cnt = counter.most_common(1)[0]
    if cnt >= 2:
        return top, f"{desc} → {cnt}:{total - cnt}", "多数票"

    # 无多数：以仲裁者为准
    arb = votes.get(ARBITER.name)
    if arb:
        return arb, f"{desc} → {':'.join(str(c) for c in counter.values())} 无多数", "仲裁者定值"
    return "", f"{desc} → 无多数且仲裁者无答案", "no_vote"


def process_one(item: Dict[str, Any], seq: int, total: int) -> Dict[str, Any]:
    """处理一道题，返回 {record, log_text, elapsed, degraded}。"""
    qid = str(item.get("id", seq))
    text = item.get("text", "")
    if not isinstance(text, str):
        text = str(text)
    t0 = time.time()

    # ---------- 1. 三个模型并行作答 ----------
    raw = ask_all_models(qid, text)

    # ---------- 2. 每个维度分别取平均（只统计有效票）----------
    per_model: Dict[str, Dict[str, Any]] = {}   # 短名 -> {scores, quality, ok}
    dim_values: Dict[str, List[float]] = {k: [] for k in DIM_KEYS}
    quality_votes: Dict[str, Optional[str]] = {}

    for cfg in MODELS:
        ok, data, err = raw.get(cfg.name, (False, None, ""))
        entry: Dict[str, Any] = {"ok": False, "scores": {}, "quality": None, "err": err}
        if ok and isinstance(data, dict):
            got_any = False
            for key in DIM_KEYS:
                found, val = lookup_value(data, key)
                num = _num(val) if found else None
                if num is not None:
                    v = round(clamp01(num), 2)
                    entry["scores"][key] = v
                    dim_values[key].append(v)
                    got_any = True
            if got_any:
                entry["ok"] = True
            found_q, raw_q = lookup_value(data, "quality")
            entry["quality"] = normalize_quality(raw_q) if found_q else None
            entry["reasoning"] = (data.get("reasoning") or data.get("analysis") or "")
        per_model[cfg.short] = entry
        quality_votes[cfg.name] = entry["quality"]

    # ---------- 3. 维度平均 + 兜底 ----------
    fb_scores, fb_notes = heuristic_scores(text)
    scores: Dict[str, float] = {}
    fallback_dims: List[str] = []
    for key in DIM_KEYS:
        vals = dim_values[key]
        if vals:
            scores[key] = round(sum(vals) / len(vals), 2)
        else:
            scores[key] = fb_scores[key]
            fallback_dims.append(DIM_LABEL[key])

    avg = round(sum(scores[k] for k in DIM_KEYS) / len(DIM_KEYS), 2)

    # ---------- 4. quality 少数服从多数；无多数取仲裁者 ----------
    voted, vote_desc, decision = vote_quality(quality_votes)
    rule_label = decide_quality_by_threshold(avg, scores)
    conflict = False

    if voted:
        quality = voted
        if quality != rule_label:
            conflict = True
            if FORCE_THRESHOLD_CONSISTENCY:
                quality = rule_label
    else:
        quality = rule_label
        decision = "三家均无结论→阈值规则"
        conflict = False

    # ---------- 5. 组装答案 ----------
    record = {
        "id": qid,
        "text": text,
        "score_completeness": float(scores["completeness"]),
        "score_density": float(scores["density"]),
        "score_coherence": float(scores["coherence"]),
        "score_purity": float(scores["purity"]),
        "score_avg": float(avg),
        "thresholds": dict(THRESHOLDS),
        "quality": quality,
    }

    elapsed = time.time() - t0

    # ---------- 6. 控制台日志 ----------
    preview = re.sub(r"\s+", " ", text).strip()
    shown = preview[:100] + "…" if len(preview) > 100 else preview
    lines = [
        hr(),
        f"【题目 id={qid}】（第 {seq}/{total} 题）",
        f"  内容: {shown}（共 {len(text)} 字）",
        "  三模型评分:",
    ]
    for cfg in MODELS:
        e = per_model[cfg.short]
        if e["ok"]:
            s_txt = " ".join(f"{DIM_LABEL[k]}{e['scores'][k]:.2f}" for k in DIM_KEYS if k in e["scores"])
            q_txt = e["quality"] or "（未给出）"
            lines.append(f"      {cfg.short}  {s_txt} → {q_txt}")
        else:
            lines.append(f"      {cfg.short}  调用失败（原因见上方[模型调用失败]提示）")

    lines.append(
        "  维度平均: " + " ".join(f"{DIM_LABEL[k]}{scores[k]:.2f}" for k in DIM_KEYS)
        + f" | 综合 {avg:.2f}"
        + (f"（{'/'.join(fallback_dims)} 无有效分，用规则兜底）" if fallback_dims else "")
    )
    lines.append(f"  仲裁情况: quality {vote_desc} ⇒ {quality.upper()}（{decision}）")
    if conflict:
        arb_reason = str(per_model.get(ARBITER.short, {}).get("reasoning", "")).replace("\n", " ")[:120]
        lines.append(
            f"      ⚠ 口径差异: 三模型投票={voted.upper()}，而按平均值+阈值推导={rule_label.upper()}"
            + ("，已按开关强制改为阈值结论" if FORCE_THRESHOLD_CONSISTENCY else "，按赛题以投票为准")
        )
        if arb_reason:
            lines.append(f"      仲裁模型分析: {arb_reason}")
    lines.append(
        f"  阈值: 完整{THRESHOLDS['completeness']}/密度{THRESHOLDS['density']}"
        f"/连贯{THRESHOLDS['coherence']}/纯净{THRESHOLDS['purity']}/综合{THRESHOLDS['avg']}"
    )
    lines.append(f"  本题耗时 {elapsed:.1f}s")

    degraded = bool(fallback_dims) or (decision == "no_vote")
    return {"record": record, "log_text": "\n".join(lines), "elapsed": elapsed, "degraded": degraded}


# ==============================================================================
# ============================== 文件读写 =====================================
# ==============================================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        log(f"[错误] 找不到题目文件：{path}")
        log("       请修改脚本顶部 INPUT_FILE 为你的实际路径。")
        sys.exit(1)
    items: List[Dict[str, Any]] = []
    bad = 0
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
                else:
                    bad += 1
                    log(f"  [警告] 第{line_no}行不是 JSON 对象，已跳过")
            except Exception as e:
                bad += 1
                log(f"  [警告] 第{line_no}行解析失败（{e}），已跳过")
    log(f"[读取] 题目文件: {path}")
    log(f"       有效题目 {len(items)} 条" + (f"，跳过异常行 {bad} 条" if bad else ""))
    return items


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    try:
        os.replace(tmp, path)
    except OSError:
        # 目标文件被占用（例如正被 Excel 打开）时退化为直接覆盖写
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        try:
            os.remove(tmp)
        except OSError:
            pass


# ==============================================================================
# ============================== 连通性自检 ===================================
# ==============================================================================

def preflight() -> None:
    log(hr("═"))
    log("[自检] 正在验证三个大模型的连通性与鉴权…")
    results: Dict[str, str] = {}
    lock = threading.Lock()

    def check(cfg: ModelConfig) -> None:
        t0 = time.time()
        try:
            content = _chat_once(cfg, [{"role": "user", "content": "只回复两个字：正常"}], 0.0)
            msg = f"OK（{time.time() - t0:.1f}s）返回: {content.strip()[:20]!r}"
        except ModelCallError as e:
            msg = f"失败 → {e.reason}"
        except Exception as e:  # noqa: BLE001
            msg = f"失败 → {type(e).__name__}: {e}"
        with lock:
            results[cfg.name] = msg

    threads = [threading.Thread(target=check, args=(m,), daemon=True) for m in MODELS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    all_ok = True
    for m in MODELS:
        r = results.get(m.name, "未执行")
        ok = r.startswith("OK")
        all_ok = all_ok and ok
        log(f"  {'✔' if ok else '✘'} {m.name:<28s} {m.model:<18s} {r}")
    if not all_ok:
        log("  [提示] 有模型自检未通过：请先检查 API_KEY / 模型名 / 余额 / 网络，"
            "否则该模型的分数与投票会缺失，平均值将只基于剩余模型计算。")
    log(hr("═"))


def check_api_keys() -> None:
    bad = [m.name for m in MODELS if "请替换" in m.api_key or not m.api_key.strip()]
    if bad:
        log("[错误] 以下模型的 API_KEY 还没替换，请修改脚本顶部 CONFIG 区域后重新运行：")
        for n in bad:
            log(f"       - {n}")
        sys.exit(1)


# ==============================================================================
# ================================ 主流程 =====================================
# ==============================================================================

def main() -> None:
    try:  # 让 Windows 控制台也按 UTF-8 输出，避免中文/特殊符号报错
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    check_api_keys()

    log(hr("═"))
    log("人工智能训练师竞赛 · 高质量数据集质量评测（三模型评分 + 多数票判定）")
    log(hr("═"))
    log("  参评模型 : " + " / ".join(m.name for m in MODELS))
    log(f"  仲裁模型 : {ARBITER.name}（无多数时以它为准）")
    log(f"  并发数   : {MAX_WORKERS}（同时处理 {MAX_WORKERS} 道题，每题 3 模型并行）")
    log("  计分规则 : 四个维度分别取三模型平均分；综合得分 = 四项平均")
    log("  判定规则 : quality 少数服从多数（3:0 / 2:1）；无多数则取仲裁者结论")
    log(f"  阈值设置 : 完整{THRESHOLDS['completeness']} / 密度{THRESHOLDS['density']} / "
        f"连贯{THRESHOLDS['coherence']} / 纯净{THRESHOLDS['purity']} / 综合{THRESHOLDS['avg']}")
    log(hr("═"))

    items = load_jsonl(INPUT_FILE)
    if not items:
        log("[错误] 题目为空，退出。")
        return

    if PREFLIGHT_CHECK:
        preflight()

    total = len(items)
    results: Dict[int, Dict[str, Any]] = {}
    res_lock = threading.Lock()
    done = [0]
    degraded_cnt = [0]
    t_start = time.time()

    log(hr("═"))
    log(f"[开始] 共 {total} 题，并发 {MAX_WORKERS}，每题 3 次模型调用…")
    log(hr("═"))

    def worker(idx: int, item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        try:
            return idx, process_one(item, idx + 1, total)
        except Exception as e:  # noqa: BLE001  单题异常不影响整体
            qid = str(item.get("id", idx + 1))
            text = item.get("text", "") if isinstance(item.get("text"), str) else str(item.get("text", ""))
            s, _notes = heuristic_scores(text)
            avg = round(sum(s[k] for k in DIM_KEYS) / len(DIM_KEYS), 2)
            log(f"    [异常] 题目id={qid} 处理失败：{type(e).__name__}: {e}（已用规则兜底）")
            rec = {
                "id": qid, "text": text,
                "score_completeness": s["completeness"], "score_density": s["density"],
                "score_coherence": s["coherence"], "score_purity": s["purity"],
                "score_avg": avg, "thresholds": dict(THRESHOLDS),
                "quality": decide_quality_by_threshold(avg, s),
            }
            return idx, {"record": rec,
                         "log_text": f"【题目 id={qid}】异常兜底，判定 {rec['quality'].upper()}",
                         "elapsed": 0.0, "degraded": True}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="q") as pool:
        futures = [pool.submit(worker, i, it) for i, it in enumerate(items)]
        for fut in as_completed(futures):
            idx, res = fut.result()
            with res_lock:
                results[idx] = res
                done[0] += 1
                if res["degraded"]:
                    degraded_cnt[0] += 1
                cur = done[0]
                elapsed_total = time.time() - t_start
                speed = cur / elapsed_total if elapsed_total > 0 else 0
                eta = (total - cur) / speed if speed > 0 else 0
            log_block(res["log_text"])  # 每题日志整块输出
            log(f"  【进度】{cur}/{total} 完成（{cur / total * 100:.1f}%）| 已用 {fmt_dur(elapsed_total)}"
                f" | 预计剩余 {fmt_dur(eta)}")
            if cur % SAVE_EVERY == 0 or cur == total:  # 断点落盘
                with res_lock:
                    snap = [results[i]["record"] for i in sorted(results)]
                try:
                    write_jsonl(OUTPUT_FILE, snap)
                except Exception as e:  # noqa: BLE001
                    log(f"  [警告] 中途落盘失败：{e}")

    # ---------------- 最终写入（按原始顺序）----------------
    final_records = [results[i]["record"] for i in range(total) if i in results]
    write_jsonl(OUTPUT_FILE, final_records)

    # ---------------- 汇总 ----------------
    total_elapsed = time.time() - t_start
    high = sum(1 for r in final_records if r["quality"] == "high")
    low = len(final_records) - high
    per = [results[i]["elapsed"] for i in range(total) if i in results]
    avg_t = sum(per) / len(per) if per else 0.0

    log(hr("═"))
    log("【运行汇总】")
    log(f"  题目总数   : {total}")
    log(f"  成功输出   : {len(final_records)} 条"
        + (f"（其中 {degraded_cnt[0]} 题存在模型失败/兜底）" if degraded_cnt[0] else ""))
    log(f"  判定分布   : high={high} | low={low}")
    log(f"  单题平均   : {avg_t:.1f}s   最长 {max(per) if per else 0:.1f}s")
    log(f"  总耗时     : {fmt_dur(total_elapsed)}（{total_elapsed:.1f} 秒）")
    log(f"  输出文件   : {OUTPUT_FILE}")
    log(hr("═"))


if __name__ == "__main__":
    main()
