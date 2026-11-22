# -*- coding: utf-8 -*-
"""
v5.py —— 人工智能训练师复赛实操：SPO 关系抽取（三模型独立抽取 + 百炼仲裁）

【单文件 / 仅用 Python 标准库 / 在 PyCharm 里直接点“运行”即可】
使用前只需在下方【配置区】把 3 个 API_KEY 换成你自己的，文件名按需替换。
所有文件读写都发生在本脚本所在目录，不需要命令行参数、不需要安装第三方库。

抽取模型（三个模型各自独立输出 spo_list）：
  1. 硅基流动        Qwen/Qwen3-32B
  2. 讯飞星火 MaaS   xopglm53
  3. 讯飞星火 MaaS   xsparkx2flash
仲裁模型：
  4. 阿里云百炼      qwen3.8-27b（对三份 spo_list 对比后裁定最终答案）

讯飞星火官网：https://maas.xfyun.cn/ （新版 MaaS 平台，OpenAI 兼容接口，非老版 ws 接口）
  OpenAI Chat     ：https://maas-api.cn-huabei-1.xf-yun.com/v2/chat/completions
  OpenAI Response ：https://maas-api.cn-huabei-1.xf-yun.com/v1/responses
  Anthropic       ：https://maas-api.cn-huabei-1.xf-yun.com/anthropic/v1/messages
  本脚本使用 OpenAI Chat 接口（最通用）。

输入 ：人工智能训练师复赛实操题.jsonl
Schema：人工智能训练师复赛实操题关系Schema.json
输出 ：人工智能训练师初赛实操题-a1.jsonl
"""

import json
import os
import re
import time
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 配置区（按需修改，其余代码不用动）
# ============================================================

# ---------- 文件名（与代码放在同一目录，可自行替换） ----------
SCHEMA_FILE = "人工智能训练师复赛实操题关系Schema.json"
INPUT_FILE = "人工智能训练师复赛实操题.jsonl"
OUTPUT_FILE = "人工智能训练师初赛实操题-a1.jsonl"

# ---------- 阿里云百炼：仲裁模型 ----------
DASHSCOPE_API_KEY = "sk-"
DASHSCOPE_MODEL = "qwen3.8-27b"   # 官网 id 为小写；若控制台显示不同请按控制台修改
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# ---------- 硅基流动：抽取模型 1 ----------
SILICONFLOW_API_KEY = "sk-"
SILICONFLOW_MODEL = "Qwen/Qwen3-32B"
SILICONFLOW_URL = "https://api.siliconflow.cn/v1/chat/completions"

# ---------- 讯飞星火 MaaS：抽取模型 2、3 ----------
# API Key 形如 ak-xxx，在 https://maas.xfyun.cn/ 控制台创建；两个模型共用同一把 Key。
XFYUN_API_KEY = "ak-"          # 例如 "ak-xxxxxxxxxxxxxxxx"
XFYUN_URL = "https://maas-api.cn-huabei-1.xf-yun.com/v2/chat/completions"
XFYUN_MODEL_A = "xopglm53"
XFYUN_MODEL_B = "xsparkx2flash"

# ---------- 并发策略 ----------
MAX_WORKERS = 4        # 同时处理几道题（默认 4）
API_POOL_WORKERS = 12  # 底层 HTTP 请求最大并发数（每道题 3 个抽取 + 1 个仲裁）

# ---------- 失败重试与超时 ----------
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1.5     # 第 n 次重试等待 n*基数 秒
REQUEST_TIMEOUT_SECONDS = 120

# ---------- 生成参数 ----------
# 注意：讯飞星火官网 temperature 取值范围是 (0, 2]，传 0 会返回 400！这里统一用 0.1。
TEMPERATURE = 0.1
MAX_TOKENS = 4096

# 硅基流动 Qwen3 官方支持 enable_thinking，关闭思考以便稳定拿到 JSON
SILICONFLOW_ENABLE_THINKING = False
# 百炼 qwen3.8-27b：混合思考模型，关闭思考；若你的账号报参数错误，把它改成 None 表示不发送该参数
DASHSCOPE_ENABLE_THINKING = False
# 讯飞 xsparkx2flash 默认开启思考，显式关闭；xopglm53 是开源模型，不发 thinking 字段
DISABLE_XF_SPARK_THINKING = True
# 讯飞 token 上限参数名：官网 OpenAI 兼容文档用 max_tokens。
# 若你的服务返回“未知参数”错误，改成 "max_completion_tokens"；改成 "" 则完全不发送。
XFYUN_MAX_TOKENS_PARAM = "max_tokens"

# 把 Schema 中每条关系的语义说明一并发给模型（有助于区分“创立/研发”“收购/计划收购”等）
INCLUDE_PREDICATE_DESCRIPTION = True

# ---------- 启动自检：正式答题前逐个 ping 四个模型，打印哪个通、哪个不通 ----------
PREFLIGHT_CHECK = True
PREFLIGHT_TIMEOUT_SECONDS = 40

# ---------- 模型标识（失败时打印用） ----------
PROVIDERS = {
    "siliconflow": {
        "platform": "硅基流动", "model": SILICONFLOW_MODEL,
        "url": SILICONFLOW_URL, "api_key": SILICONFLOW_API_KEY, "role": "抽取",
    },
    "xfyun_xopglm53": {
        "platform": "讯飞星火MaaS", "model": XFYUN_MODEL_A,
        "url": XFYUN_URL, "api_key": XFYUN_API_KEY, "role": "抽取",
    },
    "xfyun_xsparkx2flash": {
        "platform": "讯飞星火MaaS", "model": XFYUN_MODEL_B,
        "url": XFYUN_URL, "api_key": XFYUN_API_KEY, "role": "抽取",
    },
    "dashscope": {
        "platform": "阿里云百炼", "model": DASHSCOPE_MODEL,
        "url": DASHSCOPE_URL, "api_key": DASHSCOPE_API_KEY, "role": "仲裁",
    },
}
EXTRACT_PROVIDERS = ["siliconflow", "xfyun_xopglm53", "xfyun_xsparkx2flash"]
ARBITER_PROVIDER = "dashscope"

# ============================================================
# 提示词
# ============================================================

SYSTEM_PROMPT = (
    "你是一名知识抽取专员。你的唯一任务是从给定中文文本中抽取实体 SPO 三元组"
    "（主体—谓语—客体），只输出严格 JSON，不要输出解释、Markdown 或多余文字。"
)

USER_TEMPLATE = """文本：
{text}
请抽取上述文本中所有明确表达的事实三元组，规则：
1. 只抽文本中明确陈述的事实，不推断、不臆测、不补常识。
2. subject 和 object 尽量使用原文中的实体表述，不改写、不翻译、不泛化。
3. predicate 必须且只能从候选集合中选取：{predicate_list}；不属于集合的关系不要输出。
4. 严格输出：{{"spo_list":[{{"subject":"...","predicate":"...","object":"..."}}]}}，不要外层解释。
5. 无可抽取内容时输出 {{"spo_list":[]}}。
6. 不要输出 id、text 或其他字段。"""

ARBITER_SYSTEM_PROMPT = (
    "你是一名知识抽取仲裁专员。你的任务是比较多个候选 SPO 三元组结果，"
    "结合原文与全局关系 Schema 裁定最终答案。只输出严格 JSON，"
    "不要输出解释、Markdown 或多余文字。"
)

ARBITER_TEMPLATE = """原文：
{text}

三个模型各自独立抽取的候选结果：
{candidates}

全局关系 Schema（predicate 及其语义说明）：
{schema}

请裁定最终的 spo_list，规则：
1. 只保留原文明确陈述的事实，不推断、不臆测、不补常识；Schema 中未定义的关系一律不要输出。
2. subject 和 object 必须采用原文中能够准确、唯一指代该实体的名称，不得扩写、改写，
   也不得凭外部知识补充文本未出现的实体名称。
3. predicate 必须且只能从上述 Schema 的 predicate 中选取，并符合其语义说明
   （例如：仅宣布/计划而未完成的收购不算“收购”，仅计划航线不算“通航至”，
   仅同时出现不算“合作”，软件/算法/技术用“研发”而非“创立”）。
4. 逐个候选三元组核对：三个候选都给出的可采纳；只有个别候选给出的，必须回到原文逐字核验，
   确有明确依据且符合 Schema 语义才保留；相互关系用错、主体或客体张冠李戴的候选一律剔除。
5. 三个候选均未列出、但原文明确成立且符合 Schema 的三元组，可以补充。
6. 同一 (subject, predicate, object) 只保留一次，实体名称保持原文一致。
7. 严格输出：{{"spo_list":[{{"subject":"...","predicate":"...","object":"..."}}]}}，
   无结果时输出 {{"spo_list":[]}}。
8. 不要输出 id、text、候选编号或任何解释性文字。"""

# Schema 读取失败时的兜底关系清单（与本赛题 Schema 一致）
FALLBACK_PREDICATES = [
    "下令营建", "列入", "位于", "创立", "发布", "收购", "应用于", "获得",
    "创作", "翻译", "合作", "发生于", "起源于", "担任", "前身", "更名为",
    "隶属于", "任职于", "毕业于", "连接", "途经", "跨越", "迁至", "组成",
    "效力于", "击败", "通航至", "举办于", "研发", "搭载", "着陆于",
]

_print_lock = threading.Lock()
PROGRAM_START = time.time()


# ============================================================
# 通用工具
# ============================================================

def safe_print(*args, **kwargs):
    """多线程安全的控制台输出，并立即刷新。"""
    with _print_lock:
        print(*args, **kwargs)
        try:
            import sys
            sys.stdout.flush()
        except Exception:
            pass


def total_elapsed():
    return time.time() - PROGRAM_START


def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def model_name(provider_key):
    info = PROVIDERS[provider_key]
    return "%s %s（%s）" % (info["platform"], info["model"], info["role"])


def check_api_key(key, name):
    if (not key) or ("YOUR_" in key) or ("替换" in key) or ("REPLACE" in key.upper()):
        safe_print("[警告] %s 尚未替换成你自己的 API_KEY，相关调用将会失败。" % name)


# ============================================================
# 文件读写（全部在本脚本所在目录）
# ============================================================

def load_schema(base_dir):
    """读取全局 Schema，返回 (predicate 列表, predicate->description 映射)。"""
    path = os.path.join(base_dir, SCHEMA_FILE)
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, dict):  # 兼容 {"predicates": [...]} 之类的包装写法
        for key in ("predicates", "schema", "data", "relations"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("Schema 顶层必须是数组，每个元素含 predicate 与 description")

    predicates, descriptions = [], {}
    for item in data:
        if not isinstance(item, dict):
            continue
        p = clean_text(item.get("predicate"))
        if not p:
            continue
        predicates.append(p)
        descriptions[p] = clean_text(item.get("description"))

    seen, unique = set(), []
    for p in predicates:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    if not unique:
        raise ValueError("Schema 中没有解析到任何 predicate")
    return unique, descriptions


def build_predicate_list_text(predicates, descriptions):
    lines = []
    for p in predicates:
        d = descriptions.get(p, "")
        if INCLUDE_PREDICATE_DESCRIPTION and d:
            lines.append("- %s：%s" % (p, d))
        else:
            lines.append("- %s" % p)
    return "；".join(lines)


def load_items(base_dir):
    """逐行读取 JSONL，容错跳过坏行。"""
    path = os.path.join(base_dir, INPUT_FILE)
    items = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                safe_print("[警告] 输入文件第 %d 行不是合法 JSON，已跳过：%s" % (line_no, e))
                continue
            if not isinstance(obj, dict) or "id" not in obj or "text" not in obj:
                safe_print("[警告] 输入文件第 %d 行缺少 id 或 text，已跳过" % line_no)
                continue
            items.append({"id": obj["id"], "text": clean_text(obj["text"])})
    return items


# ============================================================
# HTTP 调用与响应解析
# ============================================================

def http_post_json(url, api_key, payload, timeout):
    """用标准库发 POST，失败时抛出带明确原因的异常（含 HTTP 状态与响应体片段）。"""
    headers = {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:800]
        except Exception:
            pass
        raise RuntimeError("HTTP %s %s，响应：%s" % (e.code, e.reason, detail))
    except urllib.error.URLError as e:
        raise RuntimeError("网络错误：%s" % getattr(e, "reason", e))
    except Exception as e:
        raise RuntimeError("请求异常：%s" % e)
    try:
        data = json.loads(raw)
    except Exception:
        raise RuntimeError("返回内容不是合法 JSON：%s" % raw[:800])

    # 讯飞星火等平台即使 HTTP 200 也可能在 body 里返回非零 code，必须检查
    if isinstance(data, dict) and "code" in data:
        code = data.get("code")
        if code not in (0, "0", None, ""):
            raise RuntimeError("接口返回错误码 code=%s，message=%s"
                               % (code, data.get("message", "")))
    if isinstance(data, dict) and "error" in data and data["error"]:
        raise RuntimeError("接口返回错误：%s"
                           % json.dumps(data["error"], ensure_ascii=False)[:500])
    return data


def extract_content(data):
    """从 OpenAI 兼容响应中取出 message.content。"""
    if not isinstance(data, dict):
        raise RuntimeError("返回结构异常：%s" % str(data)[:300])
    try:
        message = data["choices"][0]["message"]
    except Exception:
        raise RuntimeError("返回结构异常，未找到 choices[0].message：%s"
                           % json.dumps(data, ensure_ascii=False)[:500])
    content = message.get("content")
    if content is None or (isinstance(content, str) and content.strip() == ""):
        # 少数思考模型把结果放进 reasoning_content，这里做兜底
        reasoning = message.get("reasoning_content")
        if reasoning:
            content = reasoning
        else:
            raise RuntimeError("content 为空：%s" % json.dumps(message, ensure_ascii=False)[:500])
    if isinstance(content, list):  # 少数接口把内容拆成片段列表
        parts = []
        for seg in content:
            if isinstance(seg, dict):
                parts.append(str(seg.get("text", "")))
            else:
                parts.append(str(seg))
        content = "".join(parts)
    return str(content)


def strip_thinking(text):
    s = text
    for tag in ("think", "analysis", "reasoning"):
        s = re.sub(r"<%s>.*?</%s>" % (tag, tag), "", s, flags=re.S | re.I)
        s = re.sub(r"＜%s＞.*?＜/%s＞" % (tag, tag), "", s, flags=re.S)
    return s.strip()


def parse_spo_content(content, allowed_predicates):
    """把模型输出鲁棒解析为规范化 spo_list：过滤 Schema 外关系、去重。"""
    if content is None:
        raise RuntimeError("模型返回内容为空")
    if isinstance(content, list):
        content = "".join(
            str(seg.get("text", "")) if isinstance(seg, dict) else str(seg)
            for seg in content
        )
    if not isinstance(content, str):
        content = str(content)

    s = strip_thinking(content)
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```\s*$", "", s)
    s = s.strip()

    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("模型输出中找不到 JSON 对象：%s" % s[:300])
    candidate = s[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except Exception:
        parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))  # 去尾逗号再试

    if isinstance(parsed, list):
        raw_list = parsed
    elif isinstance(parsed, dict):
        raw_list = parsed.get("spo_list", parsed.get("data", []))
    else:
        raw_list = []
    if raw_list is None:
        raw_list = []
    if not isinstance(raw_list, list):
        raise RuntimeError("spo_list 不是数组：%s" % str(raw_list)[:300])

    allowed = set(allowed_predicates)
    result, seen = [], set()
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        subject = clean_text(item.get("subject"))
        predicate = clean_text(item.get("predicate"))
        object_ = clean_text(item.get("object"))
        if not (subject and predicate and object_):
            continue
        if predicate not in allowed:  # 不属于 Schema 的关系不纳入
            continue
        key = (subject, predicate, object_)
        if key in seen:
            continue
        seen.add(key)
        result.append({"subject": subject, "predicate": predicate, "object": object_})
    return result


def build_payload(provider_key, messages):
    """按各平台官方文档构造请求体。"""
    payload = {
        "model": PROVIDERS[provider_key]["model"],
        "messages": messages,
        "temperature": TEMPERATURE,   # 0.1：对百炼/硅基/讯飞都合法（讯飞要求 > 0）
        "stream": False,
    }
    if provider_key == "dashscope":
        payload["max_tokens"] = MAX_TOKENS
        if DASHSCOPE_ENABLE_THINKING is not None:
            payload["enable_thinking"] = DASHSCOPE_ENABLE_THINKING  # HTTP 直调放顶层即可
    elif provider_key == "siliconflow":
        payload["max_tokens"] = MAX_TOKENS
        payload["enable_thinking"] = SILICONFLOW_ENABLE_THINKING   # 官方支持该参数
    else:  # 讯飞星火 MaaS 新接口（OpenAI Chat）
        if XFYUN_MAX_TOKENS_PARAM:
            payload[XFYUN_MAX_TOKENS_PARAM] = MAX_TOKENS
        if provider_key == "xfyun_xsparkx2flash" and DISABLE_XF_SPARK_THINKING:
            payload["thinking"] = {"type": "disabled"}  # 官网：enabled / disabled / auto
        # xopglm53 是开源模型，官方约定不要发 thinking 字段
    return payload


def call_model(provider_key, messages, item_id, validator=None,
               retries=None, timeout=None):
    """
    调用一次大模型，含重试。
    validator 用于校验/解析 content（返回规范化结果；抛异常视为本次调用失败并重试）。
    成功返回 validator 的结果；最终失败返回 None，并已在控制台打印模型名与原因。
    """
    info = PROVIDERS[provider_key]
    attempts = MAX_RETRIES if retries is None else retries
    limit = REQUEST_TIMEOUT_SECONDS if timeout is None else timeout
    for attempt in range(1, attempts + 1):
        try:
            data = http_post_json(info["url"], info["api_key"],
                                  build_payload(provider_key, messages), limit)
            content = extract_content(data)
            if validator is not None:
                return validator(content)
            return content
        except Exception as e:
            safe_print("[大模型调用失败] %s | 题目id=%s | 第%d/%d次 | 原因：%s"
                       % (model_name(provider_key), item_id, attempt, attempts, e))
            if attempt < attempts:
                time.sleep(RETRY_BASE_SECONDS * attempt)
    return None


# ============================================================
# 启动自检：逐个模型 ping 一下，打印哪个通、哪个不通
# ============================================================

def preflight_check():
    probe = [
        {"role": "system", "content": "你是一个测试助手，只输出一个 JSON 对象，不要多余文字。"},
        {"role": "user", "content": "请严格只输出：{\"spo_list\":[]}"},
    ]
    safe_print("=" * 78)
    safe_print("【启动自检】正在逐个测试 4 个模型是否可调用……")
    status = {}
    for key in EXTRACT_PROVIDERS + [ARBITER_PROVIDER]:
        content = call_model(key, probe, "自检", validator=None,
                             retries=1, timeout=PREFLIGHT_TIMEOUT_SECONDS)
        status[key] = content is not None
        if content is not None:
            safe_print("  [自检通过] %s" % model_name(key))
        else:
            safe_print("  [自检失败] %s —— 请按上方失败原因检查 Key/模型ID/参数" % model_name(key))
    safe_print("自检完成：%d/%d 个模型可用" % (sum(status.values()), len(status)))
    safe_print("=" * 78)
    return status


# ============================================================
# 抽取 / 仲裁 / 兜底
# ============================================================

def build_extract_messages(text, predicate_list_text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            text=text, predicate_list=predicate_list_text)},
    ]


def format_candidates(candidates):
    lines = []
    for label, key in zip("ABC", EXTRACT_PROVIDERS):
        info = PROVIDERS[key]
        spo = candidates.get(key)
        if spo is None:
            body = "调用失败，无候选结果"
        else:
            body = json.dumps({"spo_list": spo}, ensure_ascii=False)
        lines.append("%s（%s %s）：%s" % (label, info["platform"], info["model"], body))
    return "\n".join(lines)


def build_arbitration_messages(text, candidates, predicate_list_text):
    return [
        {"role": "system", "content": ARBITER_SYSTEM_PROMPT},
        {"role": "user", "content": ARBITER_TEMPLATE.format(
            text=text,
            candidates=format_candidates(candidates),
            schema=predicate_list_text)},
    ]


def fallback_merge(candidates):
    """仲裁模型不可用时的确定性兜底：多数投票 -> 无多数则取并集。"""
    lists = [v for v in candidates.values() if v is not None]
    if not lists:
        return []
    if len(lists) == 1:
        return [dict(t) for t in lists[0]]

    counter, order = {}, []
    for lst in lists:
        for t in lst:
            k = (t.get("subject"), t.get("predicate"), t.get("object"))
            counter[k] = counter.get(k, 0) + 1
            if k not in order:
                order.append(k)
    result = [
        {"subject": k[0], "predicate": k[1], "object": k[2]}
        for k in order if counter[k] >= 2
    ]
    if not result:  # 三者完全不一致时取并集，尽量保住召回
        seen, result = set(), []
        for lst in lists:
            for t in lst:
                k = (t.get("subject"), t.get("predicate"), t.get("object"))
                if k in seen:
                    continue
                seen.add(k)
                result.append(dict(t))
    return result


def solve_one(item, context, api_pool):
    """解一道题：3 个抽取模型并发 -> 1 个仲裁模型裁定。"""
    item_id = item["id"]
    text = item["text"]
    t0 = time.time()
    validator = lambda c: parse_spo_content(c, context["allowed_predicates"])

    # 三个抽取模型并发请求
    future_map = {}
    for key in EXTRACT_PROVIDERS:
        messages = build_extract_messages(text, context["predicate_list_text"])
        future_map[api_pool.submit(call_model, key, messages, item_id, validator)] = key

    candidates = {key: None for key in EXTRACT_PROVIDERS}
    for future in as_completed(future_map):
        key = future_map[future]
        try:
            candidates[key] = future.result()
        except Exception as e:  # 理论上 call_model 内部已捕获，这里仅作兜底
            safe_print("[内部错误] %s | 题目id=%s | 原因：%s" % (model_name(key), item_id, e))
            candidates[key] = None

    ok_count = sum(1 for v in candidates.values() if v is not None)
    if ok_count == 0:
        final = []
        status = "三抽取模型均失败"
    else:
        messages = build_arbitration_messages(text, candidates, context["predicate_list_text"])
        final = api_pool.submit(call_model, ARBITER_PROVIDER, messages, item_id, validator).result()
        if final is None:
            final = fallback_merge(candidates)
            status = "仲裁失败-已兜底"
        else:
            status = "仲裁完成"

    elapsed = time.time() - t0
    return item_id, final, elapsed, status, ok_count


# ============================================================
# 主流程
# ============================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(base_dir, SCHEMA_FILE)
    input_path = os.path.join(base_dir, INPUT_FILE)
    output_path = os.path.join(base_dir, OUTPUT_FILE)
    workers = max(1, int(MAX_WORKERS))

    safe_print("=" * 78)
    safe_print("SPO 知识抽取任务启动（三模型独立抽取 + 百炼仲裁）")
    safe_print("抽取模型 1：%s" % model_name("siliconflow"))
    safe_print("抽取模型 2：%s" % model_name("xfyun_xopglm53"))
    safe_print("抽取模型 3：%s" % model_name("xfyun_xsparkx2flash"))
    safe_print("仲裁模型  ：%s" % model_name("dashscope"))
    safe_print("讯飞星火接口：%s（MaaS 新平台）" % XFYUN_URL)
    safe_print("并发题数：%d | 底层请求并发：%d | 单模型重试次数：%d"
               % (workers, API_POOL_WORKERS, MAX_RETRIES))
    safe_print("Schema：%s" % schema_path)
    safe_print("输入  ：%s" % input_path)
    safe_print("输出  ：%s" % output_path)
    safe_print("=" * 78)

    check_api_key(DASHSCOPE_API_KEY, "阿里云百炼 API_KEY")
    check_api_key(SILICONFLOW_API_KEY, "硅基流动 API_KEY")
    check_api_key(XFYUN_API_KEY, "讯飞星火 MaaS API_KEY")

    for path, name in ((schema_path, "Schema"), (input_path, "输入")):
        if not os.path.exists(path):
            safe_print("[错误] 找不到%s文件：%s" % (name, path))
            safe_print("请把该文件放到本脚本所在目录，或修改脚本顶部的文件名常量。")
            return

    try:
        predicates, descriptions = load_schema(base_dir)
    except Exception as e:
        safe_print("[错误] 读取 Schema 失败：%s" % e)
        safe_print("改用内置兜底关系清单（%d 个）。" % len(FALLBACK_PREDICATES))
        predicates = list(FALLBACK_PREDICATES)
        descriptions = {p: "" for p in predicates}
    predicate_list_text = build_predicate_list_text(predicates, descriptions)
    safe_print("已加载 Schema：共 %d 个候选关系" % len(predicates))

    # 启动自检：先确认每个模型都能调通，避免跑完全部题目才发现某个模型一直失败
    if PREFLIGHT_CHECK:
        preflight_check()

    try:
        items = load_items(base_dir)
    except Exception as e:
        safe_print("[错误] 读取输入文件失败：%s" % e)
        return
    total = len(items)
    if total == 0:
        safe_print("[错误] 输入文件中没有可处理的题目。")
        return
    safe_print("共 %d 道题目，开始并发处理……" % total)

    context = {
        "predicate_list_text": predicate_list_text,
        "allowed_predicates": set(predicates),
    }
    results = [[] for _ in items]
    failed_ids = []

    with ThreadPoolExecutor(max_workers=workers) as question_pool:
        with ThreadPoolExecutor(max_workers=API_POOL_WORKERS) as api_pool:
            future_map = {
                question_pool.submit(solve_one, item, context, api_pool): idx
                for idx, item in enumerate(items)
            }
            finished = 0
            for future in as_completed(future_map):
                idx = future_map[future]
                item_id = items[idx]["id"]
                finished += 1
                try:
                    _, spo_list, item_seconds, status, ok_count = future.result()
                except Exception as e:
                    spo_list, item_seconds, status, ok_count = [], 0.0, "任务异常：%s" % e, 0
                results[idx] = spo_list
                if ok_count == 0 or status.startswith("仲裁失败") or status.startswith("任务异常"):
                    failed_ids.append(item_id)
                safe_print(
                    "[进度 %d/%d] id=%s | 本题耗时=%.2fs | 总耗时=%.2fs | "
                    "抽取成功=%d/3 | %s | 三元组=%d"
                    % (finished, total, item_id, item_seconds, total_elapsed(),
                       ok_count, status, len(spo_list))
                )

    # 按输入原顺序写出结果
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for item, spo_list in zip(items, results):
                record = {
                    "id": item["id"],
                    "text": item["text"],
                    "spo_list": spo_list,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        safe_print("[错误] 写出结果文件失败：%s" % e)
        return

    non_empty = sum(1 for spo_list in results if spo_list)
    safe_print("=" * 78)
    safe_print("全部完成！共 %d 题，异常题 %d 道。" % (total, len(failed_ids)))
    if failed_ids:
        safe_print("异常题目 id：%s" % failed_ids)
    safe_print("有三元组 %d 题，空结果 %d 题。" % (non_empty, total - non_empty))
    safe_print("总耗时：%.2f 秒" % total_elapsed())
    safe_print("结果已写入：%s" % output_path)
    safe_print("=" * 78)


if __name__ == "__main__":
    main()
