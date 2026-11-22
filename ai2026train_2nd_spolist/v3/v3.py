# -*- coding: utf-8 -*-
"""
v3.py —— 人工智能训练师复赛实操：关系抽取（SPO 三元组）自动作答

【单文件 / 仅标准库 / PyCharm 中直接点运行即可】
使用前只需在下方【配置区】把 API_KEY 换成你自己的讯飞星火 MaaS 密钥；
三个文件名常量如与实际不一致再改。所有文件都读写本脚本所在目录。

输入 ：人工智能训练师复赛实操题.jsonl
Schema：人工智能训练师复赛实操题关系Schema.json
输出 ：人工智能训练师初赛实操题-a1.jsonl
平台 ：讯飞星火 MaaS 新平台（https://maas.xfyun.cn/），OpenAI 兼容接口
模型 ：xopglm53（唯一使用的模型）
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

# 你的讯飞星火 MaaS API Key（自己替换，从 https://maas.xfyun.cn/ 的服务管控页面获取）
API_KEY = "ak-"

# 讯飞星火 MaaS 新接口地址（OpenAI 兼容写法，不要用老版 ws:// 星火接口）
# 2026-01-10 及之后发布的服务用 v2（当前推荐）；此前存量服务可继续用 v1。
# 若你的服务卡片上显示的 API Base 与此不同，请以服务卡片上的地址为准修改本行。
API_BASE = "https://maas-api.cn-huabei-1.xf-yun.com/v2"
# 存量服务可改用：API_BASE = "http://maas-api.cn-huabei-1.xf-yun.com/v1"
API_URL = API_BASE.rstrip("/") + "/chat/completions"

# 指定使用的模型（唯一模型）
MODEL_NAME = "xopglm53"

# 并发数（默认同时处理 4 道题）
MAX_WORKERS = 4

# 失败重试
MAX_RETRIES = 3
RETRY_INTERVAL_SECONDS = 2      # 每次重试前等待基数（秒），逐次递增

# 单次请求超时（秒）
REQUEST_TIMEOUT_SECONDS = 120

# 生成参数
TEMPERATURE = 0
MAX_TOKENS = 2048

# 是否把 Schema 中每条关系的语义说明一并给模型（有助于区分相近关系，建议开启）
INCLUDE_PREDICATE_DESCRIPTION = True

# 是否启用 JSON Mode（xopglm53 属 GLM 系列，新版平台文档称支持 response_format）；
# 若平台提示不支持该参数，重试时会自动关闭，不影响作答。
ENABLE_JSON_MODE = True

# 文件名（与代码在相同目录，按需替换）
SCHEMA_FILE = "人工智能训练师复赛实操题关系Schema.json"
INPUT_FILE = "人工智能训练师复赛实操题.jsonl"
OUTPUT_FILE = "人工智能训练师初赛实操题-a1.jsonl"

# ============================================================
# 提示词（严格使用赛题给定模板）
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

# Schema 文件读取失败时的兜底关系清单（与赛题 Schema 一致）
FALLBACK_PREDICATES = [
    "下令营建", "列入", "位于", "创立", "发布", "收购", "应用于", "获得",
    "创作", "翻译", "合作", "发生于", "起源于", "担任", "前身", "更名为",
    "隶属于", "任职于", "毕业于", "连接", "途经", "跨越", "迁至", "组成",
    "效力于", "击败", "通航至", "举办于", "研发", "搭载", "着陆于",
]

_print_lock = threading.Lock()
PROGRAM_START = time.time()


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
    """从程序启动到当前的累计耗时（秒）。"""
    return time.time() - PROGRAM_START


def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def load_schema(base_dir):
    """读取全局 Schema，返回 (predicate 列表, predicate->description 映射)。"""
    path = os.path.join(base_dir, SCHEMA_FILE)
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    # 兼容 {"predicates": [...]} 之类的包装写法
    if isinstance(data, dict):
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
    """把候选关系拼成提示词里的 {predicate_list}。"""
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


def http_post_json(url, payload):
    """用标准库发 POST 请求，返回解析后的 JSON；失败抛带原因的异常。"""
    headers = {
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:800]
        except Exception:
            pass
        raise RuntimeError("HTTP %s %s，响应：%s" % (e.code, e.reason, detail))
    except urllib.error.URLError as e:
        raise RuntimeError("网络错误：%s" % e.reason)
    except Exception as e:
        raise RuntimeError("请求异常：%s" % e)

    try:
        return json.loads(raw)
    except Exception:
        raise RuntimeError("返回内容不是合法 JSON：%s" % raw[:800])


def call_llm_once(text, predicate_list_text, use_json_mode):
    """调用一次讯飞星火 MaaS 的 xopglm53，返回模型输出的 content 字符串。"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                text=text, predicate_list=predicate_list_text)},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        "search_disable": True,   # 关闭联网搜索，避免引入文本之外的信息
    }
    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}

    data = http_post_json(API_URL, payload)
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError("返回结构异常，未找到 choices[0].message.content：%s"
                           % json.dumps(data, ensure_ascii=False)[:800])
    if content is None:
        content = ""
    if isinstance(content, list):
        # 少数接口把内容拆成片段列表，这里做兼容拼接
        parts = []
        for seg in content:
            if isinstance(seg, dict):
                parts.append(str(seg.get("text", "")))
            else:
                parts.append(str(seg))
        content = "".join(parts)
    return str(content)


def parse_spo_content(content, allowed_predicates):
    """把模型输出鲁棒解析为规范化 spo_list，过滤 Schema 之外的关系并去重。"""
    if not isinstance(content, str):
        raise RuntimeError("模型 content 不是字符串")
    s = content.strip()

    # 去掉可能存在的思考块
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
    s = re.sub(r"＜think＞.*?＜/think＞", "", s, flags=re.S | re.I)

    # 去掉 Markdown 代码围栏
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    s = s.strip()

    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("模型输出中找不到 JSON 对象：%s" % s[:500])

    candidate = s[start:end + 1]
    try:
        obj = json.loads(candidate)
    except Exception:
        # 兜底：去掉 JSON 中多余的尾逗号后再解析一次
        obj = json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))

    if not isinstance(obj, dict):
        raise RuntimeError("模型输出最外层不是 JSON 对象")

    raw_list = obj.get("spo_list", [])
    if raw_list is None:
        raw_list = []
    if not isinstance(raw_list, list):
        raise RuntimeError("spo_list 不是数组")

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
        if predicate not in allowed:          # 不属于 Schema 的关系不纳入
            continue
        key = (subject, predicate, object_)
        if key in seen:                       # 去重
            continue
        seen.add(key)
        result.append({"subject": subject, "predicate": predicate, "object": object_})
    return result


def solve_one(item, predicate_list_text, allowed_predicates):
    """解一道题：失败重试，返回 (spo_list, 本题耗时, 最后错误)。"""
    item_id = item["id"]
    t0 = time.time()
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 第一次尝试 JSON Mode；若平台不支持该参数或输出异常，重试时自动退化为普通请求
            use_json_mode = ENABLE_JSON_MODE and (attempt == 1)
            content = call_llm_once(item["text"], predicate_list_text, use_json_mode)
            spo_list = parse_spo_content(content, allowed_predicates)
            return spo_list, time.time() - t0, None
        except Exception as e:
            last_error = e
            # 【失败打印】明确写出平台、模型、题目 id 和失败原因
            safe_print("[大模型调用失败] 平台=讯飞星火MaaS(maas.xfyun.cn) | 模型=%s "
                       "| 题目id=%s | 第%d/%d次 | 原因：%s"
                       % (MODEL_NAME, item_id, attempt, MAX_RETRIES, e))
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_INTERVAL_SECONDS * attempt)

    return [], time.time() - t0, last_error


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(base_dir, SCHEMA_FILE)
    input_path = os.path.join(base_dir, INPUT_FILE)
    output_path = os.path.join(base_dir, OUTPUT_FILE)

    safe_print("=" * 72)
    safe_print("开始知识抽取任务")
    safe_print("平台：讯飞星火 MaaS（https://maas.xfyun.cn/） | 模型：%s | 并发数：%d"
               % (MODEL_NAME, MAX_WORKERS))
    safe_print("接口：%s" % API_URL)
    safe_print("Schema：%s" % schema_path)
    safe_print("输入  ：%s" % input_path)
    safe_print("输出  ：%s" % output_path)
    safe_print("=" * 72)

    if (not API_KEY) or ("替换" in API_KEY) or ("YOUR" in API_KEY.upper()):
        safe_print("[重要提示] 请先在配置区把 API_KEY 换成你自己的讯飞星火 MaaS 密钥，否则调用会失败。")

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
    allowed_predicates = set(predicates)
    safe_print("已加载 Schema：共 %d 个候选关系" % len(predicates))

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

    results = [[] for _ in items]
    failed_ids = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(solve_one, item, predicate_list_text, allowed_predicates): idx
            for idx, item in enumerate(items)
        }
        finished = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            item = items[idx]
            item_id = item["id"]
            finished += 1
            try:
                spo_list, item_seconds, error = future.result()
            except Exception as e:
                spo_list, item_seconds, error = [], total_elapsed(), e
            results[idx] = spo_list
            if error is not None:
                failed_ids.append(item_id)
            flag = "  [本题最终失败]" if error is not None else ""
            safe_print("[进度 %d/%d] id=%s | 本题耗时=%.2fs | 总耗时=%.2fs | 抽取三元组=%d%s"
                       % (finished, total, item_id, item_seconds,
                          total_elapsed(), len(spo_list), flag))

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
    safe_print("=" * 72)
    safe_print("全部完成！成功 %d 题，最终失败 %d 题。"
               % (total - len(failed_ids), len(failed_ids)))
    if failed_ids:
        safe_print("最终失败的题目 id：%s" % failed_ids)
    safe_print("有三元组 %d 题，空结果 %d 题。" % (non_empty, total - non_empty))
    safe_print("总耗时：%.2f 秒" % total_elapsed())
    safe_print("结果已写入：%s" % output_path)
    safe_print("=" * 72)


if __name__ == "__main__":
    main()
