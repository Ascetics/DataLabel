# -*- coding: utf-8 -*-
"""
人工智能训练师复赛 - SPO 三元组抽取与多模型仲裁 (v7.py)

使用方法：把本文件与以下两个文件放在同一目录，然后在 PyCharm 中直接右键 Run：
    人工智能训练师复赛实操题关系Schema.json
    人工智能训练师复赛实操题.jsonl
运行结束后在同目录生成 v7.jsonl。

流程：
  1) 三个抽取模型（TaoToken 的 kimi-k2.6、glm-5.3、deepseek-v4.1-flash-1017）
     共享同一套知识抽取提示词，对每道题各自独立抽取 spo_list；
  2) TaoToken 的 qwen3.7-flash 对三份 spo_list 做对比仲裁，产出最终 spo_list；
  3) 若仲裁调用失败，自动退化为多数投票合并策略，并在控制台说明。

全部为 Python 标准库实现，无需安装任何第三方包。
"""

import json
import os
import re
import time
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================================================
# 1. 配置区（按需修改）
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCHEMA_FILE = "人工智能训练师复赛实操题关系Schema.json"  # 关系 Schema 文件名
INPUT_FILE = "人工智能训练师复赛实操题.jsonl"  # 输入赛题文件名
OUTPUT_FILE = "v7.jsonl"  # 输出答案文件名

SCHEMA_PATH = os.path.join(BASE_DIR, SCHEMA_FILE)
INPUT_PATH = os.path.join(BASE_DIR, INPUT_FILE)
OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_FILE)

# 并发设置：同时处理多少道题（默认 4）
MAX_WORKERS = 4

# 单模型调用超时（秒）与失败重试次数
REQUEST_TIMEOUT = 120
MAX_RETRIES = 2

# ---------- API Key：自行替换 ----------
TAOTOKEN_API_KEY = "sk-dfgasjmkqxrzygzwtxkdmbmxihsfwzgoqvmkhijqfwtib"        # 4 个模型共用（含抽取与仲裁）

# ---------- 模型接口 ----------
# 四个模型全部走 TaoToken：OpenAI 兼容，base_url 填到 /api/v1
# 三个抽取模型共享同一把 TaoToken Key 与同一套抽取提示词
TAOTOKEN_BASE_URL = "https://taotoken.net/api/v1"
EXTRACTORS = [
    {
        "label": "TaoToken/kimi-k2.6(抽取)",
        "api_key": TAOTOKEN_API_KEY,
        "base_url": TAOTOKEN_BASE_URL,
        "model": "kimi-k2.6",
    },
    {
        "label": "TaoToken/glm-5.3(抽取)",
        "api_key": TAOTOKEN_API_KEY,
        "base_url": TAOTOKEN_BASE_URL,
        "model": "glm-5.3",
    },
    {
        "label": "TaoToken/deepseek-v4.1-flash-1017(抽取)",
        "api_key": TAOTOKEN_API_KEY,
        "base_url": TAOTOKEN_BASE_URL,
        "model": "deepseek-v4.1-flash-1017",
    },
]

# TaoToken 的 qwen3.7-flash：对三份 spo_list 做对比仲裁，共用同一把 TaoToken Key
ARBITRATOR = {
    "label": "TaoToken/qwen3.7-flash(仲裁)",
    "api_key": TAOTOKEN_API_KEY,
    "base_url": TAOTOKEN_BASE_URL,
    "model": "qwen3.7-flash",
}

# 控制台打印加锁，避免并发输出串行错乱
PRINT_LOCK = threading.Lock()


def safe_print(message):
    with PRINT_LOCK:
        print(message, flush=True)


# =====================================================================
# 2. 基础工具：文件读取、LLM 调用、JSON 解析
# =====================================================================

def load_schema(path):
    """读取全局关系 Schema，返回 [(predicate, description), ...]。"""
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    pairs = []
    for item in schema:
        predicate = item.get("predicate")
        if predicate:
            pairs.append((predicate, item.get("description", "")))
    if not pairs:
        raise ValueError("Schema 中未读取到任何 predicate，请检查文件格式。")
    return pairs


def build_predicate_text(pairs):
    """把 Schema 转成供模型使用的候选关系说明文本。"""
    return "\n".join(
        "- {0}：{1}".format(p, (d if d else "无额外说明")) for p, d in pairs
    )


def load_jsonl(path):
    """逐行读取 JSONL 赛题。"""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                safe_print("[警告] 输入文件第 {0} 行不是合法 JSON，已跳过：{1}".format(line_no, exc))
    return tasks


def call_llm(config, system_prompt, user_prompt, task_id):
    """
    调用 OpenAI 兼容的 chat/completions 接口，返回 message.content 字符串。
    失败时重试，重试耗尽则抛出异常，由调用方打印具体模型与原因。
    """
    url = config["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "stream": False,
    }
    headers = {
        "Authorization": "Bearer " + config["api_key"],
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (SPO-Extractor/7.0)",
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
                raw = response.read().decode("utf-8", errors="replace")
            result = json.loads(raw)
            content = result["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("模型返回 content 不是字符串：{0}".format(type(content)))
            return content
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            last_error = "HTTP {0} {1} {2}".format(exc.code, exc.reason, detail)
        except urllib.error.URLError as exc:
            last_error = "网络错误：{0}".format(exc.reason)
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
            last_error = "返回格式异常：{0}".format(exc)
        except Exception as exc:
            last_error = "{0}: {1}".format(type(exc).__name__, exc)

        if attempt < MAX_RETRIES:
            safe_print(
                "[重试 {0}/{1}] 模型={2} 题目id={3} 原因={4}".format(
                    attempt, MAX_RETRIES, config["label"], task_id, last_error
                )
            )
            time.sleep(1)

    raise RuntimeError("调用失败（已重试 {0} 次）：{1}".format(MAX_RETRIES, last_error))


def parse_spo_list(content):
    """从模型文本中健壮地解析出 spo_list，容忍 Markdown 代码块和前后解释。"""
    if not content:
        return []
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^\s*```(?:json)?", "", text).strip()
        text = re.sub(r"```\s*$", "", text).strip()

    candidates = [text]
    # 截取最外层 JSON 对象或数组，应对模型附加的说明文字
    for left, right in (("{", "}"), ("[", "]")):
        start = text.find(left)
        end = text.rfind(right)
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            spo = obj.get("spo_list", [])
        elif isinstance(obj, list):
            spo = obj
        else:
            continue
        if isinstance(spo, list):
            return spo
    return []


def normalize_spo_list(spo_list, predicate_set):
    """过滤非法三元组、按候选集合约束 predicate，并去重保序。"""
    cleaned = []
    seen = set()
    for item in spo_list:
        if not isinstance(item, dict):
            continue
        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")
        if not all(isinstance(x, str) and x.strip() for x in (subject, predicate, obj)):
            continue
        if predicate.strip() not in predicate_set:
            continue
        triple = (subject.strip(), predicate.strip(), obj.strip())
        if triple in seen:
            continue
        seen.add(triple)
        cleaned.append({"subject": triple[0], "predicate": triple[1], "object": triple[2]})
    return cleaned


# =====================================================================
# 3. 提示词与仲裁
# =====================================================================

# 三个抽取模型共享的同一套知识抽取提示词
EXTRACT_SYSTEM = (
    "你是一名知识抽取专员。你的唯一任务是从给定中文文本中抽取实体 SPO 三元组"
    "（主体—谓语—客体），只输出严格 JSON，不要输出解释、Markdown 或多余文字。"
)


def build_extract_user(text, predicate_text):
    return (
        "文本：\n{0}\n"
        "请抽取上述文本中所有明确表达的事实三元组，规则：\n"
        "1. 只抽文本中明确陈述的事实，不推断、不臆测、不补常识。\n"
        "2. subject 和 object 尽量使用原文中的实体表述，不改写、不翻译、不泛化。\n"
        "3. predicate 必须且只能从候选集合中选取；不属于集合的关系不要输出。\n"
        "4. 严格输出：{{\"spo_list\":[{{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\"}}]}}，不要外层解释。\n"
        "5. 无可抽取内容时输出 {{\"spo_list\":[]}}。\n"
        "6. 不要输出 id、text 或其他字段。\n"
        "候选集合（关系名称及语义说明）：\n{1}\n"
    ).format(text, predicate_text)


ARBITRATE_SYSTEM = (
    "你是一名知识抽取质检与仲裁专家。你只输出严格 JSON，不输出解释、Markdown 或多余文字。"
)


def build_arbitrate_user(text, predicate_text, candidate_blocks):
    return (
        "原始文本：\n{0}\n\n"
        "以下是一家三套抽取结果（每套结果互相独立，可能正确、遗漏或错误）：\n"
        "{1}\n"
        "请对比仲裁，确定最终的三元组列表，规则：\n"
        "1. 以原始文本为唯一事实依据：只有原文明确陈述、且符合候选关系语义说明的三元组才保留。\n"
        "2. 三套结果一致的三元组优先保留；有分歧时逐条回到原文核验，不因多数人错而保留错误结果。\n"
        "3. predicate 必须且只能取自候选集合，subject/object 必须使用原文中能唯一指代实体的表述，"
        "不得扩写或补充外部知识。\n"
        "4. 合并去重，输出规范化的完整结果。\n"
        "5. 严格输出：{{\"spo_list\":[{{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\"}}]}}；"
        "无可保留内容时输出 {{\"spo_list\":[]}}。\n\n"
        "候选集合（关系名称及语义说明）：\n{2}\n"
    ).format(text, "\n".join(candidate_blocks), predicate_text)


def fallback_merge(results):
    """仲裁模型不可用时的兜底：出现 >=2 次的三元组保留；否则取全部并集。"""
    counts = {}
    order = []
    for spo_list in results:
        for item in spo_list:
            key = (item["subject"], item["predicate"], item["object"])
            if key not in counts:
                counts[key] = 0
                order.append(key)
            counts[key] += 1
    majority = [k for k in order if counts[k] >= 2]
    chosen = majority if majority else order
    return [{"subject": k[0], "predicate": k[1], "object": k[2]} for k in chosen]


# =====================================================================
# 4. 单题求解：3 个抽取模型并发 + 1 个仲裁模型
# =====================================================================

def solve_task(task, predicate_text, predicate_set, global_start):
    task_start = time.time()
    task_id = task.get("id")
    text = task.get("text", "")
    if not isinstance(text, str):
        text = str(text)

    # 三个 TaoToken 抽取模型并发调用，共享同一套抽取提示词；单个模型失败不影响其余模型
    spo_results = []
    with ThreadPoolExecutor(max_workers=len(EXTRACTORS)) as pool:
        futures = [
            pool.submit(call_llm, cfg, EXTRACT_SYSTEM, build_extract_user(text, predicate_text), task_id)
            for cfg in EXTRACTORS
        ]
        for cfg, future in zip(EXTRACTORS, futures):
            try:
                content = future.result()
                spo = normalize_spo_list(parse_spo_list(content), predicate_set)
                spo_results.append(spo)
                safe_print("  [抽取完成] 题目id={0} 模型={1} 三元组数={2}".format(task_id, cfg["label"], len(spo)))
            except Exception as exc:
                safe_print(
                    "  [抽取失败] 题目id={0} 模型={1} 失败原因：{2}".format(task_id, cfg["label"], exc)
                )
                spo_results.append([])

    # 仲裁：TaoToken qwen3.7-flash 对比三份结果确定最终答案
    blocks = []
    for cfg, spo in zip(EXTRACTORS, spo_results):
        blocks.append(
            "{0}：\n{1}".format(cfg["label"], json.dumps({"spo_list": spo}, ensure_ascii=False))
        )

    final_spo = None
    try:
        content = call_llm(
            ARBITRATOR,
            ARBITRATE_SYSTEM,
            build_arbitrate_user(text, predicate_text, blocks),
            task_id,
        )
        final_spo = normalize_spo_list(parse_spo_list(content), predicate_set)
    except Exception as exc:
        safe_print(
            "  [仲裁失败] 题目id={0} 模型={1} 失败原因：{2}".format(
                task_id, ARBITRATOR["label"], exc
            )
        )
        safe_print("  [仲裁降级] 题目id={0} 启用多数投票合并策略".format(task_id))
        final_spo = normalize_spo_list(fallback_merge(spo_results), predicate_set)

    answer = {"id": task_id, "text": text, "spo_list": final_spo}
    elapsed = time.time() - task_start
    return answer, elapsed


# =====================================================================
# 5. 主流程：题目级并发、进度与耗时输出、结果落盘
# =====================================================================

def main():
    program_start = time.time()

    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError("找不到 Schema 文件：{0}".format(SCHEMA_PATH))
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("找不到输入文件：{0}".format(INPUT_PATH))

    pairs = load_schema(SCHEMA_PATH)
    predicate_text = build_predicate_text(pairs)
    predicate_set = set(p for p, _ in pairs)
    tasks = load_jsonl(INPUT_PATH)
    total = len(tasks)

    safe_print("=" * 72)
    safe_print("SPO 抽取启动 | 题数={0} | 并发={1} | 候选关系数={2}".format(total, MAX_WORKERS, len(pairs)))
    safe_print("抽取模型：{0}".format("、".join(c["label"] for c in EXTRACTORS)))
    safe_print("仲裁模型：{0}".format(ARBITRATOR["label"]))
    safe_print("Schema: {0}".format(SCHEMA_PATH))
    safe_print("输入  : {0}".format(INPUT_PATH))
    safe_print("输出  : {0}".format(OUTPUT_PATH))
    safe_print("=" * 72)

    answers = {}
    completed = 0
    total_spo_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {
            pool.submit(solve_task, task, predicate_text, predicate_set, program_start): task
            for task in tasks
        }
        for future in as_completed(future_map):
            task = future_map[future]
            task_id = task.get("id")
            try:
                answer, elapsed = future.result()
            except Exception as exc:
                safe_print("  [未预期错误] 题目id={0} 失败原因：{1}".format(task_id, exc))
                answer = {"id": task_id, "text": task.get("text", ""), "spo_list": []}
                elapsed = 0.0
            answers[task_id] = answer
            total_spo_count += len(answer["spo_list"])
            completed += 1
            safe_print(
                "[进度 {0}/{1}] 题目id={2} 本题耗时={3:.2f}s 累计耗时={4:.2f}s 累计三元组={5}".format(
                    completed, total, task_id, elapsed, time.time() - program_start, total_spo_count
                )
            )

    # 按输入文件中的原始顺序写出，保证结果稳定可核对
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for task in tasks:
            answer = answers.get(task.get("id"))
            if answer is None:
                continue
            f.write(json.dumps(answer, ensure_ascii=False) + "\n")

    safe_print("=" * 72)
    safe_print("全部完成！输出文件：{0}".format(OUTPUT_PATH))
    safe_print("共处理 {0} 道题，抽取三元组 {1} 条，总耗时 {2:.2f}s".format(
        total, total_spo_count, time.time() - program_start
    ))
    safe_print("=" * 72)


if __name__ == "__main__":
    main()
