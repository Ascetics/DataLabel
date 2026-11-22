# -*- coding: utf-8 -*-
"""
================================================================================
 生成式人工智能系统应用员复赛 —— Text-to-SQL 全流程程序（单文件版）
 模型：硅基流动 SiliconFlow（OpenAI 兼容 API） / Qwen/Qwen3-Coder-30B-A3B-Instruct
================================================================================

【运行前必做 —— 填 API Key】
    打开本文件顶部"一、配置区"，把 API_KEY = "请在这里填入硅基流动的API_KEY"
    引号内的占位文字替换为你的真实硅基流动 API Key，然后保存。
    （如不想改文件，也可设置环境变量 SILICONFLOW_API_KEY，程序会在占位符未填写时读取它。）

【命令行用法】（在 PyCharm 中直接 Run 本文件即可；默认读写本 .py 同目录文件）
    python text2sql_Qwen3-Coder-30B-A3B-Instruct.py
    python text2sql_Qwen3-Coder-30B-A3B-Instruct.py --fresh          # 清空历史结果+缓存+日志后全流程重跑
    python text2sql_Qwen3-Coder-30B-A3B-Instruct.py --limit 10       # 只跑前 10 题，便于调试
    python text2sql_Qwen3-Coder-30B-A3B-Instruct.py \
        --input  "生成式人工智能系统应用员复赛实操题.jsonl" \
        --db     "生成式人工智能系统应用员复赛实操题数据库.db" \
        --output "result-Qwen3-Coder-30B-A3B-Instruct.jsonl" --fresh

【默认文件名】（均在本 .py 同目录，可用命令行参数覆盖）
    输入：生成式人工智能系统应用员复赛实操题.jsonl
    数据库：生成式人工智能系统应用员复赛实操题数据库.db
    输出：result-Qwen3-Coder-30B-A3B-Instruct.jsonl

【产出文件】（均在本 .py 同目录）
    result-Qwen3-Coder-30B-A3B-Instruct.jsonl   最终提交结果，每行 {"id","question","sql","result"}
    text2sql_cache.json                          断点续跑缓存（题目 id -> sql/result）
    text2sql_call_log.jsonl                      模型原始调用日志（请求、响应、工具回传、耗时）

【设计要点】
    * 所有语义理解与 SQL 生成均由大模型完成；程序不含任何按 id/顺序/预设答案的映射。
    * 通过 OpenAI 兼容 function-calling 让模型自主调用 execute_sql 工具做自校验，
      工具真实执行结果以 tool 消息回传给模型；模型可据此多轮修正。
    * 最终写入输出文件的 SQL 严格等于模型最后返回的 SQL；程序只负责在本地库上执行
      该 SQL 得到 result，绝不改写模型结论。
    * temperature=0 + 固定 seed，相同输入尽量得到相同输出；缓存支持断点续跑。
    * 仅使用 Python 标准库，无需 pip install，PyCharm 可直接运行。
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# 一、配置区（模型名 / base_url / API Key 全部在这里）
# ============================================================================
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
BASE_URL = "https://api.siliconflow.cn/v1"          # 硅基流动 OpenAI 兼容端点
API_KEY = "sk-"            # 【必改】把占位文字替换成真实 API Key；也可留空并设环境变量

# 若顶部未填写（仍是占位文字或留空），则回退读取环境变量 SILICONFLOW_API_KEY
_PLACEHOLDERS = ("请在这里填入", "填入硅基流动", "your_api_key", "sk-xxx")
if (not API_KEY.strip()) or any(p in API_KEY for p in _PLACEHOLDERS):
    API_KEY = os.environ.get("SILICONFLOW_API_KEY", "").strip()

# 默认文件名（与本 .py 同目录；命令行 --input/--db/--output 可覆盖）
INPUT_NAME = "生成式人工智能系统应用员复赛实操题.jsonl"          # 输入题目集
DB_NAME = "生成式人工智能系统应用员复赛实操题数据库.db"          # SQLite 数据库
OUTPUT_NAME = "result-Qwen3-Coder-30B-A3B-Instruct.jsonl"       # 最终提交结果
CACHE_NAME = "text2sql_cache.json"                              # 断点续跑缓存
LOG_NAME = "text2sql_call_log.jsonl"                            # 模型原始调用日志

ENABLE_TOOL_CALLING = True        # 是否向模型开放 execute_sql 自校验工具
MAX_TOOL_TURNS = 8                # 单题最多模型交互轮数（最后一轮强制不给工具，逼出最终 SQL）
MAX_API_RETRIES = 3               # 单次 HTTP 请求失败重试次数
API_TIMEOUT = 180                 # 单次请求超时（秒）
REQUEST_TEMPERATURE = 0.2           # 贪心解码，保证稳定
REQUEST_SEED = 0                  # 固定随机种子（配合 temperature=0，尽量保证结果可复现）
TOOL_RESULT_MAX_ROWS = 50         # 回传给模型的工具结果最多行数（最终结果不截断）

# ============================================================================
# 二、数据库结构（来自赛事提供的《数据库结构.md》，作为模型提示词上下文）
# ============================================================================
SCHEMA_TEXT = r"""
## regions — 销售区域
region_id INTEGER PK 区域ID; region_name TEXT 区域名称
## stores — 线下门店
store_id INTEGER PK 门店ID; store_name TEXT 门店名称; region_id INTEGER FK->regions.region_id 所属区域ID; city TEXT 所在城市; district TEXT 所在区县; store_type TEXT 门店类型; open_date TEXT 开业日期
## warehouses — 仓库信息
warehouse_id INTEGER PK 仓库ID; warehouse_name TEXT 仓库名称; region_id INTEGER FK->regions.region_id 所属区域ID; city TEXT 所在城市
## categories — 商品类别
category_id INTEGER PK 类别ID; category_name TEXT 类别名称; parent_group TEXT 上级业务分组
## brands — 商品品牌
brand_id INTEGER PK 品牌ID; brand_name TEXT 品牌名称; origin_country TEXT 品牌来源国家; brand_tier TEXT 品牌定位层级
## suppliers — 商品供应商
supplier_id INTEGER PK 供应商ID; supplier_name TEXT 供应商名称; city TEXT 所在城市; rating REAL 合作评级; cooperation_date TEXT 开始合作日期
## products — 商品主数据
product_id INTEGER PK 商品ID; product_name TEXT 商品名称; category_id INTEGER FK->categories.category_id; brand_id INTEGER FK->brands.brand_id; supplier_id INTEGER FK->suppliers.supplier_id; list_price REAL 当前标价; cost_price REAL 当前成本价; product_status TEXT 商品状态; launch_date TEXT 上市日期
## product_price_history — 商品历史价格
price_history_id INTEGER PK; product_id INTEGER FK->products.product_id; effective_from TEXT 价格生效开始日期; effective_to TEXT 价格生效结束日期（空值表示当前仍有效）; price REAL 该有效期内价格
## customer_levels — 会员等级定义
level_code TEXT PK 会员等级代码; level_name TEXT 会员等级名称; min_points INTEGER 该等级最低积分; max_points INTEGER 该等级最高积分（空值表示无上限）
## customers — 客户信息
customer_id INTEGER PK; customer_name TEXT 客户姓名; city TEXT 所在城市; level_code TEXT FK->customer_levels.level_code; points INTEGER 当前会员积分; register_date TEXT 注册日期; birth_year INTEGER 出生年份; referrer_id INTEGER FK->customers.customer_id 推荐人客户ID（可为空）
## employees — 门店员工
employee_id INTEGER PK; employee_name TEXT 员工姓名; store_id INTEGER FK->stores.store_id; department TEXT 部门; hire_date TEXT 入职日期; salary REAL 月薪; manager_id INTEGER FK->employees.employee_id 直属经理员工ID（可为空）
## inventory — 仓库商品当前库存（联合主键 warehouse_id+product_id）
warehouse_id INTEGER FK->warehouses.warehouse_id; product_id INTEGER FK->products.product_id; stock_qty INTEGER 当前库存数量; safety_stock INTEGER 安全库存数量; last_update TEXT 库存更新时间
## promotions — 促销活动
promotion_id INTEGER PK; promotion_name TEXT 促销名称; promo_type TEXT 促销类型; start_date TEXT 活动开始日期; end_date TEXT 活动结束日期; discount_rate REAL 折扣率（部分为空）; threshold_amount REAL 满减门槛金额（部分为空）; reduction_amount REAL 满减优惠金额（部分为空）
## promotion_products — 促销适用商品（联合主键 promotion_id+product_id）
promotion_id INTEGER FK->promotions.promotion_id; product_id INTEGER FK->products.product_id
## orders — 订单主表
order_id INTEGER PK; customer_id INTEGER FK->customers.customer_id; store_id INTEGER FK->stores.store_id; order_date TEXT 下单日期; order_status TEXT 订单状态; channel TEXT 下单渠道; gross_amount REAL 订单商品原价合计; order_discount_amount REAL 订单级优惠金额; shipping_fee REAL 运费; payable_amount REAL 订单应付金额
## order_items — 订单明细
order_item_id INTEGER PK; order_id INTEGER FK->orders.order_id; product_id INTEGER FK->products.product_id; quantity INTEGER 购买数量; unit_price REAL 下单时商品单价; discount_rate REAL 该明细折扣率; promotion_id INTEGER FK->promotions.promotion_id（可为空）
## payments — 订单支付记录
payment_id INTEGER PK; order_id INTEGER FK->orders.order_id; payment_method TEXT 支付方式; paid_amount REAL 该笔支付金额; paid_at TEXT 支付时间; payment_status TEXT 支付状态
## shipping_records — 订单物流记录
shipping_id INTEGER PK; order_id INTEGER FK->orders.order_id; carrier TEXT 承运商; shipping_status TEXT 物流状态; shipped_date TEXT 发货日期（可为空）; delivered_date TEXT 签收日期（可为空）; destination_city TEXT 收货城市
## returns — 订单明细退货退款记录
return_id INTEGER PK; order_item_id INTEGER FK->order_items.order_item_id; return_qty INTEGER 退货数量; return_date TEXT 退货申请日期; return_status TEXT 退款状态; refund_amount REAL 退款金额; reason TEXT 退货原因
## reviews — 商品评价
review_id INTEGER PK; customer_id INTEGER FK->customers.customer_id; product_id INTEGER FK->products.product_id; rating INTEGER 评分; review_date TEXT 评价日期; content TEXT 评价文字内容（可为空）
## store_targets — 门店月度经营目标（联合主键 store_id+target_month）
store_id INTEGER FK->stores.store_id; target_month TEXT 目标月份 YYYY-MM; sales_target REAL 月度销售额目标; order_target INTEGER 月度订单数目标

【日期字段都是 TEXT】一般日期 YYYY-MM-DD；payments.paid_at 为 YYYY-MM-DD HH:MM:SS；store_targets.target_month 为 YYYY-MM。
"""

# ============================================================================
# 三、给模型的工具定义（execute_sql 自校验）
# ============================================================================
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "在本地 SQLite 数据库上执行一条只读 SELECT / WITH 查询，返回真实的表头与数据行（最多 50 行）。"
                "用于校验你生成的 SQL 语法是否正确、枚举值与字段是否真实存在、结果是否符合问题语义。"
                "工具返回的是真实执行结果，请据此修正你的 SQL。最终回复仍须只给 JSON 对象。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "要执行的单条只读 SQLite SELECT 语句（禁止写操作、禁止多语句）",
                    }
                },
                "required": ["sql"],
            },
        },
    }
]


# ============================================================================
# 四、通用工具函数
# ============================================================================
def now_str() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_log(log_path: Path, record: Dict[str, Any]) -> None:
    """追加一条原始调用日志（JSONL），用于审计模型的真实调用过程。"""
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # 日志失败不应中断主流程
        print(f"[warn] 写日志失败: {exc}", file=sys.stderr)


def json_safe(value: Any) -> Any:
    """把 SQLite 返回值转成可 JSON 序列化的形式。"""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def read_jsonl(path: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj or "question" not in obj:
                raise ValueError(f"{path} 第 {lineno} 行缺少 id 或 question 字段")
            items.append({"id": str(obj["id"]), "question": str(obj["question"])})
    return items


def discover_input(script_dir: Path) -> Path:
    """自动发现输入题目 JSONL：优先名字含“实操题/question/input”的文件。"""
    cands = [
        p for p in script_dir.glob("*.jsonl")
        if p.name not in (OUTPUT_NAME,) and "提交示例" not in p.name and "示例" not in p.name
    ]
    for kw in ("实操题", "question", "input", "问题"):
        hit = [p for p in cands if kw in p.name]
        if hit:
            return sorted(hit)[0]
    if cands:
        return sorted(cands)[0]
    raise FileNotFoundError(f"在 {script_dir} 下未找到输入 JSONL，请用 --input 指定")


def discover_db(script_dir: Path) -> Path:
    """自动发现 SQLite 数据库文件。"""
    cands = sorted(script_dir.glob("*.db")) + sorted(script_dir.glob("*.sqlite*"))
    for kw in ("数据库", "database", "db"):
        hit = [p for p in cands if kw in p.name]
        if hit:
            return hit[0]
    if cands:
        return cands[0]
    raise FileNotFoundError(f"在 {script_dir} 下未找到 .db 数据库，请用 --db 指定")


def collect_enum_hints(db_path: Path) -> str:
    """
    从本地数据库真实数据中抽取低基数字段的取值，作为提示词上下文，
    帮助模型原样选用中文枚举值（如 '支付成功'、'金卡会员'、GOLD）。
    注意：这里只提供“字段有哪些可能取值”，不包含任何题目的答案或业务计算公式。
    """
    lines: List[str] = []
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        tables = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )]
        for table in tables:
            cols = list(cur.execute(f'PRAGMA table_info("{table}")'))
            for col in cols:
                name, decl_type = col[1], (col[2] or "").upper()
                low = name.lower()
                if "TEXT" not in decl_type:
                    continue
                if low.endswith("_id") or low.startswith("id_"):
                    continue
                # 日期类字段由规则处理，不必占用枚举上下文
                if any(k in low for k in ("date", "month", "paid_at", "effective_")):
                    continue
                try:
                    vals = [
                        str(r[0]) for r in cur.execute(
                            f'SELECT DISTINCT "{name}" FROM "{table}" '
                            f'WHERE "{name}" IS NOT NULL ORDER BY 1 LIMIT 21'
                        )
                    ]
                except Exception:
                    continue
                if 0 < len(vals) <= 20:
                    joined = " / ".join(vals)
                    if len(joined) > 400:
                        joined = joined[:400] + " ..."
                    lines.append(f"- {table}.{name}: {joined}")
        con.close()
    except Exception as exc:
        print(f"[warn] 抽取枚举提示失败（不影响主流程）: {exc}", file=sys.stderr)
    return "\n".join(lines)


# ============================================================================
# 五、系统提示词（SQL 生成规则 + 数据库结构 + 真实枚举上下文）
# ============================================================================
PROMPT_TEMPLATE = r"""你是一名资深数据分析师与 SQLite SQL 专家。你的任务是：把用户的自然语言问题转换成【唯一一条】可执行的 SQLite SELECT 查询语句。

【输出格式 — 严格】
只返回一个 JSON 对象，格式为 {"sql": "SELECT ..."}。不要输出解释、注释、markdown 代码块或任何多余字段。

【可用工具 execute_sql】
你可以（可选）调用 execute_sql(sql) 在本地 SQLite 数据库上执行一条只读 SELECT，工具会返回真实表头与数据。请用它检查语法、字段名、枚举值、结果语义；工具结果会真实回传给你。最终回复仍必须只给 {"sql": "..."}。不要调用工具执行写操作。

【数据库结构】
@@SCHEMA@@

【从本地数据库真实数据采样得到的低基数字段取值（仅用于原样选取枚举值，不代表任何题目的答案）】
@@ENUM@@

【SQL 硬性要求】
1. 只能生成一条只读 SELECT（WITH ... SELECT 形式的单条查询允许）。禁止多语句、分号拼接、INSERT/UPDATE/DELETE/DROP 等写操作。
2. 使用 SQLite 语法；若出现中文表名或字段名须用双引号包裹；不要使用 MySQL/PostgreSQL 专有函数。本库表名字段名均为英文，原样使用，不要臆造。
3. 【严格按题目指定的返回字段】SELECT 出的列名、顺序、别名必须与题目要求完全一致（题目"返回 X、Y"就只 SELECT X、Y）。不要多列、不要少列、不要改名、不要用 SELECT *。唯一例外：题目明确要求命名的别名（如"结果字段命名为 customer_count"）必须照办。
4. 【严格按题目指定的排序】题目说"按 X 升序/降序、Y 升序"就完全照做，ORDER BY 的字段与方向一字不差。题目用"并列全部保留"时，不要额外加 LIMIT 1 或擅自去重。
5. 多表关联必须用 JOIN ... ON 明确写出关联条件；注意一对多 JOIN 后要用 DISTINCT 或聚合避免行重复放大。
6. 不确定的中文枚举值必须原样选取【数据库结构】及上方"实际取值"中列出的值，不要自行翻译或猜测（例如支付状态写 '支付成功' 而不是 'success'；金卡会员用 customers.level_code='GOLD' 或 join customer_levels 取 level_name='金卡会员'；区域用 '华东'；订单状态用 '已完成' 等）。
7. 【日期字段都是 TEXT】一般日期格式 YYYY-MM-DD，payments.paid_at 为 YYYY-MM-DD HH:MM:SS，store_targets.target_month 为 YYYY-MM。因此：
   - 年份/日期范围用字符串比较，如 order_date >= '2025-01-01' AND order_date < '2026-01-01'；
   - 按月统计用 substr(order_date, 1, 7) 或 strftime('%Y-%m', order_date)；按年用 substr(order_date,1,4)；
   - 天数差用 julianday(delivered_date) - julianday(shipped_date)。
8. 【NULL 用 IS NULL / IS NOT NULL】effective_to、content、shipped_date、delivered_date、manager_id、referrer_id 等字段的空值是 SQL NULL（不是字符串）。"当前有效"= effective_to IS NULL；"没有文字评价"= content IS NULL。
9. 复杂逻辑优先用 CTE（WITH）分步拆解：先过滤、再聚合、最后筛选/排名，保证可读与正确。
10. 聚合后筛选必须用 HAVING，不要用 WHERE 过滤聚合结果。ROUND(x, 2) 保留小数。
11. NULL 值不要用等号比较。DISTINCT/去重、HAVING、窗口函数（ROW_NUMBER/RANK/DENSE_RANK）按需使用。
    【重要】SQLite 不能在 WHERE 中直接引用 SELECT 里定义的窗口函数别名（会报 "misuse of aliased window function"）。需筛选排名时，把窗口函数放在内层子查询/CTE，外层再 WHERE ranking<=N。
12. 绝不编造字段。
"""


def build_system_prompt(enum_hints: str) -> str:
    return (PROMPT_TEMPLATE
            .replace("@@SCHEMA@@", SCHEMA_TEXT.strip())
            .replace("@@ENUM@@", enum_hints.strip() if enum_hints.strip() else "（无）"))


# ============================================================================
# 六、SQLite 只读执行器（工具自校验 & 最终结果都走这里）
# ============================================================================
def _remove_semicolons_outside_strings(sql: str) -> str:
    """去掉字符串字面量之外的分号，用于检测“多语句”。"""
    out: List[str] = []
    quote: Optional[str] = None
    for ch in sql:
        if quote is not None:
            out.append(ch)
            if ch == quote:
                quote = None
        else:
            if ch in ("'", '"', "`"):
                quote = ch
                out.append(ch)
            elif ch == ";":
                continue
            else:
                out.append(ch)
    return "".join(out).strip()


def _semicolon_outside_strings(sql: str) -> bool:
    """判断是否存在字符串字面量之外的分号（即多语句分隔）。"""
    quote: Optional[str] = None
    for ch in sql:
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch == ";":
            return True
    return False


def execute_sql_readonly(
    db_path: Path, sql: str, max_rows: Optional[int] = None
) -> Dict[str, Any]:
    """
    在 SQLite 上执行一条只读查询，返回 {'ok', 'columns', 'rows', 'row_count', ...}。
    max_rows=None 表示取全部行（用于最终输出）；指定时用于工具回传，避免太长。
    """
    result: Dict[str, Any] = {
        "ok": False, "columns": [], "rows": [], "row_count": 0,
        "truncated": False, "error": None,
    }
    if not isinstance(sql, str) or not sql.strip():
        result["error"] = "SQL 为空"
        return result

    if _semicolon_outside_strings(sql):
        result["error"] = "禁止多语句（分号拼接）"
        return result

    cleaned = _remove_semicolons_outside_strings(sql)
    first = re.split(r"\s+", cleaned, 1)[0].upper() if cleaned else ""
    if first not in ("SELECT", "WITH"):
        result["error"] = "只允许只读 SELECT / WITH 查询"
        return result
    if re.search(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
        r"PRAGMA|VACUUM|TRIGGER|BEGIN|COMMIT|ROLLBACK|EXPLAIN)\b",
        cleaned, re.IGNORECASE,
    ):
        result["error"] = "检测到写操作或禁止的关键字"
        return result

    con: Optional[sqlite3.Connection] = None
    try:
        con = sqlite3.connect(str(db_path))
        con.execute("PRAGMA query_only = ON")
        cur = con.execute(cleaned)
        cols = [d[0] for d in cur.description] if cur.description else []
        if max_rows is None:
            raw_rows = cur.fetchall()
            truncated = False
        else:
            raw_rows = cur.fetchmany(max_rows)
            truncated = cur.fetchone() is not None
        rows = [
            {cols[i]: json_safe(v) for i, v in enumerate(row)}
            for row in raw_rows
        ]
        result.update({
            "ok": True, "columns": cols, "rows": rows,
            "row_count": len(rows), "truncated": truncated, "error": None,
        })
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if con is not None:
            con.close()
    return result


# ============================================================================
# 七、模型调用（OpenAI 兼容 /chat/completions，含原始请求-响应日志）
# ============================================================================
def post_chat(payload: Dict[str, Any], log_path: Path, qid: str) -> Dict[str, Any]:
    url = BASE_URL.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_API_RETRIES + 1):
        t0 = time.time()
        append_log(log_path, {
            "ts": now_str(), "id": qid, "type": "model_request_raw",
            "attempt": attempt, "url": url, "model": payload.get("model"),
            "payload": payload,
        })
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                raw_text = resp.read().decode("utf-8", "replace")
            data = json.loads(raw_text)
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(f"API error: {data['error']}")
            append_log(log_path, {
                "ts": now_str(), "id": qid, "type": "model_response_raw",
                "attempt": attempt, "elapsed_ms": int((time.time() - t0) * 1000),
                "raw": data,
            })
            return data
        except Exception as exc:
            last_err = exc
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read().decode("utf-8", "replace")[:1500]
                except Exception:
                    pass
            append_log(log_path, {
                "ts": now_str(), "id": qid, "type": "model_error",
                "attempt": attempt, "error": str(exc), "detail": detail,
            })
            if attempt < MAX_API_RETRIES:
                time.sleep(2 ** (attempt - 1))  # 1s, 2s 退避

    raise RuntimeError(f"模型调用失败（重试 {MAX_API_RETRIES} 次）: {last_err}")


# ============================================================================
# 八、解析模型最终输出 -> SQL
# ============================================================================
def parse_final_sql(content: Optional[str]) -> Optional[str]:
    """从模型文本中稳健解析 {"sql": "..."}，容忍 markdown 代码块和少量前后文字。"""
    if not content:
        return None
    text = content.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    def try_obj(obj: Any) -> Optional[str]:
        if isinstance(obj, dict) and isinstance(obj.get("sql"), str) and obj["sql"].strip():
            return obj["sql"].strip()
        return None

    try:
        found = try_obj(json.loads(text))
        if found:
            return found
    except Exception:
        pass

    # 在文本里扫描平衡的 {...}，逐个尝试解析
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        end = -1
        for i in range(start, len(text)):
            ch = text[i]
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
                        end = i
                        break
        if end != -1:
            try:
                found = try_obj(json.loads(text[start:end + 1]))
                if found:
                    return found
            except Exception:
                pass
        start = text.find("{", start + 1)
    return None


# ============================================================================
# 九、单题全流程：模型生成 + 工具自校验 + 失败回喂修正
# ============================================================================
def generate_one(
    qid: str,
    question: str,
    db_path: Path,
    system_prompt: str,
    log_path: Path,
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            f"自然语言问题：{question}\n"
            "请先理解问题语义并生成 SQL；可调用 execute_sql 工具做真实校验；"
            '最终严格只返回 {"sql": "SELECT ..."}。'
        )},
    ]

    last_parsed: Optional[str] = None
    last_good: Optional[str] = None
    last_error: Optional[str] = None
    tools_available = ENABLE_TOOL_CALLING   # 若端点不支持 function-calling，会自动降级

    for turn in range(1, MAX_TOOL_TURNS + 1):
        payload: Dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "temperature": REQUEST_TEMPERATURE,
            "seed": REQUEST_SEED,
        }
        # 最后一轮收回工具，强制模型给出最终 JSON 结论
        if tools_available and turn < MAX_TOOL_TURNS:
            payload["tools"] = TOOLS
            payload["tool_choice"] = "auto"

        try:
            data = post_chat(payload, log_path, qid)
        except RuntimeError as exc:
            # 兼容处理：若平台/模型不支持工具调用，去掉工具参数重试，仍由模型直接生成 SQL
            if "tools" in payload and tools_available:
                append_log(log_path, {
                    "ts": now_str(), "id": qid, "type": "tools_unsupported_fallback",
                    "error": str(exc),
                })
                payload.pop("tools", None)
                payload.pop("tool_choice", None)
                tools_available = False
                data = post_chat(payload, log_path, qid)
            else:
                raise
        choices = data.get("choices") or []
        msg = (choices[0].get("message") if choices else {}) or {}
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        append_log(log_path, {
            "ts": now_str(), "id": qid, "type": "assistant_turn",
            "turn": turn, "content": content[:4000],
            "tool_calls": [
                {
                    "id": tc.get("id"),
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": (tc.get("function") or {}).get("arguments"),
                }
                for tc in tool_calls
            ],
            "usage": data.get("usage"),
        })

        # 1) 先把模型这一轮消息原样放进上下文（function-calling 协议要求）
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # 2) 真实执行模型自主发起的工具调用，并把结果回传给模型
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name")
            if name == "execute_sql":
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                sql_arg = args.get("sql", "")
                tool_res = execute_sql_readonly(db_path, sql_arg, max_rows=TOOL_RESULT_MAX_ROWS)
            else:
                tool_res = {"ok": False, "error": f"未知工具: {name}"}
            append_log(log_path, {
                "ts": now_str(), "id": qid, "type": "tool_result_to_model",
                "turn": turn, "tool_call_id": tc.get("id"),
                "name": name, "result": tool_res,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": json.dumps(tool_res, ensure_ascii=False),
            })

        # 3) 检查这一轮是否已经给出最终 SQL
        parsed = parse_final_sql(content)
        if parsed:
            last_parsed = parsed
            check = execute_sql_readonly(db_path, parsed, max_rows=None)
            if check["ok"]:
                last_good = parsed
                append_log(log_path, {
                    "ts": now_str(), "id": qid, "type": "final_accepted",
                    "turn": turn, "sql": parsed, "row_count": check["row_count"],
                })
                # 【关键】sql 原样采用模型输出；result 只是本地执行得到的客观结果
                return {"sql": parsed, "result": check["rows"], "ok": True, "error": None}
            last_error = check["error"]
            messages.append({
                "role": "user",
                "content": (
                    f"你给出的 SQL 在本地校验执行失败：{last_error}\n"
                    '请修正后严格只返回 {"sql": "SELECT ..."}；可继续调用 execute_sql 工具。'
                ),
            })
            continue

        # 4) 没有最终 SQL：如果有工具结果就提示模型据此收尾，否则提示输出格式
        if tool_calls:
            messages.append({
                "role": "user",
                "content": (
                    "已把 execute_sql 的真实执行结果回传给你。请结合结果确认 SQL 正确，"
                    '然后严格只返回 {"sql": "SELECT ..."}。'
                ),
            })
        else:
            messages.append({
                "role": "user",
                "content": '输出不符合要求：必须严格只返回一个 JSON 对象 {"sql": "SELECT ..."}。',
            })

    # 轮数用尽：优先采用曾经执行成功过的模型 SQL；否则采用最后一次模型 SQL
    if last_good:
        check = execute_sql_readonly(db_path, last_good, max_rows=None)
        return {"sql": last_good, "result": check["rows"], "ok": True, "error": None}
    if last_parsed:
        check = execute_sql_readonly(db_path, last_parsed, max_rows=None)
        append_log(log_path, {
            "ts": now_str(), "id": qid, "type": "final_fallback",
            "sql": last_parsed, "error": last_error,
        })
        return {
            "sql": last_parsed,
            "result": check["rows"] if check["ok"] else [],
            "ok": False,
            "error": last_error,
        }
    raise RuntimeError(f"未能从模型获得任何 SQL（最后错误：{last_error}）")


# ============================================================================
# 十、结果 / 缓存 落盘
# ============================================================================
def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def write_output(path: Path, items: List[Dict[str, str]], cache: Dict[str, Any]) -> None:
    """按输入顺序重建输出文件；id/question 直接取自输入，保证一字不差。"""
    lines: List[str] = []
    for item in items:
        rec = cache.get(item["id"])
        if not rec:
            continue
        out_obj = {
            "id": item["id"],
            "question": item["question"],
            "sql": rec.get("sql", ""),
            "result": rec.get("result", []),
        }
        lines.append(json.dumps(out_obj, ensure_ascii=False))
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        print(f"[warn] 缓存读取失败，将忽略: {exc}", file=sys.stderr)
        return {}


def save_cache(path: Path, cache: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(cache, ensure_ascii=False, indent=1))


# ============================================================================
# 十一、主流程
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="硅基流动 Qwen3-Coder-30B Text-to-SQL 复赛实操程序（单文件）"
    )
    parser.add_argument("--input", default="", help=f"输入题目 JSONL 路径（默认同目录 {INPUT_NAME}）")
    parser.add_argument("--db", default="", help=f"SQLite 数据库路径（默认同目录 {DB_NAME}）")
    parser.add_argument("--output", default="", help=f"输出 JSONL 路径（默认同目录 {OUTPUT_NAME}）")
    parser.add_argument("--fresh", action="store_true",
                        help="清空历史结果、缓存与调用日志后，全流程重新运行")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 题（0=全部），便于调试")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    # 优先用命令行参数；否则用配置区固定文件名；文件不存在时再自动发现同目录文件
    if args.input:
        input_path = Path(args.input).resolve()
    else:
        input_path = (script_dir / INPUT_NAME).resolve()
        if not input_path.exists():
            input_path = discover_input(script_dir)
    if args.db:
        db_path = Path(args.db).resolve()
    else:
        db_path = (script_dir / DB_NAME).resolve()
        if not db_path.exists():
            db_path = discover_db(script_dir)
    output_path = Path(args.output).resolve() if args.output else (script_dir / OUTPUT_NAME).resolve()
    cache_path = output_path.parent / CACHE_NAME
    log_path = output_path.parent / LOG_NAME

    for label, p in (("输入文件", input_path), ("数据库", db_path)):
        if not p.exists():
            raise FileNotFoundError(f"{label}不存在: {p}")
        print(f"[info] {label}: {p}")
    if not API_KEY:
        raise RuntimeError(
            "未配置 API Key：请在本文件顶部配置区填写 API_KEY，"
            "或设置环境变量 SILICONFLOW_API_KEY。"
        )

    items = read_jsonl(input_path)
    if args.limit and args.limit > 0:
        items = items[:args.limit]
    print(f"[info] 共 {len(items)} 道题；模型={MODEL}；base_url={BASE_URL}")

    # --fresh：清空历史结果 + 缓存 + 原始日志，然后完整重跑
    if args.fresh:
        for p in (output_path, cache_path, log_path):
            if p.exists():
                p.unlink()
                print(f"[info] 已清空: {p.name}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {} if args.fresh else load_cache(cache_path)
    enum_hints = collect_enum_hints(db_path)
    system_prompt = build_system_prompt(enum_hints)
    print(f"[info] 提示词已就绪（含 {len(SCHEMA_TEXT)} 字符结构 + "
          f"{len(enum_hints.splitlines())} 条真实枚举线索）")

    ok_cnt = fail_cnt = cache_cnt = 0
    for idx, item in enumerate(items, 1):
        qid, question = item["id"], item["question"]
        cached = cache.get(qid)
        if (not args.fresh) and cached and cached.get("question") == question \
                and isinstance(cached.get("result"), list):
            print(f"[{idx}/{len(items)}] id={qid} 命中缓存，跳过模型调用")
            cache_cnt += 1
            continue

        print(f"[{idx}/{len(items)}] id={qid} 调用模型 -> {question[:40]}...")
        try:
            out = generate_one(qid, question, db_path, system_prompt, log_path)
            if out.get("ok"):
                ok_cnt += 1
            else:
                fail_cnt += 1
                print(f"  [warn] 该题模型输出未通过最终校验: {out.get('error')}")
        except Exception as exc:
            fail_cnt += 1
            print(f"  [error] 该题处理失败: {exc}", file=sys.stderr)
            append_log(log_path, {
                "ts": now_str(), "id": qid, "type": "question_failed", "error": str(exc),
            })
            out = {"sql": "", "result": [], "ok": False}

        cache[qid] = {
            "id": qid,
            "question": question,
            "sql": out.get("sql", ""),
            "result": out.get("result", []),
        }
        write_output(output_path, items, cache)   # 逐题落盘，中断也不丢进度
        save_cache(cache_path, cache)

    print(f"[done] 输出: {output_path}")
    print(f"[done] 统计: 成功={ok_cnt} 失败/降级={fail_cnt} 缓存命中={cache_cnt} 总计={len(items)}")
    print(f"[done] 原始调用日志: {log_path}")


if __name__ == "__main__":
    main()
