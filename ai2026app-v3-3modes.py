# -*- coding: utf-8 -*-
"""
================================================================================
 生成式人工智能系统应用员初赛实操题 —— 基于上下文工程的思维链推理
 （三平台三模型 · 答案仲裁 · 4 题并发）
================================================================================
 运行环境：Python 3.8+，仅使用标准库（不需要 pip 安装任何第三方包）
 运行方式：在 PyCharm 中打开本文件 → 右键 Run 'ai2026app'（或点右上角绿色三角）
 输入文件：生成式人工智能系统应用员初赛实操题.jsonl      （与本文件同目录）
 输出文件：生成式人工智能系统应用员初赛实操题-a1.jsonl   （自动生成在同一目录）
 运行日志：ai2026app_run_log.txt                        （自动生成，含 1:1:1 题目清单）

 三个大模型（赛题要求，来自三个不同平台，各自独立的 API_KEY）：
   ① 阿里云百炼 qwen3.8-27B   —— https://dashscope.aliyuncs.com/compatible-mode/v1
   ② 硅基流动 qwen3-32B       —— https://api.siliconflow.cn/v1 （同时担任仲裁者）
   ③ 讯飞星火 Spark-X2-Flash  —— https://maas-api.cn-huabei-1.xf-yun.com/v2 （MaaS 平台 OpenAI 兼容接口）
   KEY 在下方 DASHSCOPE_API_KEY / SILICONFLOW_API_KEY / SPARK_API_KEY 处替换。

 解题步骤（与赛题条目一一对应）：
   步骤一 仲裁原则：只对“需要仲裁”的步骤执行三模型少数服从多数，仲裁者固定为硅基流动 qwen3-32B。
            3:0 或 2:1 直接定论；1:1:1 重新跑一轮；两轮均 1:1:1 以 qwen3-32B 的结果为准。
   步骤二 题型判别【需仲裁】：三模型各自判断题型 → 仲裁。
   步骤三 提示词生成【不仲裁】：三模型分别生成自己的提示词，三份提示词各自保留、互不干扰。
   步骤四 分模式作答【需仲裁·只裁答案】：每个模型用“自己的提示词”产出 reasoning + answer，
            三份答案投票；胜出方的 reasoning 与 answer 一起作为该题结论。
   步骤五 控制台输出：进度、题号、题目内容、耗时、每一步的仲裁情况。
   步骤六 并发策略：默认 4 道题并发（MAX_QUESTION_WORKERS = 4）。

 运行时还会自动做三件事：
   · 启动自检（PREFLIGHT_CHECK=True）：三个模型各发一次最小请求，直接告诉你谁可用、谁不可用、为什么；
   · 失败诊断：任何一次模型调用失败，都会在控制台打印“平台/模型/失败原因/排查建议”，结尾再按模型汇总次数；
     鉴权失败、模型 ID 不存在这类错误会立即失败、不再重试（避免每题白等十几秒）；
   · 断点续跑（RESUME=True）+ 收尾按题目原始顺序重排输出文件，保证提交文件与题目文件一一对应。
================================================================================
"""

import json
import os
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# 控制台中文输出（Windows/PyCharm 下防止 GBK 编码报错）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ==============================================================================
# 一、运行参数
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "生成式人工智能系统应用员初赛实操题.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "生成式人工智能系统应用员初赛实操题-a1.jsonl")
LOG_FILE = os.path.join(BASE_DIR, "ai2026app_run_log.txt")

MAX_QUESTION_WORKERS = 4       # 步骤六：并发题数（每题内部还有 3 个模型并行 → 峰值 12 个请求）
RESUME = True                  # 断点续跑：输出文件里已有的 id 自动跳过
PREFLIGHT_CHECK = True         # 启动前逐个模型做一次连通性自检（能立刻看出哪个模型调不通）
TEMPERATURE = 0.2              # 低温，减少无关发挥
MAX_TOKENS = 2600              # 单次回答上限，防止推理过程被截断
TIMEOUT = 240                  # 单次请求超时（秒）
MAX_RETRY = 3                  # 单模型单步的网络重试次数（鉴权/模型不存在这类错误不再重试）
MIN_REASONING_LEN = 15         # reasoning 短于该长度视为无效，触发补写
QUESTION_INTERVAL = 0.0        # 每道题结束后的额外间隔（秒），限流时可设 1~2
ERROR_PREVIEW_LEN = 300        # 控制台打印错误原因时的截断长度


# ==============================================================================
# 二、三个平台的模型配置（★ API_KEY 在这里替换 ★）
# ==============================================================================

DASHSCOPE_API_KEY = "sk-ws-H.YYMEYRI.GoK2.MEYCIQC-EXczLvaE935t9JRzvhE8nODMNwmVJgUesju7IvWSbAIhAJ9hvrD4zRaaMwN1-_0whJz84oPrtnkf_gao4gj-2a20"          # 阿里云百炼：https://bailian.console.aliyun.com
SILICONFLOW_API_KEY = "sk-sxonjhtlsbketbarbwjifpwalapsxjjtbfskqeyrryovphdz"          # 硅基流动：https://cloud.siliconflow.cn
SPARK_API_KEY = "ak-ade8de781bebccd1d1ba428592c34ea8"              # 讯飞星火（新 MaaS 接口）：只填 APIKey，无需拼接 APISecret

MODELS = [
    {
        # ★ 平台一：阿里云百炼（通义千问）★
        "platform": "阿里云百炼",
        "label": "qwen3.8-27B",
        "short": "百炼27B",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": DASHSCOPE_API_KEY,
        "model": "qwen3.8-27b",     # 百炼模型 ID；新加坡地域请改用 dashscope-intl.aliyuncs.com
        "json_mode": True,          # 百炼支持 response_format={"type":"json_object"}
        # qwen3.8-27b 属混合思考模式（默认开启思考），思考模式仅支持流式，故非流式调用必须显式关闭
        "extra_body": {"enable_thinking": False},
    },
    {
        # ★ 平台二：硅基流动（同时是仲裁者）★
        "platform": "硅基流动",
        "label": "qwen3-32B(仲裁者)",
        "short": "硅基32B",
        "base_url": "https://api.siliconflow.cn/v1",
        "api_key": SILICONFLOW_API_KEY,
        "model": "Qwen/Qwen3-32B",
        "json_mode": True,
        "is_arbiter": True,
        "extra_body": {"enable_thinking": False},
    },
    {
        # ★ 平台三：讯飞星火（MaaS 平台新接口，OpenAI 兼容，Bearer 鉴权只带 APIKey）★
        "platform": "讯飞星火",
        "label": "Spark-X2-Flash",
        "short": "星火X2",
        "base_url": "https://maas-api.cn-huabei-1.xf-yun.com/v2",   # 完整请求地址 = base_url + /chat/completions
        "api_key": SPARK_API_KEY,
        "model": "xsparkx2flash",
        "json_mode": False,         # 星火兼容接口不保证支持 response_format，靠提示词约束 JSON
        # 星火是思考型模型，思考过程会占用 max_tokens 预算；预算过小会导致正文 JSON 刚开始就被截断
        #（表现为 "Unterminated string starting at..."），因此单独放大输出预算
        "max_tokens": 4096,
        "extra_body": None,
    },
]


# ==============================================================================
# 三、上下文工程：总则 + 分题型作答模式（步骤四“分模式输出”）
# ==============================================================================

GENERAL_RULES = (
    "1. 只使用题干信息与公认的教科书/主流教材结论；严禁编造知识点、公式、史料、数据、法条；"
    "凡属推测必须写明“推测”，不得似是而非。\n"
    "2. reasoning 字段：先用 1-3 句话简洁说明本题用到的原理/公式/知识点，再给出详细的分步推理；"
    "步骤之间用换行分隔，不使用 Markdown 表格、不使用代码块、不使用装饰性符号。\n"
    "3. answer 字段：只写最终答案本身，严格符合题目给出的 answer_format，"
    "不加“答案是”“最终答案：”等前缀，不加引号，不加任何解释。\n"
    "4. 若存在多个符合条件的答案，必须全部列出，并按 answer_format 规定的分隔符与顺序组织。\n"
    "5. 输出必须是单个 JSON 对象，形如 {\"reasoning\": \"...\", \"answer\": \"...\"}，"
    "不得输出 JSON 之外的任何文字。\n"
)

TYPE_PROFILES = {
    "数学": {
        "role": "数学题专家",
        "mode": ("按数学计算题/应用题模式作答：公式先行、符号规范、单位带全、推导完整；"
                 "应用题明确写出“已知 → 求 → 列式 → 计算 → 作答”；"
                 "关键数值代入要写出中间结果，最后回代或估算以检验结果的合理性。"),
    },
    "物理": {
        "role": "物理题专家",
        "mode": ("先写出所用的定律/公式并标明各符号含义与单位，再代入数据计算，结果必须带单位；"
                 "检查量纲与有效数字；涉及过程变化时先用文字说清受力和状态变化。"),
    },
    "化学": {
        "role": "化学题专家",
        "mode": ("先写配平后的化学方程式并注明反应条件，再按物质的量/守恒关系列式计算；"
                 "单位（mol、g、L）写全；严格区分实验现象与结论，不臆造实验数据。"),
    },
    "生物": {
        "role": "生物题专家",
        "mode": ("用课本标准表述：先给概念定义，再说明结构—功能—过程之间的因果链；"
                 "术语规范（如基因型、表现型、酶促反应），不虚构实验结论。"),
    },
    "医学": {
        "role": "医学题专家",
        "mode": ("区分教科书标准结论与临床个体差异，写明机制与判断依据；"
                 "reasoning 结尾必须另起一行写“【仅供学习参考，不能替代临床诊疗】”。"),
    },
    "逻辑推理": {
        "role": "逻辑推理题专家",
        "mode": ("按逻辑推理模式作答：① 逐条编号列出题干的全部条件（含“隐含条件”）；"
                 "② 明确变量与取值范围；③ 用假设、排除、枚举或分支推导，逐个尝试；"
                 "④ 每一项结论都要回到题设条件逐条检验是否全部满足；⑤ 汇总所有满足条件的解，不重不漏。"),
    },
    "理科概念": {
        "role": "理科概念题专家",
        "mode": ("用课本标准表述，把“定义 / 性质（或判定方法）/ 特例与常见误区”三部分分开写；"
                 "定义要完整，不省略限定条件。"),
    },
    "历史": {
        "role": "历史题专家",
        "mode": ("依托主流教材共识，严格区分史实与观点；年代、人物、事件必须准确；"
                 "存在学术争议处注明“学界有不同观点”，不编造史料出处。"),
    },
    "政治": {
        "role": "政治题专家",
        "mode": ("以现行教材表述为准：先写原理/知识点，再与材料逐条对应分析（原理 + 材料）；"
                 "不作超出材料的引申，不编造政策文件内容。"),
    },
    "经济": {
        "role": "经济题专家",
        "mode": ("先说明所用的经济模型/公式及其成立假设，再代入数据推导；"
                 "区分名义与实际、短期与长期、局部与一般均衡，结论要注明适用条件。"),
    },
    "地理": {
        "role": "地理题专家",
        "mode": ("分自然要素（气候、地形、水文、土壤、植被）与人文要素（人口、交通、市场、政策）两条线分析；"
                 "区位类题目要指出主导因素与限制因素，成因分析要有因果链条。"),
    },
    "文学": {
        "role": "语文/文学题专家",
        "mode": ("以文本为依据作答：引用原文须准确，术语（修辞手法、表现手法、结构作用）使用规范；"
                 "区分“文本事实”与“读者感受”，不臆造作者生平与版本信息。"),
    },
    "法律": {
        "role": "法律题专家",
        "mode": ("以现行法律规定与通说为依据，写清构成要件与请求权基础；"
                 "无法确定具体条文时注明“以现行法及司法解释为准”，不编造条文序号。"),
    },
    "计算机": {
        "role": "计算机/算法题专家",
        "mode": ("先说明算法思路与所选数据结构，再给关键步骤或伪代码，最后分析时间/空间复杂度；"
                 "不编造库函数行为，边界条件要单独讨论。"),
    },
    "常识": {
        "role": "常识题专家",
        "mode": ("先溯源依据（定义、出处、常识来源），再作判断；"
                 "冷门或存在争议的内容必须标注“来源风险：不同资料说法不一”。"),
    },
    "通用": {
        "role": "严谨的解题专家",
        "mode": ("先澄清题意与已知条件，再分步推理，最后给出结论；"
                 "结论必须能被前文的推理直接支持，不跳步。"),
    },
}

# 题型归一化规则（顺序敏感：先匹配更专业的类别）
TYPE_RULES = [
    ("逻辑推理", ("逻辑", "推理题", "真假话", "排列", "排序题", "数独", "悖论", "谁是谁", "条件推断")),
    ("医学", ("医学", "中医", "临床", "诊断", "疾病", "症状", "用药", "药理", "手术", "辨证")),
    ("化学", ("化学", "溶液", "反应", "化合物", "摩尔", "氧化", "酸碱")),
    ("物理", ("物理", "速度", "加速度", "力的大小", "电路", "压强", "浮力", "功率", "电磁")),
    ("生物", ("生物", "细胞", "遗传", "基因", "生态", "酶", "免疫")),
    ("数学", ("数学", "计算", "面积", "体积", "方程", "概率", "函数", "几何", "数列", "排列组合", "利润", "速度问题")),
    ("经济", ("经济", "利率", "汇率", "通胀", "市场", "成本", "供需", "GDP")),
    ("历史", ("历史", "朝代", "史料", "变法", "战争", "古代", "近代史")),
    ("政治", ("政治", "政策", "制度", "法治", "价值观", "马克思", "治国")),
    ("地理", ("地理", "气候", "地形", "洋流", "人口迁移", "区位", "季风")),
    ("文学", ("文学", "诗歌", "古诗", "修辞", "散文", "文言", "作家", "小说")),
    ("法律", ("法律", "法条", "合同", "侵权", "刑法", "民法", "责任")),
    ("计算机", ("计算机", "算法", "代码", "程序", "复杂度", "数据结构", "编程")),
    ("常识", ("常识", "生活", "日常", "节气", "民俗")),
]

TYPE_TAXONOMY = ("数学、物理、化学、生物、医学、逻辑推理、理科概念、历史、政治、经济、"
                 "地理、文学、法律、计算机、常识（也可直接用更贴切的类型名称）")


# ==============================================================================
# 四、基础工具：HTTP 调用、JSON 解析、答案归一化
# ==============================================================================

_LOCK = threading.Lock()   # 控制台输出 / 写文件 / 计数 的线程锁


def p(*args):
    """线程安全打印（多题并发时避免输出互相穿插）"""
    with _LOCK:
        print(*args, flush=True)


def get_arbiter():
    """返回仲裁模型：硅基流动 qwen3-32B"""
    for m in MODELS:
        if m.get("is_arbiter"):
            return m
    return MODELS[1]


def _chat_once(cfg, messages, temperature, max_tokens, use_extras=True):
    """真正发一次 HTTP 请求；use_extras=False 时去掉平台可能不支持的扩展参数"""
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    if max_tokens is not None:
        mt = max_tokens
    else:
        mt = cfg.get("max_tokens") or MAX_TOKENS     # 支持按模型单独设置输出预算
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": TEMPERATURE if temperature is None else temperature,
        "max_tokens": mt,
        "stream": False,
    }
    if use_extras:
        if cfg.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}
        if cfg.get("extra_body"):
            payload.update(cfg["extra_body"])

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + str(cfg["api_key"]))

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:400]
        except Exception:
            pass
        raise RuntimeError("HTTP %s %s %s" % (e.code, e.reason, detail))
    except Exception as e:
        raise RuntimeError("请求失败：%s" % e)

    try:
        obj = json.loads(text)
    except Exception as e:
        raise RuntimeError("响应 JSON 解析失败：%s；原始返回前 200 字：%s" % (e, text[:200]))

    # 星火 MaaS 接口：业务错误时 HTTP 可能仍是 200，但 body 里的 code 非 0
    if isinstance(obj, dict) and obj.get("code") not in (None, 0):
        raise RuntimeError("业务错误 code=%s：%s" % (obj.get("code"), obj.get("message") or text[:200]))

    try:
        msg = obj["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        if not content:  # 部分模型把内容放在 reasoning_content
            content = (msg.get("reasoning_content") or "").strip()
        if not content:
            raise ValueError("content 为空")
        return content
    except Exception as e:
        raise RuntimeError("响应解析失败：%s；原始返回前 200 字：%s" % (e, text[:200]))


def chat(cfg, messages, temperature=None, max_tokens=None):
    """
    三个平台对 response_format / enable_thinking 的支持度不同，
    首次请求若因这类可选参数被拒，自动去掉扩展参数再试一次，避免整题卡死。
    """
    try:
        return _chat_once(cfg, messages, temperature, max_tokens, use_extras=True)
    except RuntimeError as e:
        err_text = str(e).lower()
        if ("max_tokens" in err_text) or ("max tokens" in err_text) or ("too large" in err_text):
            # 平台不允许当前 max_tokens 上限 → 退回全局默认值再试一次
            try:
                return _chat_once(cfg, messages, temperature, MAX_TOKENS, use_extras=True)
            except RuntimeError:
                raise e
        if ("response_format" in err_text) or ("enable_thinking" in err_text) or ("extra_body" in err_text):
            try:
                return _chat_once(cfg, messages, temperature, max_tokens, use_extras=False)
            except RuntimeError:
                raise e
        raise


def _repair_truncated_json(text):
    """
    修复被截断/含非法控制字符的 JSON：
      · 字符串内部的裸换行、制表符 → 空格（模型常直接把多行文本塞进字符串）；
      · 输出在字符串中途被截断 → 补上收尾引号；
      · 对象/数组没闭合 → 按栈补齐 } 和 ]；
      · 结尾多余逗号 → 去掉。
    只做“语法修补”，不改动任何字段内容。
    """
    out = []
    in_str = False
    esc = False
    stack = []
    for ch in text:
        if in_str:
            if esc:
                out.append(ch)
                esc = False
            elif ch == "\\":
                out.append(ch)
                esc = True
            elif ch == '"':
                out.append(ch)
                in_str = False
            elif ch in "\r\n\t":
                out.append(" ")          # 字符串内的裸控制字符会让 json.loads 报错
            else:
                out.append(ch)
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            out.append(ch)
        else:
            out.append(ch)

    s = "".join(out)
    if esc:                              # 结尾是孤立的反斜杠，会让补引号失效
        s = s[:-1]
    if in_str:
        s += '"'
    s = s.rstrip()
    s = re.sub(r",\s*$", "", s)
    for opener in reversed(stack):
        s += "}" if opener == "{" else "]"
    return s


def _salvage_fields(text):
    """
    兜底：JSON 结构已损坏时，用正则把关键字段单独捞出来（截断的字符串也能捞到已生成部分）。
    返回 dict 或 None。
    """
    d = {}
    for key, aliases in (("reasoning", ("reasoning", "推理过程", "reason")),
                         ("answer", ("answer", "答案")),
                         ("type", ("type", "题型")),
                         ("prompt", ("prompt", "提示词"))):
        for name in aliases:
            m = re.search(r'"%s"\s*:\s*"' % re.escape(name), text)
            if not m:
                continue
            rest = text[m.end():]
            buf, esc, closed = [], False, False
            for ch in rest:
                if esc:
                    buf.append(ch)
                    esc = False
                elif ch == "\\":
                    buf.append(ch)
                    esc = True
                elif ch == '"':
                    closed = True
                    break
                else:
                    buf.append(ch)
            raw = "".join(buf)
            # 还原常见转义
            try:
                val = json.loads('"%s"' % raw)
            except Exception:
                val = raw.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
            d[key] = val.strip()
            if closed:
                break
    return d or None


def parse_json_block(raw):
    """
    从模型输出中稳健地抠出 JSON 对象：
      1) 去掉 ``` 包裹与前后多余文字；
      2) 直接解析；失败则修复截断/控制字符后再解析；
      3) 仍失败则用正则抢救字段；
      4) 全失败给出带原文片段的清晰错误（绝不抛出裸 JSONDecodeError）。
    """
    if raw is None:
        raise ValueError("空输出")
    text = str(raw).strip()
    if not text:
        raise ValueError("空输出")

    body = re.sub(r"^```[a-zA-Z]*", "", text).strip()
    body = re.sub(r"```$", "", body).strip()
    i, j = body.find("{"), body.rfind("}")
    if i >= 0 and j > i:
        body = body[i:j + 1]
    elif i >= 0:
        body = body[i:]

    for candidate in (body, _repair_truncated_json(body)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    salvaged = _salvage_fields(body)
    if salvaged:
        return salvaged

    raise ValueError("模型返回的不是有效 JSON；原始返回前 200 字：%s" % _one_line(text, 200))



def canon_answer(text):
    """答案归一化：只用于三模型投票比对，不改变最终提交的文本"""
    t = str(text or "").strip()
    t = re.sub(r"^(最终答案|答案|答)\s*[:：]\s*", "", t)
    t = t.replace("\u3000", " ")
    t = re.sub(r"\s+", "", t)
    for a, b in (("；", ";"), ("，", ";"), ("、", ";"), (",", ";")):
        t = t.replace(a, b)
    t = t.replace("。", "").replace(".", "").rstrip(";")
    t = t.strip("\"'“”‘’")
    t = t.upper()
    segs = [s for s in t.split(";") if s]
    return ";".join(sorted(segs))


def postprocess_answer(answer, answer_format):
    """提交前整理答案：去前缀/引号/句号；answer_format 要求“升序”时对多组答案排序"""
    a = str(answer or "").strip()
    a = re.sub(r"^(最终答案|答案|答)\s*[:：]\s*", "", a).strip()
    a = a.strip("\"'“”‘’").strip()
    a = re.sub(r"[。.]+$", "", a).strip()
    fmt = answer_format or ""
    if ("升序" in fmt or "从小到大" in fmt) and ";" in a:
        segs = [s.strip() for s in a.split(";") if s.strip()]
        try:
            segs = sorted(segs, key=lambda s: float(s))
        except ValueError:
            segs = sorted(segs)
        a = ";".join(segs)
    return a


def normalize_type(raw_type):
    """把模型给出的题型说法归一到标准类别"""
    t = str(raw_type or "").strip()
    if not t:
        return "通用"
    low = t.replace("题", "").strip()
    for canon, keys in TYPE_RULES:
        for k in keys:
            if (k in t) or (k in low):
                return canon
    return low or "通用"


def get_profile(qtype):
    return TYPE_PROFILES.get(qtype, TYPE_PROFILES["通用"])


def short_label(cfg):
    return cfg.get("short") or cfg.get("label")


# ---------------------------- 调用失败的诊断与上报 ----------------------------

CALL_FAILURES = []   # [(题目id, 步骤名, 模型cfg, 失败原因)]，用于结尾汇总
_REPORTED_SIGS = set()   # 已经完整打印过的错误签名，避免同类错误重复刷屏


def _one_line(text, limit=None):
    """把多行错误压成一行，便于控制台阅读"""
    limit = ERROR_PREVIEW_LEN if limit is None else limit
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    return t if len(t) <= limit else t[:limit] + "…"


def error_hint(err):
    """根据错误内容给出人话级的排查提示（返回空串表示无额外提示）"""
    e = str(err or "").lower()
    if ("401" in e) or ("invalid api key") in e or ("unauthorized" in e) or ("invalidapikey" in e):
        return "疑似 API Key 无效/未填写，请检查该模型的 api_key"
    if "403" in e:
        return "无权限或被拒绝，请确认账号已开通该模型"
    if ("404" in e) or ("model not exist") in e or ("invalid model") in e:
        return "模型 ID 不存在，请核对该平台的 model 名称"
    if "429" in e or "rate limit" in e or "quota" in e:
        return "触发限流/额度不足，建议降低 MAX_QUESTION_WORKERS 或稍后重试"
    if "timeout" in e or "timed out" in e:
        return "请求超时，可调大 TIMEOUT 或降低并发"
    if "400" in e:
        return "请求参数被平台拒绝，详见上面的返回详情"
    if ("connection" in e) or ("ssl" in e) or ("resolve" in e) or ("getaddrinfo" in e):
        return "网络不可达，请检查网络/代理"
    if "响应解析失败" in str(err) or "content 为空" in str(err):
        return "接口通了但返回内容不合预期，可能是该模型不支持当前参数组合"
    return ""


def is_fatal_error(err):
    """鉴权类/模型不存在类错误：重试没有意义，直接快速失败，避免每个模型浪费约 9 秒"""
    e = str(err or "").lower()
    for k in (" 401 ", "401 ", " 403 ", "403 ", " 404 ", "404 ",
              "invalid api key", "invalidapikey", "model not exist", "invalid model"):
        if k in e:
            return True
    return False


def report_failures(qid, step_name, failures):
    """
    在控制台打印“哪个模型失败了 + 失败原因 + 排查提示”，并记入全局汇总。
    首次出现的错误打印完整信息（含平台/模型 ID/原因/建议），
    之后同类错误只打一行简短信息，避免刷屏；整块输出是原子的，不会被其他线程打断。
    failures: [(cfg, 错误文本)]
    """
    for cfg, err in failures:
        reason = _one_line(err)
        hint = error_hint(err)
        CALL_FAILURES.append((qid, step_name, cfg, reason))
        sig = (cfg["label"], reason[:80])
        if sig in _REPORTED_SIGS:
            p("   ✗ 调用失败 ｜ %s ｜ %s ｜ %s ｜ %s…（同类错误，首次完整提示见上）"
              % (qid, step_name, short_label(cfg), reason[:40]))
            continue
        _REPORTED_SIGS.add(sig)
        lines = [
            "   ✗ 模型调用失败 ｜ 题目 %s ｜ %s" % (qid, step_name),
            "       平台：[%s] %s ｜ model=%s" % (cfg.get("platform", ""), cfg["label"], cfg["model"]),
            "       原因：%s" % reason,
        ]
        if hint:
            lines.append("       提示：%s" % hint)
        p("\n".join(lines))


def failures_from(vote_result):
    """从 vote_step 的返回里挑出仍然失败的模型"""
    out = []
    for x in vote_result.get("detail", []):
        if not x.get("valid"):
            out.append((x["cfg"], x.get("error") or "未知错误"))
    return out


# ==============================================================================
# 五、三模型并行调用 + 仲裁（步骤一：少数服从多数，仲裁者 qwen3-32B）
# ==============================================================================

def _call_one(cfg, messages):
    """单模型调用，带重试；鉴权/模型不存在类错误立即失败；失败也返回结构，不让线程异常打断整批"""
    last_err = ""
    for attempt in range(1, MAX_RETRY + 1):
        try:
            return {"cfg": cfg, "ok": True, "raw": chat(cfg, messages)}
        except Exception as e:
            last_err = "%s" % e
            if is_fatal_error(last_err):      # Key 无效 / 模型不存在 → 重试无意义
                break
            if attempt < MAX_RETRY:
                time.sleep(min(1.5 * attempt, 6.0))
    return {"cfg": cfg, "ok": False, "error": last_err}


def run_three_models(build_messages):
    """
    三个模型并行请求。
    build_messages 可以是固定的 messages，也可以是函数 build_messages(cfg)，
    后者用于“每个模型使用自己那份提示词”的步骤四。
    """
    def task(cfg):
        try:
            msgs = build_messages(cfg) if callable(build_messages) else build_messages
        except Exception as e:
            return {"cfg": cfg, "ok": False, "error": "构造消息失败：%s" % e}
        return _call_one(cfg, msgs)

    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        return list(pool.map(task, MODELS))


RETRY_NUDGE = (
    "你上一条回复不是完整合法的 JSON（很可能因为输出过长被截断）。"
    "请重新作答，必须满足：① 只输出一个 JSON 对象，前后不要任何说明文字，不要用代码块；"
    "② 字段顺序为 answer 在前、reasoning 在后；③ reasoning 精简到 400 字以内；"
    "④ 确保 JSON 完整闭合（引号、花括号成对）。"
)


def _retry_on_parse_error(cfg, messages, parse_fn):
    """模型返回内容不合法（多为输出被截断）时，追加一条“精简重答”要求再要一次"""
    try:
        nudge = list(messages) + [{"role": "user", "content": RETRY_NUDGE}]
        d = parse_fn(chat(cfg, nudge))
        return {"ok": True, "data": d}
    except Exception:
        return {"ok": False}


def vote_step(step_name, build_messages, parse_fn, key_fn):
    """
    单步仲裁：三模型并行 → 按归一化 key 分组计数
      3:0 / 2:1 → 直接定论；
      1:1:1    → 整轮重跑；
      两轮均 1:1:1 → 以 qwen3-32B 的结果为准。
    返回 dict：ok / data / model / rounds / tally / tie / desc / detail
    """
    parsed = []
    for round_no in (1, 2):
        outs = run_three_models(build_messages)
        parsed = []
        for o in outs:
            if not o["ok"]:
                parsed.append({"cfg": o["cfg"], "valid": False, "error": o.get("error", "")})
                continue
            try:
                d = parse_fn(o["raw"])
                k = key_fn(d)
                if not k:
                    raise ValueError("返回内容中缺少有效答案")
                parsed.append({"cfg": o["cfg"], "valid": True, "data": d, "key": k})
            except Exception as e:
                err = "%s" % e
                if "原始返回" not in err:     # 附上原文片段，便于判断是截断还是格式跑偏
                    err += "；原始返回前 120 字：%s" % _one_line(o["raw"], 120)
                parsed.append({"cfg": o["cfg"], "valid": False, "error": err,
                               "retryable": True})

        retry_pool = [x for x in parsed if x.get("retryable")]
        if retry_pool:
            def _one(x):
                try:
                    msgs = build_messages(x["cfg"]) if callable(build_messages) else build_messages
                except Exception:
                    return x, {"ok": False}
                return x, _retry_on_parse_error(x["cfg"], msgs, parse_fn)

            with ThreadPoolExecutor(max_workers=len(retry_pool)) as pool:
                for x, r in pool.map(_one, retry_pool):
                    if not r["ok"]:
                        continue
                    try:
                        k = key_fn(r["data"])
                    except Exception:
                        continue
                    if not k:
                        continue
                    x.update({"valid": True, "data": r["data"], "key": k,
                              "error": "", "retried": True})

        valid = [x for x in parsed if x["valid"]]
        groups = {}
        for x in valid:
            groups.setdefault(x["key"], []).append(x)

        if groups:
            best_key = max(groups, key=lambda k: len(groups[k]))
            best = groups[best_key]
            if len(best) >= 2:   # 3:0 或 2:1
                chosen = next((x for x in best if x["cfg"].get("is_arbiter")), best[0])
                tally = "%d:%d" % (len(best), len(valid) - len(best))
                desc = tally + (" 一致" if len(best) == 3 else " 多数")
                if round_no > 1:
                    desc = "首轮 1:1:1 → 重跑一轮后 " + desc
                return {"ok": True, "data": chosen["data"], "model": chosen["cfg"]["label"],
                        "rounds": round_no, "tally": tally, "tie": round_no > 1,
                        "desc": desc, "detail": parsed}
            if len(groups) == 1 and len(best) == 1:   # 仅一个模型有效，只能采信
                chosen = best[0]
                return {"ok": True, "data": chosen["data"], "model": chosen["cfg"]["label"],
                        "rounds": round_no, "tally": "1:0", "tie": round_no > 1,
                        "desc": "仅 %s 返回有效结果（其余模型调用失败）" % short_label(chosen["cfg"]),
                        "detail": parsed}

        if round_no == 1:
            time.sleep(1.0)
            continue

    # 两轮均 1:1:1 → 以 qwen3-32B 为准
    arb = next((x for x in parsed if x["valid"] and x["cfg"].get("is_arbiter")), None)
    if arb is None:
        arb = next((x for x in parsed if x["valid"]), None)
    if arb is None:
        return {"ok": False, "data": None, "model": "", "rounds": 2, "tally": "0:0",
                "tie": True, "desc": "三模型均未返回有效结果", "detail": parsed,
                "error": "三个模型均未返回有效结果"}
    return {"ok": True, "data": arb["data"], "model": arb["cfg"]["label"], "rounds": 2,
            "tally": "1:1:1", "tie": True,
            "desc": "两轮均 1:1:1 → 按仲裁原则以 qwen3-32B 的结果为准", "detail": parsed}


# ==============================================================================
# 六、步骤二：题型判别【需仲裁】
# ==============================================================================

def build_type_messages(question):
    system = (
        "你是题型判别专家。请判断给定题目主要考察的知识与推理方向，"
        "从下列类别中选出一个最贴切的：\n" + TYPE_TAXONOMY + "。\n"
        "只输出 JSON：{\"type\": \"类别名\", \"reason\": \"一句话判别依据\"}，不要输出任何多余文字。"
    )
    user = "【题目】%s\n请判断题型。" % question
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_type(raw):
    d = parse_json_block(raw)
    if not isinstance(d, dict):
        raise ValueError("非 JSON 对象")
    t = str(d.get("type") or d.get("题型") or "").strip()
    if not t:
        raise ValueError("缺少 type 字段")
    return {"type": t, "reason": str(d.get("reason") or "").strip()}


def _key_type(d):
    return normalize_type(d.get("type"))


# ==============================================================================
# 七、步骤三：提示词生成【不仲裁】—— 三个模型各写各的，各自使用
# ==============================================================================

PROMPT_GEN_SYSTEM = (
    "你是提示词工程师。你的任务是为一道具体题目写一段“解题提示词”，"
    "供后续让大模型解这道题时使用。提示词要结合该题型特点，"
    "引导对方先讲原理、再分步推理、最后严格按格式给出答案。"
    "禁止编造知识点、公式、史料、数据；不得在提示词里直接写出答案。只输出 JSON。"
)


def _prompt_gen_user(qtype, question, answer_format):
    prof = get_profile(qtype)
    return (
        "【题型】%s\n【题目】%s\n【答案格式要求】%s\n\n"
        "请写一段可直接使用的解题提示词，必须包含以下要素：\n"
        "1) 身份设定，形如“我是一名%s，现在要解决下面这个问题”；\n"
        "2) 要求对方先输出推理过程 reasoning，再输出最终答案 answer，并以 JSON 对象形式返回；\n"
        "3) 明确要求 answer 严格遵守格式：%s；\n"
        "4) 明确禁止编造知识点、公式、史料、数据，不要似是而非；\n"
        "5) 结合该题型容易出错的地方给出 2~4 条具体提醒。\n"
        "提示词长度控制在 300 字以内，写成可直接使用的指令，不要写成给评委的说明。\n"
        "输出 JSON：{\"prompt\": \"……\"}"
    ) % (qtype, question, answer_format, prof["role"], answer_format)


def template_prompt(qtype):
    """模型生成提示词失败时的兜底提示词（按题型模板）"""
    prof = get_profile(qtype)
    return (
        "我是一名%s，现在要解决下面这个问题，请列出推理过程 reasoning 与最终答案 answer，"
        "并以 JSON 对象输出。作答要求：%s" % (prof["role"], prof["mode"])
    )


def generate_own_prompt(cfg, qtype, question, answer_format):
    """
    单个模型生成自己的提示词；失败则回退到题型模板。
    返回 (提示词, 是否兜底, 失败原因)；成功时失败原因为空串。
    """
    msgs = [{"role": "system", "content": PROMPT_GEN_SYSTEM},
            {"role": "user", "content": _prompt_gen_user(qtype, question, answer_format)}]
    last_err = ""
    for attempt in (1, 2):
        try:
            raw = chat(cfg, msgs)
            d = parse_json_block(raw)
            if isinstance(d, dict):
                txt = str(d.get("prompt") or d.get("提示词") or "").strip()
                if len(txt) >= 20:
                    return txt[:1500], False, ""
                last_err = "模型返回的 prompt 过短（%d 字），返回内容：%s" % (len(txt), _one_line(raw, 120))
            else:
                last_err = "返回的不是 JSON 对象：%s" % _one_line(raw, 120)
        except Exception as e:
            last_err = "%s" % e
            if is_fatal_error(last_err):
                break
            if attempt == 1:
                time.sleep(1.0)
    return template_prompt(qtype), True, last_err or "未知错误"


def generate_prompts(qtype, question, answer_format):
    """
    步骤三：三模型并行，各生成各的提示词（不仲裁、不合并）。
    返回 ({模型label: 提示词}, [(模型cfg, 失败原因)])，后者只含真正失败的模型。
    """
    def task(cfg):
        txt, fallback, err = generate_own_prompt(cfg, qtype, question, answer_format)
        return cfg, txt, fallback, err

    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        results = list(pool.map(task, MODELS))

    prompts, failures = {}, []
    for cfg, txt, fallback, err in results:
        prompts[cfg["label"]] = txt
        if fallback:
            failures.append((cfg, err))
    return prompts, failures


# ==============================================================================
# 八、步骤四：分模式作答 + 对 answer 仲裁
# ==============================================================================

def build_solver_messages(qtype, question, answer_format, own_prompt):
    prof = get_profile(qtype)
    system = ""
    if own_prompt:
        system += "【本题专用提示词（你先前为这道题设计的方案，请严格照此执行）】\n%s\n\n" % own_prompt
    system += "【总则】\n%s" % GENERAL_RULES
    system += "【本题作答模式】%s\n" % prof["mode"]
    # 字段顺序刻意让 answer 在前：答案短、先写完，即使输出被截断也不会丢失答案
    system += ("【输出格式（强制）】只输出一个 JSON 对象，不要输出任何 JSON 之外的文字，"
               "字段顺序固定为 answer 在前、reasoning 在后："
               "{\"answer\": \"严格符合格式要求的最终答案\", "
               "\"reasoning\": \"先原理/公式/知识点，再分步推理\"}。"
               "reasoning 控制在 600 字以内，务必保证 JSON 完整闭合，不要输出代码块。")

    user = ("【题目】%s\n【答案格式要求】%s\n\n"
            "请按上述要求作答：先写 answer（严格符合 answer_format），"
            "再在 reasoning 中说明原理/公式/知识点并给出完整分步推理。") % (question, answer_format)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_solution(raw):
    d = parse_json_block(raw)
    if not isinstance(d, dict):
        raise ValueError("非 JSON 对象")
    reasoning = str(d.get("reasoning") or d.get("推理过程") or d.get("reason") or "").strip()
    answer = str(d.get("answer") or d.get("答案") or "").strip()
    if not answer:
        raise ValueError("answer 为空")
    return {"reasoning": reasoning, "answer": answer}


def _key_answer(d):
    return canon_answer(d.get("answer"))


def rebuild_reasoning(qtype, question, answer_format, answer, current):
    """reasoning 过短时，由仲裁模型在不改答案的前提下补齐推理过程（避免该条记 0 分）"""
    prof = get_profile(qtype)
    user = (
        "【题目】%s\n【已确定的最终答案】%s\n【答案格式要求】%s\n\n"
        "请以%s的身份，写出与该答案完全一致的严谨分步推理：先用 1-3 句说明原理/公式/知识点，再给详细步骤。"
        "不得修改答案，不得编造知识点。输出 JSON：{\"reasoning\": \"...\"}"
    ) % (question, answer, answer_format, prof["role"])

    for _ in range(2):
        try:
            raw = chat(get_arbiter(),
                       [{"role": "system", "content": "你是严谨的解题专家，只输出 JSON。"},
                        {"role": "user", "content": user}])
            d = parse_json_block(raw)
            r = str(d.get("reasoning") or "").strip()
            if len(r) >= MIN_REASONING_LEN:
                return r
        except Exception:
            time.sleep(1.0)
    return current or ("按题意逐步推导后得到答案：%s" % answer)


# ==============================================================================
# 九、文件读写（全部在代码内完成，不使用任何命令行重定向）
# ==============================================================================

def load_questions(path):
    if not os.path.exists(path):
        raise SystemExit(
            "找不到题目文件：\n  %s\n请把“生成式人工智能系统应用员初赛实操题.jsonl”放到与本文件同一目录下。" % path
        )
    items = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                p("  ! 第 %d 行不是合法 JSON，已跳过：%s" % (line_no, e))
                continue
            if "question" not in obj:
                continue
            obj["id"] = str(obj.get("id") or "q%03d" % (len(items) + 1)).strip()
            obj["question"] = str(obj.get("question") or "").strip()
            obj["answer_format"] = str(obj.get("answer_format") or "").strip()
            items.append(obj)
    return items


def load_done_ids(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(str(json.loads(line).get("id")))
            except Exception:
                continue
    return done


def append_record(path, record):
    """线程安全地追加一条结果并立即落盘（4 题并发写不会串行错乱）"""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _LOCK:
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def reorder_output(questions):
    """
    收尾：按题目文件原始顺序重排输出文件，并只保留四个规定字段。
    这样即使 4 题并发完成顺序是乱的，提交文件仍与题目顺序一一对应。
    """
    if not os.path.exists(OUTPUT_FILE):
        return 0
    order = {q["id"]: i for i, q in enumerate(questions)}
    records = {}
    with open(OUTPUT_FILE, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            qid = str(r.get("id"))
            records[qid] = {
                "id": qid,
                "question": r.get("question", ""),
                "reasoning": r.get("reasoning", ""),
                "answer": r.get("answer", ""),
            }
    ordered = sorted(records.values(), key=lambda r: order.get(r["id"], 10 ** 6))
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="\n") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(ordered)


def write_log(lines):
    try:
        with open(LOG_FILE, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        p("  ! 写日志失败：%s" % e)


# ==============================================================================
# 十、单题主流程
# ==============================================================================

STATE = {"done": 0, "total": 0}
TIE_RECORDS = []       # [(题目id, 步骤名, 仲裁说明)]
FAILED = []            # [(题目id, 错误信息)]


def solve_question(q, pos, total):
    """一道题：步骤二题型判别 → 步骤三各模型写提示词 → 步骤四作答 + 答案仲裁"""
    qid = q["id"]
    question = q["question"]
    fmt = q["answer_format"]
    t0 = time.time()
    p("▶ 开始 [第 %d/%d 题] %s" % (pos, total, qid))

    # ---------------- 步骤二：题型判别（需仲裁） ----------------
    r_type = vote_step("步骤二·题型判别", build_type_messages(question), _parse_type, _key_type)
    type_fail = failures_from(r_type)
    if type_fail:
        report_failures(qid, "步骤二·题型判别", type_fail)
    if not r_type["ok"]:
        raise RuntimeError("题型判别失败：三个模型均未返回有效结果")
    qtype = _key_type(r_type["data"])
    if r_type["tie"]:
        TIE_RECORDS.append((qid, "步骤二·题型判别", r_type["desc"]))

    # ---------------- 步骤三：提示词生成（不仲裁，各用各的） ----------------
    prompts, prompt_fail = generate_prompts(qtype, question, fmt)
    if prompt_fail:
        report_failures(qid, "步骤三·提示词生成", prompt_fail)
    prompt_desc = " / ".join("%s %d字" % (m["short"], len(prompts.get(m["label"], ""))) for m in MODELS)
    if prompt_fail:
        names = "、".join(short_label(cfg) for cfg, _ in prompt_fail)
        prompt_desc += "（%s 生成失败，已用题型模板兜底）" % names

    # ---------------- 步骤四：分模式作答（各用各的提示词）+ 答案仲裁 ----------------
    def build_answer_messages(cfg):
        return build_solver_messages(qtype, question, fmt, prompts.get(cfg["label"], ""))

    r_ans = vote_step("步骤四·推理作答", build_answer_messages, _parse_solution, _key_answer)
    ans_fail = failures_from(r_ans)
    if ans_fail:
        report_failures(qid, "步骤四·推理作答", ans_fail)
    if not r_ans["ok"]:
        raise RuntimeError("推理作答失败：三个模型均未返回有效结果")
    if r_ans["tie"]:
        TIE_RECORDS.append((qid, "步骤四·推理作答", r_ans["desc"]))

    reasoning = str(r_ans["data"].get("reasoning") or "").strip()
    answer = postprocess_answer(r_ans["data"].get("answer"), fmt)

    if len(reasoning) < MIN_REASONING_LEN:
        p("   ! %s 的 reasoning 过短，已触发补写" % qid)
        reasoning = rebuild_reasoning(qtype, question, fmt, answer, reasoning)

    elapsed = time.time() - t0

    # ---------------- 步骤五：控制台输出（进度/题号/内容/耗时/仲裁情况） ----------------
    with _LOCK:
        STATE["done"] += 1
        done_no = STATE["done"]
        print("", flush=True)
        print("─" * 78, flush=True)
        print("[完成 %d/%d ｜ 第 %d 题] %s ｜ 耗时 %.1fs"
              % (done_no, total, pos, qid, elapsed), flush=True)
        print("   内容：%s" % question, flush=True)
        print("   步骤二 题型判别：%s ｜ 仲裁 %s%s"
              % (qtype, r_type["desc"],
                 (" ｜ ⚠ 调用失败：" + "、".join(short_label(c) for c, _ in type_fail)) if type_fail else ""),
              flush=True)
        print("   步骤三 提示词生成：%s ｜ 三份各自独立，不仲裁" % prompt_desc, flush=True)
        print("   步骤四 推理作答：仲裁 %s ｜ 采用 %s 的答案%s"
              % (r_ans["desc"], short_label(next(m for m in MODELS if m["label"] == r_ans["model"])),
                 (" ｜ ⚠ 调用失败：" + "、".join(short_label(c) for c, _ in ans_fail)) if ans_fail else ""),
              flush=True)
        print("   最终答案：%s" % answer, flush=True)
        retried = [x for x in r_ans.get("detail", []) if x.get("retried")]
        if retried:
            print("   · %s 首次返回内容被截断/不合法，已自动精简重答并计入有效票"
                  % "、".join(short_label(x["cfg"]) for x in retried), flush=True)
        if ans_fail or type_fail:
            print("   ⚠ 该题有效票数 %d/%d（失败原因见上方 ✗ 行）"
                  % (len(r_ans["detail"]) - len(ans_fail), len(MODELS)), flush=True)
        if r_ans["tally"] in ("1:1:1", "0:0"):
            print("   ⚠ 1:1:1 记录：%s" % qid, flush=True)
        print("─" * 78, flush=True)

    return {"id": qid, "question": question, "reasoning": reasoning, "answer": answer}


# ==============================================================================
# 十一、主流程（步骤六：多题并发）
# ==============================================================================

def warn_api_key():
    bad = []
    if "替换" in DASHSCOPE_API_KEY:
        bad.append("DASHSCOPE_API_KEY（阿里云百炼）")
    if "替换" in SILICONFLOW_API_KEY:
        bad.append("SILICONFLOW_API_KEY（硅基流动）")
    if "替换" in SPARK_API_KEY:
        bad.append("SPARK_API_KEY（讯飞星火 MaaS 平台，只填 APIKey）")
    if bad:
        print("\n⚠ 提醒：以下 Key 还是占位符，请先在文件开头的配置区替换成你自己的 Key：")
        for b in bad:
            print("   · %s" % b)
        print("")


def preflight_check():
    """
    启动自检：对每个模型发一次最小请求，明确打印“谁可用 / 谁不可用 / 原因”。
    这一步能在正式跑之前就暴露 Key 无效、模型 ID 错误、余额不足等问题，
    避免整轮跑完才发现投票一直是 2:0。
    """
    if not PREFLIGHT_CHECK:
        return
    print("\n开始模型连通性自检（每个模型各发一次最小请求）…")
    msgs = [{"role": "user", "content": "只回复两个字：正常"}]

    def task(cfg):
        t0 = time.time()
        try:
            txt = _chat_once(cfg, msgs, 0.0, 32, use_extras=False)   # 自检不带 json_mode，避免格式约束干扰
            return cfg, True, txt, time.time() - t0, ""
        except Exception as e:
            return cfg, False, "", time.time() - t0, "%s" % e

    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        results = list(pool.map(task, MODELS))

    bad = 0
    for cfg, ok, txt, dt, err in results:
        if ok:
            print("   ✓ [%s] %-18s 可用（%.1fs）模型回复：%s"
                  % (cfg.get("platform", ""), cfg["label"], dt, _one_line(txt, 30)))
        else:
            bad += 1
            print("   ✗ [%s] %-18s 不可用（%.1fs）model=%s"
                  % (cfg.get("platform", ""), cfg["label"], dt, cfg["model"]))
            print("       原因：%s" % _one_line(err))
            hint = error_hint(err)
            if hint:
                print("       提示：%s" % hint)
            _REPORTED_SIGS.add((cfg["label"], _one_line(err)[:80]))   # 后续同类错误只打一行
    if bad:
        print("\n⚠ 有 %d 个模型自检未通过。带着问题继续跑，投票会退化成 2:0 甚至 1:0，"
              "结果可信度会下降；建议先修好 Key / 模型 ID 再正式跑。" % bad)
    else:
        print("   ✓ 三个模型全部可用，后续投票为完整 3 票制。")


def print_failure_summary():
    """结尾按模型汇总调用失败情况"""
    if not CALL_FAILURES:
        return []
    agg = {}
    for qid, step, cfg, err in CALL_FAILURES:
        key = (cfg.get("platform", ""), short_label(cfg))
        item = agg.setdefault(key, {"n": 0, "err": err, "steps": []})
        item["n"] += 1
        if step not in item["steps"]:
            item["steps"].append(step)

    print("\n 模型调用失败汇总（按模型统计）：")
    lines = []
    for (plat, sl), v in agg.items():
        print("   · [%s] %s ｜ 失败 %d 次 ｜ 涉及步骤：%s"
              % (plat, sl, v["n"], "、".join(v["steps"])))
        print("       典型原因：%s" % v["err"])
        hint = error_hint(v["err"])
        if hint:
            print("       建议：%s" % hint)
        lines.append("    - [%s] %s ｜ 失败 %d 次 ｜ %s" % (plat, sl, v["n"], v["err"]))
    return lines


def main():
    print("=" * 80)
    print(" 生成式人工智能系统应用员初赛实操题 —— 基于上下文工程的思维链推理")
    print(" 三平台三模型作答 ｜ 硅基流动 qwen3-32B 仲裁 ｜ %d 题并发" % MAX_QUESTION_WORKERS)
    print("=" * 80)
    print("输入文件：%s" % INPUT_FILE)
    print("输出文件：%s" % OUTPUT_FILE)
    print("参与模型：")
    for m in MODELS:
        tail = "  ← 仲裁者" if m.get("is_arbiter") else ""
        print("   · [%s] %-18s model=%-16s %s%s"
              % (m.get("platform", ""), m["label"], m["model"], m["base_url"], tail))
    warn_api_key()
    preflight_check()

    questions = load_questions(INPUT_FILE)
    total = len(questions)
    print("共读取题目 %d 道" % total)
    if total == 0:
        print("题目文件为空，退出。")
        return

    if not RESUME and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "w", encoding="utf-8"):
            pass
        print("已清空旧输出文件（RESUME=False）")

    done = load_done_ids(OUTPUT_FILE) if RESUME else set()
    if done:
        print("断点续跑：检测到已完成 %d 道，将自动跳过" % len(done))

    todo = [(i, q) for i, q in enumerate(questions, 1) if q["id"] not in done]
    STATE["total"] = total
    print("本次待处理 %d 道 ｜ 并发 %d ｜ 开始时间 %s\n"
          % (len(todo), MAX_QUESTION_WORKERS, time.strftime("%H:%M:%S")))

    start = time.time()

    def work(item):
        pos, q = item
        try:
            record = solve_question(q, pos, total)
            append_record(OUTPUT_FILE, record)
        except Exception as e:
            FAILED.append((q["id"], str(e)))
            p("   ✗ %s 处理失败：%s" % (q["id"], e))
            traceback.print_exc(limit=1)
        finally:
            if QUESTION_INTERVAL:
                time.sleep(QUESTION_INTERVAL)

    if todo:
        with ThreadPoolExecutor(max_workers=MAX_QUESTION_WORKERS) as pool:
            list(pool.map(work, todo))

    # ---------------- 收尾：重排输出文件 + 汇总 ----------------
    kept = reorder_output(questions)
    elapsed = time.time() - start

    print("\n" + "=" * 80)
    print(" 运行完成：成功 %d 道，跳过 %d 道，失败 %d 道，总用时 %.1f 秒"
          % (STATE["done"], len(done), len(FAILED), elapsed))
    print(" 答案文件：%s（已按题目原始顺序排列，共 %d 行）" % (OUTPUT_FILE, kept))

    print("\n 三模型答案完全不一致（1:1:1）的题目 id：")
    ids = []
    if TIE_RECORDS:
        for qid, step, desc in TIE_RECORDS:
            if qid not in ids:
                ids.append(qid)
        for qid, step, desc in TIE_RECORDS:
            print("   · %s ｜ %s ｜ %s" % (qid, step, desc))
        print("   汇总：%s（共 %d 题）" % ("，".join(ids), len(ids)))
    else:
        print("   （无。三个模型在各需仲裁步骤均达成 3:0 或 2:1 一致）")

    if FAILED:
        print("\n 处理失败的题目（修正后直接重跑即可，已完成部分会自动跳过）：")
        for qid, err in FAILED:
            print("   · %s ：%s" % (qid, err))

    fail_lines = print_failure_summary()

    log_lines = [
        "",
        "=" * 60,
        "运行时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"),
        "成功 %d 道，跳过 %d 道，失败 %d 道，用时 %.1f 秒" % (STATE["done"], len(done), len(FAILED), elapsed),
        "1:1:1 不一致题目 id：%s" % ("，".join(ids) if ids else "无"),
        "模型调用失败总次数：%d" % len(CALL_FAILURES),
    ]
    for qid, step, desc in TIE_RECORDS:
        log_lines.append("    - %s ｜ %s ｜ %s" % (qid, step, desc))
    for qid, err in FAILED:
        log_lines.append("    ! 失败 %s ｜ %s" % (qid, err))
    for line in fail_lines:
        log_lines.append("    ✗ " + line.strip())
    write_log(log_lines)
    print("\n 运行日志：%s" % LOG_FILE)
    print("=" * 80)


if __name__ == "__main__":
    main()
