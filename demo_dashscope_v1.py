import dashscope
from dashscope import Generation
import json
from typing import Dict, Optional

# 设置API Key
# https://bailian.console.aliyun.com/?tab=model#/api-key
dashscope.api_key = "sk-27fdb227d40d4e0c84a9270f5b198da5"  # sk-your-api-key-here


class OnlineTextLabeler:
    """基于API的文本标注器"""

    def __init__(self, model="qwen-plus"):
        self.model = model

    def build_prompt(self, text: str) -> str:
        """构建结构化提示词"""
        return f"""请你作为文本质量评估专家，对以下描述进行两步评估：

【待评估文本】
{text}

【第一步：事实核查】
分析文本中的客观陈述：
- 若陈述正确且有可靠证据支持 → "正确"
- 若陈述错误且有可靠反证 → "错误"
- 若无法验证（主观、预测、证据不足）→ "不确定"
必须给出具体核查理由，提及相关证据或缺乏证据的原因。

【第二步：知识价值评估】
从四个维度评分（1-5分，5分为最高价值）：
1. 实用性：信息对解决实际问题的帮助程度
2. 新颖性：相比常识的增量信息价值
3. 系统性：信息组织的逻辑深度与结构性
4. 可靠性：来源权威性与可验证性

【第三步：综合标签判定规则】
- high：事实核查不为"错误" AND (≥3个维度≥4分)
- low：事实核查为"错误" OR (≥3个维度≤2分)
- medium：其他情况

【输出格式要求】
你必须返回以下JSON格式，不要任何额外解释：
```json
{{
  "fact_check": {{
    "verdict": "正确/错误/不确定",
    "reason": "具体核查理由"
  }},
  "value_assessment": {{
    "label": "high/medium/low",
    "dimensions": {{
      "practicality": 1,
      "novelty": 1,
      "systematic": 1,
      "reliability": 1
    }},
    "overall_reason": "综合评估理由"
  }}
}}
