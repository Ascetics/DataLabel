import dashscope
from dashscope import Generation
import json, jsonlines
from typing import Dict

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
                从以下维度评分（1-5分，5分为最高价值）：
                1. 可靠性：来源权威性与可验证性。                
                2. 实用性：信息对解决实际问题的帮助程度。
                3. 系统性：信息组织的逻辑深度与结构性。
                
                
                【第三步：综合标签判定规则】
                - high：事实核查不为"错误" AND ((≥2个维度≥4分) OR (可靠性维度≥4分)) 
                - low：事实核查为"错误" OR (≥2个维度≤2分) OR (可靠性维度≤2分)) 
                - unknown：其他情况
                
                【输出格式要求】
                你必须返回以下JSON格式，不要任何额外解释。
                所有返回的数据必须是UTF-8编码兼容的，比如：遇到一些物理问题不要返回UTF-8编码不兼容的内容
                对于base转码的问题，base解码得到raw格式的数据，再按照ASCII或者UTF-8解码，再进一步判断：
                ```json
                {{
                  "fact_check": {{
                    "verdict": "正确/错误/不确定",
                    "reason": "具体核查理由"
                  }},
                  "value_assessment": {{
                    "label": "high/unknown/low",
                    "dimensions": {{
                      "reliability": 1,
                      "practicality": 1,
                      "systematic": 1,
                      
                    }},
                    "overall_reason": "综合评估理由"
                  }}
                }}
                现在开始评估："""

    def label_single_text(self, text: str, max_retries: int = 3) -> Dict:
        """标注单个文本（带重试机制）"""

        prompt = self.build_prompt(text)

        for attempt in range(max_retries):
            try:
                # 调用DashScope API
                response = Generation.call(
                    model=self.model,
                    prompt=prompt,
                    temperature=0.1,  # 低随机性保证输出稳定
                    top_p=0.8,
                    max_tokens=1500,  # 足够生成完整JSON
                    result_format='message'  # 返回结构化消息
                )

                if response.status_code == 200:
                    # 提取模型回复内容
                    reply_content = response.output.choices[0].message.content

                    # 提取JSON部分（处理可能包含的代码块）
                    json_str = self.extract_json(reply_content)

                    # 解析JSON
                    result = json.loads(json_str)

                    # 添加元数据
                    result["metadata"] = {
                        "model": self.model,
                        "text": text,
                        "usage": {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    }

                    return result

                else:
                    print(f"API错误 (尝试 {attempt + 1}/{max_retries}): {response.code} - {response.message}")

            except Exception as e:
                print(f"处理异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return self.create_fallback_result(text, f"处理失败: {str(e)}")

        return self.create_fallback_result(text, "超过最大重试次数")

    def extract_json(self, text: str) -> str:
        """从回复中提取JSON字符串"""
        import re

        # 尝试匹配JSON代码块
        json_block_pattern = r'```json\s*([\s\S]*?)\s*```'
        match = re.search(json_block_pattern, text)

        if match:
            return match.group(1).strip()

        # 尝试匹配纯JSON对象
        json_pattern = r'\{[\s\S]*\}'
        match = re.search(json_pattern, text)

        if match:
            return match.group(0).strip()

        # 如果都没有找到，返回原始文本（最后尝试）
        return text.strip()

    def create_fallback_result(self, text: str, error_msg: str) -> Dict:
        """创建降级结果"""
        return {
            "fact_check": {
                "verdict": "不确定",
                "reason": f"自动评估失败: {error_msg}"
            },
            "value_assessment": {
                "label": "unknown",
                "dimensions": {
                    "reliability": 1,
                    "practicality": 3,
                    "systematic": 3,

                },
                "overall_reason": "系统处理异常，建议人工复核"
            },
            "metadata": {
                "model": self.model,
                "text": text,
                "error": error_msg,
                "fallback": True
            }
        }

    def batch_label_f(self, input_file: str, output_file: str = 'data_dashscope_juesai.jsonl') -> None:
        print(f"开始批量标注文本")
        reader = jsonlines.open(input_file, mode='r')
        writer = jsonlines.open(output_file, mode='w')
        for i, one_json in enumerate(reader, 1):
            text = one_json.get('text')
            result = self.label_single_text(text)

            value_assessment = result.get("value_assessment")
            label = value_assessment.get("label")
            overall_reason = value_assessment.get("overall_reason")

            fact_check = result.get("fact_check")
            reason = fact_check.get("reason")

            one_json["llm_eval_result"] = label
            one_json["llm_eval_reason"] = reason
            writer.write(one_json)
            print(f"正在处理第{i}个文本...")
        print(f"标注完成！结果保存至 {output_file}")
        reader.close()
        writer.close()


if __name__ == "__main__":
    # 初始化标注器（可更换模型）
    MODEL = "qwen-turbo"  # 或 qwen-turbo, qwen-plus, qwen-max
    labeler = OnlineTextLabeler(model=MODEL)

    # 读取文件批量标注
    FILE_HEAD = 'data-cs'
    INPUT_FILE = f'{FILE_HEAD}.jsonl'
    OUTPUT_FILE = f'{FILE_HEAD}-result-{MODEL}.jsonl'
    labeler.batch_label_f(INPUT_FILE, OUTPUT_FILE)
