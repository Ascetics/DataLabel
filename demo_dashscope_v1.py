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
                1. 可靠性：来源权威性与可验证性。                
                2. 实用性：信息对解决实际问题的帮助程度。
                3. 系统性：信息组织的逻辑深度与结构性。
                
                【第三步：综合标签判定规则】
                - high：事实核查不为"错误" AND ((≥2个维度≥4分) OR (可靠性维度≥4分)) 
                - low：事实核查为"错误" OR (≥2个维度≤2分) OR (可靠性维度≤2分)) 
                - unknown：其他情况
                
                【输出格式要求】
                你必须返回以下JSON格式，不要任何额外解释：
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

    def batch_label(self, texts: list, output_file: str = "labels.jsonl") -> None:
        """批量标注并保存为JSONL"""
        results = []

        print(f"开始批量标注 {len(texts)} 个文本...")

        for i, text in enumerate(texts, 1):
            print(f"处理第 {i}/{len(texts)} 个文本...")

            result = self.label_single_text(text)
            results.append(result)

            # 每10条保存一次进度
            if i % 10 == 0:
                self.save_progress(results, f"progress_{output_file}")
                print(f"  已保存进度 ({i}条)")

        # 最终保存
        self.save_as_jsonl(results, output_file)
        print(f"标注完成！结果保存至 {output_file}")

        # 输出统计信息
        self.print_statistics(results)

    def save_as_jsonl(self, data: list, filename: str) -> None:
        """保存为JSONL格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def save_progress(self, data: list, filename: str) -> None:
        """保存进度"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def print_statistics(self, results: list) -> None:
        """打印统计信息"""
        labels = [r["value_assessment"]["label"] for r in results]
        verdicts = [r["fact_check"]["verdict"] for r in results]

        print("\n" + "=" * 50)
        print("标注统计:")
        print("-" * 30)
        print(f"总计: {len(results)} 条")

        print(f"\n事实核查分布:")
        for v in ["正确", "错误", "不确定"]:
            count = verdicts.count(v)
            print(f"  {v}: {count} ({count / len(results) * 100:.1f}%)")

        print(f"\n价值标签分布:")
        for l in ["high", "low", "unknown"]:
            count = labels.count(l)
            print(f"  {l}: {count} ({count / len(results) * 100:.1f}%)")

        # 计算平均维度分数
        avg_scores = {"reliability": 0, "practicality": 0, "systematic": 0}
        for r in results:
            for dim in avg_scores.keys():
                avg_scores[dim] += r["value_assessment"]["dimensions"][dim]

        print(f"\n平均维度分数:")
        for dim, total in avg_scores.items():
            avg = total / len(results)
            print(f"  {dim}: {avg:.2f}")

        print("=" * 50)


if __name__ == "__main__":
    # 1. 初始化标注器（可更换模型）
    labeler = OnlineTextLabeler(model="qwen-plus")  # 或 qwen-max, qwen-turbo

    # 2. 测试单个文本
    test_text = "地球绕太阳公转一周需要365.25天。"
    print(f"测试文本: {test_text}")

    result = labeler.label_single_text(test_text)

    print(f"\n事实核查: {result['fact_check']['verdict']}")
    print(f"核查理由: {result['fact_check']['reason']}")
    print(f"\n价值标签: {result['value_assessment']['label']}")
    print(f"综合理由: {result['value_assessment']['overall_reason']}")

    # 3. 批量处理示例
    sample_texts = [
        "水在零下100摄氏度会沸腾。",
        "Python是静态类型语言。",
        "人工智能将彻底改变医疗诊断的准确性。",
        "每天睡4小时对大多数成年人来说是最佳睡眠时长。",
        "太阳从西边升起。"
    ]

    # 批量标注
    labeler.batch_label(sample_texts, "res_dashscope_v1.jsonl")