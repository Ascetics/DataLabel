import json
import dashscope
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "qwen-plus"
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 500
    confidence_threshold: float = 0.8  # 置信度阈值


class DashScopeAnnotator:
    """基于DashScope API的自动化标注器"""

    def __init__(self, api_key: str, model_config: ModelConfig = None):
        """
        初始化标注器

        Args:
            api_key: DashScope API密钥
            model_config: 模型配置
        """
        dashscope.api_key = api_key

        if model_config is None:
            model_config = ModelConfig()
        self.config = model_config

        # 验证模型是否可用
        self._validate_model()

    def _validate_model(self):
        """验证模型是否可用"""
        try:
            from dashscope import Generation
            # 简单测试连接
            response = Generation.call(
                model='qwen-turbo',
                prompt='test',
                max_tokens=1
            )
            logger.info(f"模型验证成功，使用模型: {self.config.name}")
        except Exception as e:
            logger.error(f"模型验证失败: {e}")
            raise

    def _create_prompt(self, text: str) -> str:
        """
        创建标注prompt

        根据评分规则设计专业prompt：
        1. 要求模型给出明确判断（high/low/unknown）
        2. 要求提供详细理由
        3. 要求给出置信度
        """
        prompt_template = f"""
请对以下文本陈述的事实准确性进行判断：

文本："{text}"

请按以下步骤分析：

【第一步：事实核查】
基于公开可验证的知识（科学事实、历史事件、统计数据等），判断文本陈述是否正确。

【第二步：标注判断】
- 若文本描述**正确无误** → 标注为"high"
- 若文本描述**有错误或虚假** → 标注为"low"
- 若**无法确定或信息不足** → 标注为"unknown"

【第三步：详细说明】
提供清晰的理由，解释判断依据。如果标注为unknown，请说明不确定的原因。

【第四步：输出格式】
请以JSON格式回复，严格包含以下字段：
{{
    "verdict": "high/low/unknown",
    "confidence": 0.0到1.0之间的置信度,
    "reason": "详细的判断理由，至少50字"
}}

要求：
1. 判断必须基于客观事实，而非主观意见
2. 如果文本涉及模糊或争议性内容，优先标注为unknown
3. 理由必须具体、有逻辑
"""
        return prompt_template

    def annotate_single(self, text: str) -> Dict:
        """
        标注单个文本

        Returns:
            标注结果字典
        """
        from dashscope import Generation

        prompt = self._create_prompt(text)

        try:
            response = Generation.call(
                model=self.config.name,
                prompt=prompt,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens
            )

            result_text = response.output.text

            # 提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)

            if json_match:
                result = json.loads(json_match.group())

                # 验证结果格式
                verdict = result.get("verdict", "").lower()
                confidence = float(result.get("confidence", 0.0))
                reason = result.get("reason", "")

                # 置信度检查
                if confidence < self.config.confidence_threshold:
                    verdict = "unknown"
                    reason = f"置信度过低({confidence:.2f})，建议人工核查。" + reason

                # 验证verdict有效性
                if verdict not in ["high", "low", "unknown"]:
                    verdict = "unknown"
                    reason = f"模型返回格式异常: {verdict}。建议人工核查。" + reason

                return {
                    "llm_eval_result": verdict,
                    "llm_eval_reason": reason[:500] if len(reason) > 500 else reason,
                    "confidence": confidence,
                    "raw_response": result_text[:1000]  # 保存原始响应用于调试
                }

            else:
                # JSON解析失败
                logger.warning(f"JSON解析失败: {result_text[:100]}")
                return {
                    "llm_eval_result": "unknown",
                    "llm_eval_reason": "模型返回格式异常，无法解析JSON。",
                    "confidence": 0.0,
                    "raw_response": result_text[:1000]
                }

        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return {
                "llm_eval_result": "unknown",
                "llm_eval_reason": f"API调用异常: {str(e)[:100]}",
                "confidence": 0.0,
                "raw_response": ""
            }

    def batch_annotate(self,
                       input_file: str,
                       output_file: str,
                       max_samples: int = None,
                       skip_human_annotated: bool = True) -> Dict:
        """
        批量标注

        Args:
            input_file: 输入JSONL文件路径
            output_file: 输出JSONL文件路径
            max_samples: 最大处理样本数（None表示全部）
            skip_human_annotated: 是否跳过已有人工标注的样本

        Returns:
            统计信息字典
        """
        # 读取数据
        samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    samples.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    logger.warning(f"JSON解析失败的行: {line[:100]}")

        # 限制样本数
        if max_samples and len(samples) > max_samples:
            samples = samples[:max_samples]

        logger.info(f"开始处理 {len(samples)} 个样本...")

        # 统计信息
        stats = {
            "total": len(samples),
            "processed": 0,
            "skipped": 0,
            "auto_annotated": 0,
            "unknown_count": 0,
            "cost_estimate": 0.0
        }

        # 处理每个样本
        for i, sample in enumerate(tqdm(samples, desc="标注进度")):
            # 检查是否需要跳过
            if (skip_human_annotated and
                    sample.get("human_annotated_result") and
                    sample.get("human_annotated_result") != ""):
                stats["skipped"] += 1
                continue

            # 进行自动化标注
            text = sample.get("text", "")
            if not text:
                logger.warning(f"样本 {sample.get('id')} 无文本内容")
                continue

            # 调用API标注
            result = self.annotate_single(text)

            # 更新样本
            sample["llm_eval_result"] = result["llm_eval_result"]
            sample["llm_eval_reason"] = result["llm_eval_reason"]

            # 更新统计
            stats["processed"] += 1
            if result["llm_eval_result"] != "":
                stats["auto_annotated"] += 1
            if result["llm_eval_result"] == "unknown":
                stats["unknown_count"] += 1

            # 成本估算（假设平均每个样本500 tokens）
            stats["cost_estimate"] += 0.0005 if self.config.name == "qwen-turbo" else 0.0025

        # 保存结果
        self._save_results(samples, output_file)

        # 计算自动化率
        automation_rate = stats["auto_annotated"] / stats["total"] if stats["total"] > 0 else 0

        logger.info(f"""
=== 处理完成 ===
总计样本: {stats['total']}
已处理: {stats['processed']}
跳过: {stats['skipped']}
自动化标注: {stats['auto_annotated']}
Unknown数量: {stats['unknown_count']}
预估成本: ¥{stats['cost_estimate']:.2f}
自动化率: {automation_rate:.2%}
        """)

        return {
            "automation_rate": automation_rate,
            "stats": stats
        }

    def _save_results(self, samples: List[Dict], output_file: str):
        """保存结果到JSONL"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        logger.info(f"结果已保存至: {output_file}")


class ScoreCalculator:
    """得分计算器"""

    @staticmethod
    def calculate_scores(ground_truth_file: str, prediction_file: str) -> Dict:
        """
        计算最终得分

        Args:
            ground_truth_file: 包含人工标注的文件
            prediction_file: 包含自动化标注的文件

        Returns:
            得分字典
        """
        # 读取真实标签
        truth_data = {}
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                if sample.get("human_annotated_result"):
                    truth_data[sample["id"]] = {
                        "result": sample["human_annotated_result"],
                        "reason": sample["human_annotated_reason"]
                    }

        # 读取预测结果
        pred_data = {}
        with open(prediction_file, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                pred_data[sample["id"]] = {
                    "result": sample.get("llm_eval_result", ""),
                    "reason": sample.get("llm_eval_reason", "")
                }

        # 初始化计数器
        n_correct = 0
        n_unknown = 0
        n_llm = 0
        n_total = len(truth_data)

        # 计算指标
        for sample_id, truth in truth_data.items():
            if sample_id not in pred_data:
                continue

            pred = pred_data[sample_id]

            # 统计自动化标注样本
            if pred["result"] != "":
                n_llm += 1

            # 统计正确数量和unknown数量
            if pred["result"] == truth["result"]:
                n_correct += 1

            if pred["result"] == "unknown":
                n_unknown += 1

        # 计算得分
        if n_total == 0:
            return {
                "effective_accuracy": 0,
                "automation_rate": 0,
                "final_score": 0
            }

        effective_accuracy = (n_correct + n_unknown * 0.3) / n_total
        automation_rate = n_llm / n_total
        final_score = effective_accuracy * 0.7 + automation_rate * 0.3

        return {
            "effective_accuracy": effective_accuracy,
            "automation_rate": automation_rate,
            "final_score": final_score,
            "n_correct": n_correct,
            "n_unknown": n_unknown,
            "n_llm": n_llm,
            "n_total": n_total
        }


# 主函数
def main():
    """主执行函数"""

    # 配置参数
    API_KEY = "sk-27fdb227d40d4e0c84a9270f5b198da5"  # 替换为你的API密钥
    MODEL = "qwen-plus"  # 或 "qwen-turbo" 以降低成本
    FILE_HEAD = "data-cs"
    INPUT_FILE = f"{FILE_HEAD}.jsonl"  # 输入文件
    OUTPUT_FILE = f"{FILE_HEAD}-results-{MODEL}.jsonl"  # 输出文件
    GROUND_TRUTH_FILE = f"{FILE_HEAD}-ground-truth.jsonl"  # 真实标签文件（用于计算得分）

    # 模型配置（根据需求调整）
    model_config = ModelConfig(
        name=MODEL,
        temperature=0.1,
        confidence_threshold=0.7  # 置信度阈值，低于此值标为unknown
    )

    # 初始化标注器
    annotator = DashScopeAnnotator(api_key=API_KEY, model_config=model_config)

    # 批量标注
    stats = annotator.batch_annotate(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        max_samples=100,  # 测试时可限制样本数
        skip_human_annotated=True
    )

    # 计算得分（如果有真实标签）
    if GROUND_TRUTH_FILE:
        calculator = ScoreCalculator()
        scores = calculator.calculate_scores(GROUND_TRUTH_FILE, OUTPUT_FILE)

        print("\n" + "=" * 50)
        print("得分报告:")
        print(f"有效准确率 (Effective Accuracy): {scores['effective_accuracy']:.4f}")
        print(f"自动化率 (Automation Rate): {scores['automation_rate']:.4f}")
        print(f"最终得分 (Final Score): {scores['final_score']:.4f}")
        print(f"正确样本数: {scores['n_correct']}/{scores['n_total']}")
        print(f"Unknown样本数: {scores['n_unknown']}")
        print(f"自动化标注样本数: {scores['n_llm']}")
        print("=" * 50)


if __name__ == "__main__":
    main()
