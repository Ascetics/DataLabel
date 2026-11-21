from labeler import DashScopeAnnotator, ModelConfig

'''
分两个阶段标注。
'''


class CostOptimizedAnnotator(DashScopeAnnotator):
    """成本优化标注器"""

    def __init__(self, api_key: str):
        # 使用更便宜的模型
        model_config = ModelConfig(
            name="qwen-turbo",
            confidence_threshold=0.6,  # 降低阈值以增加自动化率
            temperature=0.2
        )
        super().__init__(api_key, model_config)

    def two_stage_annotate(self, text: str) -> Dict:
        """
        两阶段标注：
        1. 先用廉价模型快速判断
        2. 对不确定的样本用更贵模型深度分析
        """
        # 第一阶段：快速判断
        self.config.name = "qwen-turbo"
        stage1_result = self.annotate_single(text)

        # 如果置信度高，直接返回
        if (stage1_result["llm_eval_result"] != "unknown" and
                stage1_result.get("confidence", 0) > 0.8):
            return stage1_result

        # 第二阶段：深度分析
        self.config.name = "qwen-plus"
        self.config.temperature = 0.1
        stage2_result = self.annotate_single(text)

        return stage2_result
