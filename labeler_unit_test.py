import json

'''
生成一些测试样本。
'''

def create_sample_data(output_file: str, num_samples: int = 10):
    """创建测试数据"""
    samples = []

    test_texts = [
        "地球是太阳系中最大的行星。",
        "水的沸点在标准大气压下是100摄氏度。",
        "秦始皇统一六国后建立了唐朝。",
        "光在真空中的传播速度是每秒30万公里。",
        "人类可以在火星表面不穿宇航服生存。",
        "珠穆朗玛峰是世界最高峰。",
        "大熊猫是肉食性动物。",
        "COVID-19病毒可以通过空气传播。",
        "月亮本身会发光。",
        "中国的首都是上海。"
    ]

    for i in range(num_samples):
        sample = {
            "id": f"sample_{i + 1:04d}",
            "text": test_texts[i % len(test_texts)] if i < len(test_texts) else f"测试文本 {i + 1}",
            "llm_eval_result": "",
            "llm_eval_reason": "",
            "human_annotated_result": "",
            "human_annotated_reason": ""
        }
        samples.append(sample)

    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"已创建 {num_samples} 个测试样本到 {output_file}")


# 运行
create_sample_data("data-10.jsonl", 10)