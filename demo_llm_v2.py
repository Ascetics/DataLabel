from openai import OpenAI
import jsonlines

client = OpenAI(
    api_key="ms-d0b5328e-df10-4beb-a525-0fef6a75f083",
    # api_key: 使用魔搭的访问令牌(Access Token),
    # 可以从您的魔搭账号中获取：https://modelscope.cn/my/myaccesstoken
    base_url="https://api-inference.modelscope.cn/v1/"
    # base url: 指向魔搭API-Inference服务
    # https://api-inference.modelscope.cn/v1/。
)

judge_txt = '''
基于事实核查结果，评估以下描述的知识价值：
事实核查：%s

评估维度：
1. 可靠性：来源权威性与可验证性
2. 实用性：信息对实际问题的帮助程度
3. 系统性：信息组织的逻辑深度
4. 新颖性：相比常识的增量价值

请先判断前述内容综合价值判断，如果高价值那么先输出high，如果低价值那么先输入low，如果无法判断正确与否先输出unknown。
然后再说明为什么（给前述内容一段评价），对于时间、地点等不确定因素一律按照今天、中国大陆境内判断。
'''

fdesc = jsonlines.open('desc.jsonl', 'r')
fres = jsonlines.open('res_llm_v2.jsonl', 'w')
for index_line, desc_line in enumerate(fdesc):
    desc_txt = desc_line.get('text')
    content_text = judge_txt % desc_txt
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",  # 模型名字(model):使用魔搭上开源模型的Model Id，例如Qwen/Qwen2.5-Coder-32B-Instruct 。
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': content_text
            }
        ],
        stream=True
    )
    llm_eval_result = ''
    llm_eval_reason = ''
    for index_chunk, chunk in enumerate(response):
        if 1 == index_chunk:
            llm_eval_result = chunk.choices[0].delta.content
        if index_chunk > 2:
            llm_eval_reason += chunk.choices[0].delta.content
    desc_line['llm_eval_result'] = llm_eval_result
    desc_line['llm_eval_reason'] = llm_eval_reason
    fres.write(desc_line)

    # print(desc_line)
    # if index_line == 0:
    #     break

fdesc.close()
fres.close()
