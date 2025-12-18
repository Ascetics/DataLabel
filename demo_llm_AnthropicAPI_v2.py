import anthropic, jsonlines

client = anthropic.Anthropic(
    api_key="ms-d0b5328e-df10-4beb-a525-0fef6a75f083",  # 请替换成您的ModelScope Access Token
    base_url="https://api-inference.modelscope.cn")

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
fres = jsonlines.open('res_llm_AnthropicAPI_v2.jsonl', 'w')
for index_line, desc_line in enumerate(fdesc):
    desc_txt = desc_line.get('text')
    content_text = judge_txt % desc_txt
    llm_eval_result = ''
    llm_eval_reason = ''
    with client.messages.stream(
            model="Qwen/Qwen2.5-7B-Instruct",  # ModelScope Model-Id
            messages=[
                {"role": "user", "content": content_text}
            ],
            max_tokens=1024
    ) as stream:
        for text in stream.text_stream:
            if text in ('low', 'high', 'unknown'):
                llm_eval_result = text
            elif text.isspace():
                continue
            else:
                llm_eval_reason += text
    desc_line['llm_eval_result'] = llm_eval_result
    desc_line['llm_eval_reason'] = llm_eval_reason
    fres.write(desc_line)

    # print(desc_line)
    # if index_line == 0:
    #     break

fdesc.close()
fres.close()
