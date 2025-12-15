import anthropic, jsonlines

client = anthropic.Anthropic(
    api_key="ms-d0b5328e-df10-4beb-a525-0fef6a75f083",  # 请替换成您的ModelScope Access Token
    base_url="https://api-inference.modelscope.cn")

judge_txt = '请先判断前述内容是否正确，如果正确那么先输出high，如果不正确那么先输入low，如果不能判断正确与否先输出unknown，然后再说明为什么（给前述内容一段评价）。'
fdesc = jsonlines.open('desc.jsonl', 'r')
fres = jsonlines.open('res_llm_AnthropicAPI.jsonl', 'w')
for index_line, desc_line in enumerate(fdesc):
    desc_txt = desc_line.get('text')
    content_text = desc_txt + judge_txt
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
