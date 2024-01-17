def append_question(args, start_template, item, tokenizer, contexts=[]):
    
    max_value = args.max_context_window
    length_template = len(tokenizer(start_template)['input_ids'])

    if args.dataset_name == "medqa":
        options = item['options']
        qst = f"""\n### Question: 
{item['question'].strip()}
(A) {options['A'].strip()}
(B) {options['B'].strip()}
(C) {options['C'].strip()}
(D) {options['D'].strip()}
"""
        if len(options) == 5:
            qst += f"(E) {options['E'].strip()}\n"

        qst += "[/INST]"
    elif args.dataset_name in ["medmcqa", "mmlu"]:
        qst = f"""\n### Question:
{item['question']}
(A) {item['opa']}
(B) {item['opb']}
(C) {item['opc']}
(D) {item['opd']}
[/INST]
"""
        
    length_qst = len(tokenizer(qst)['input_ids'])
    max_value = max_value - (length_template+length_qst)
    
    if contexts:
        text = "\n### Context:\n"
        for ctx in contexts:
            if len(tokenizer(ctx)['input_ids']) <= max_value:
                text += f'{ctx}\n'
                max_value = max_value - len(tokenizer(ctx)['input_ids'])
    
    text = text + f"{qst}" if contexts else f"{qst}"
    return text 

def get_template_no_ctxs():
    return f"""[INST] <<SYS>>
You are a medical expert. Just answer the question as concise as possible based on the user needs. Always return the response in the expected format required by the user. 
<</SYS>>
Select the correct option.
"""

def get_template_medqa(shots): 
    template = f"""[INST] <<SYS>>
You are a medical expert. Just answer the question as concise as possible based on the user needs. Always return the response in the expected format required by the user. 
<</SYS>>
Make a choice based on the question and context. Take the following two questions as examples.
The response must be always only one single line as the following one: 
"The answer is (correct letter) correct option."

"""
    for i, shot in enumerate(shots): 
        template += f"""# Example {i+1}
### Context:
{shot['context'].strip()}

### Question:
{shot['question']}
(A) {shot['options']['A']}
(B) {shot['options']['B']}
(C) {shot['options']['C']}
(D) {shot['options']['D']}
"""
        template += f"(E) {shot['options']['E']}\n\nThe answer is ({shot['answer_idx']}) {shot['answer']}.\n\n" if len(shot['options']) == 5 else f"\nThe answer is ({shot['answer_idx']}) {shot['answer']}.\n\n"

    template += "Now help me with another question\n"
    return template

def get_template_medmcqa(shots, id2lbl={0: "A", 1: "B", 2: "C", 3: "D"}): 
    template = f"""[INST] <<SYS>>
You are a medical expert. Just answer the question as concise as possible based on the user needs. Always return the response in the expected format required by the user. 
<</SYS>>
Make a choice based on the question and context. Take the following two questions as examples.
The response must be always only one single line as the following one: 
"The answer is (correct letter) correct option."

"""
    for i, shot in enumerate(shots): 
        template += f"""# Example {i+1}
### Context:
{shot['context'].strip()}

### Question:
{shot['question']}
(A) {shot['opa']}
(B) {shot['opb']}
(C) {shot['opc']}
(D) {shot['opd']}

The answer is ({id2lbl[shot['cop']]}) {[shot['opa'], shot['opb'], shot['opc'], shot['opd']][shot['cop']]}.

"""
    template += "Now help me with another question\n"

    return template