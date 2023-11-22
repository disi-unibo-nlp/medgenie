def get_prompt_medmcqa(prompt_template, question, opa, opb, opc, opd):
    prompt = prompt_template + f"""\n\n### Question:
{question}
- {opa}
- {opb}
- {opc}
- {opd}

### Context:
"""	
    return prompt

def clean_generated_text(args, generated_text):
    if "pmc-llama" in args.model_name.lower():
        for delimiter in ["###Answer", "### Answer"]:
            if delimiter in generated_text:
                generated_text = generated_text.split(delimiter, 1)[0]
                break
    return generated_text.strip()


def get_prompts_medmcqa(template, data):
   
    questions = data['question']
    opas = data['opa']
    opbs = data['opb']
    opcs = data['opc']
    opds = data['opd']
    prompts = [get_prompt_medmcqa(template, question, opa, opb, opc, opd) for question, opa, opb, opc, opd in zip(questions, opas, opbs, opcs, opds)]
    
    return prompts
