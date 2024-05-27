def append_question(args, start_template, item, tokenizer, contexts=[]):
    
    max_value = args.max_context_window
    length_template = len(tokenizer(start_template)['input_ids'])

    if "pmc-llama" in args.model_name.lower():
        question = item['question'].strip()
        qst = f"\n### Question:\n{question}" if contexts else f"\n\n### Question:\n{question}"
    elif "llama-3" in args.model_name.lower() or "llama3" in args.model_name.lower() or "phi" in args.model_name.lower():
        question = item['question'].strip()
        qst = f"\nQuestion: {question}" if contexts else f"\n\nQuestion: {question}"
    else:
        question = item['question'].strip().replace("\nA.", "\n(A)").replace("\nB.", "\n(B)").replace("\nC.", "\n(C)").replace("\nD.", "\n(D)").replace("\nE.", "\n(E)")
        qst = f"\n### Question:\n{question}" if contexts else f"\n\n### Question:\n{question}"
    
    if "zephyr" in args.model_name.lower():
        qst += "</s>\n<|assistant|>"
    elif "llama-2" in args.model_name.lower():
        qst += "\n[/INST]"
    elif "pmc-llama" in args.model_name.lower():
        qst += "\n\n### Answer:\n"
    elif "llama-3" in args.model_name.lower() or "llama3" in args.model_name.lower():
        if "medqa" in  args.dataset_name.lower()  or "medmcqa" in args.dataset_name.lower():
            qst+="\nAnswer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "mmlu" in args.dataset_name.lower():
            qst+="\n\nAnswer by returning only the correct letter among A, B, C or D. Don't explain your choice.\nAnswer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "phi" in args.model_name.lower():
        qst+="\n\nAnswer by returning only the correct letter among A, B, C or D. Don't explain your choice.\nAnswer:<|end|>\n<|assistant|>"

    length_qst = len(tokenizer(qst)['input_ids'])
    max_value = max_value - (length_template+length_qst)


    if contexts and ("phi" in args.model_name.lower() and "mmlu" in args.dataset_name.lower()):
        text = ""
        for idx, ctx in enumerate(contexts[:args.n_contexts]):
            text += f"\nContext {idx+1}:\n"
            if len(tokenizer(ctx)['input_ids']) <= max_value:
                text += f'```{ctx}```\n'
                max_value = max_value - len(tokenizer(ctx)['input_ids'])
    elif contexts:
        if "llama-3" in args.model_name.lower() or "llama3" in args.model_name.lower():
            text = "\n\nContext: "
        elif "phi" in args.model_name.lower():
            text = "\nContext: "
        else:
            "\n\n### Context:\n"

        for ctx in contexts[:args.n_contexts]:
            if len(tokenizer(ctx)['input_ids']) <= max_value:
                text += f'{ctx}\n'
                max_value = max_value - len(tokenizer(ctx)['input_ids'])
                
    text = text + qst if contexts else qst

    return text 


def get_template(args): 

    if args.no_contexts:
        if args.n_options==4:
            with open(f'{args.templates_dir}/{args.model_name.split("/")[len(args.model_name.split("/"))-1]}/{args.dataset_name}/{args.dataset_name}{args.n_options}opt_no_ctxs.txt') as f:
                template = f.read().strip()
        else:
            with open(f'{args.templates_dir}/{args.model_name.split("/")[len(args.model_name.split("/"))-1]}/{args.dataset_name}/{args.dataset_name}{args.n_options}opt_no_ctxs.txt') as f:
                template = f.read().strip()
    elif args.n_options==4:
        with open(f'{args.templates_dir}/{args.model_name.split("/")[len(args.model_name.split("/"))-1]}/{args.dataset_name}/{args.dataset_name}{args.n_options}opt.txt') as f:
            template = f.read().strip()
    else:
        with open(f'{args.templates_dir}/{args.model_name.split("/")[len(args.model_name.split("/"))-1]}/{args.dataset_name}/{args.dataset_name}{args.n_options}opt.txt') as f:
            template = f.read().strip() 

    return template