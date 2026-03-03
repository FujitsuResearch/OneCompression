import os
import json
import datetime
import csv
import re
import portalocker

def save_experiment_results(args, results, result_dir="../results"):
    os.makedirs(result_dir, exist_ok=True)
    
    json_dir = os.path.join(result_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    
    summary_csv = os.path.join(result_dir, "summary.csv")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"results_{timestamp}.json"
    json_path = os.path.join(json_dir, json_filename)

    pattern_llama = r'llama-(\d+)'
    pattern_B = r'(\d+)b'
    llama_match = re.findall(pattern_llama, args.model)
    B_match = re.findall(pattern_B, args.model)
    model_str = f"llama-{llama_match[0]}-{B_match[0]}b"
    
    run_results = {
        "timestamp": timestamp,
        "model": model_str,
        "method": args.method,
        "seed": args.seed,
        "nsamples": args.nsamples,
        "wbits": args.wbits,
        "groupsize": args.groupsize,
        "qep": args.qep,
        "percdampqep": args.percdampqep,
        "perccorr": args.perccorr,
        "percdamp": args.percdamp,
        "act-order": args.act_order,
    }
    run_results.update(results)

    summary = {
        "model": model_str,
        "method": args.method,
        "wbits": args.wbits,
        "groupsize": args.groupsize,
        "qep": args.qep,
    }
    summary.update(results)
    
    with open(json_path, "w") as f:
        json.dump(run_results, f, indent=2)
    
    fieldnames = list(summary.keys())
    
    with open(summary_csv, "a+", newline="", encoding="utf-8") as csvfile:
        portalocker.lock(csvfile, portalocker.LOCK_EX)
        try:
            csvfile.seek(0)
            first_line = csvfile.readline()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not first_line:
                writer.writeheader()
            writer.writerow(summary)
        finally:
            portalocker.unlock(csvfile)