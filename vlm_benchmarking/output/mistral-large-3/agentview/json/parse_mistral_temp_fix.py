import json
import csv
import os
from pathlib import Path

jsonl_dir = Path(r"c:\Users\hassa\OneDrive\Desktop\vlm_benchmarking\output\mistral-large-3\agentview\json")


def extract_valid_triples(response_text):
    valid_triples = []
    for line in response_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('```'):
            continue
        parts = line.split(',')
        if len(parts) == 3:
            if all(' ' not in p.strip() for p in parts):
                valid_triples.append([p.strip() for p in parts])
    return valid_triples


for input_file in sorted(jsonl_dir.glob("*.jsonl")):
    output_file = input_file.with_suffix('.csv')

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', newline='', encoding='utf-8') as f_out:

        csv_writer = csv.writer(f_out)
        csv_writer.writerow(['task', 'demo', 'frame', 'camera', 'objectA', 'relation', 'objectB'])

        for line in f_in:
            if line.strip():
                try:
                    data = json.loads(line)
                    triples = extract_valid_triples(data.get('response', ''))
                    for subj, pred, obj in triples:
                        csv_writer.writerow([
                            data.get('task'), data.get('demo'), data.get('frame'),
                            data.get('camera'), subj, pred, obj
                        ])
                except json.JSONDecodeError:
                    continue

    print(f"Parsed: {input_file.name} -> {output_file.name}")
