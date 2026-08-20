import json
from pathlib import Path
from collections import Counter

report = json.loads((Path(__file__).resolve().parents[1] / 'reports' / 'dataset_audit.json').read_text(encoding='utf-8'))
c = report['captured']
print('class_counts')
for split, counts in c['class_counts'].items():
    vals = list(counts.values())
    print(split, 'classes', len(counts), 'total', sum(vals), 'min', min(vals), 'max', max(vals), 'median', sorted(vals)[len(vals)//2])
print('split_source_overlap')
for k, v in c['split_source_overlap'].items():
    print(k, v)
print('cross_split_near_duplicate_candidates')
for x in c['cross_split_near_duplicate_candidates']:
    print(x)
print('official')
for name, d in report['official']['directories'].items():
    print(name, 'count', d['record_count'], 'errors', len(d['errors']), 'duplicates', d['duplicate_file_count'], 'dims', d['dimensions'])
print('captured_exact_duplicate_groups')
for x in c['exact_duplicate_groups']:
    print(x)
