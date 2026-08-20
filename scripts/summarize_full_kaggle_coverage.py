from __future__ import annotations
import argparse, json, re
from pathlib import Path

def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower().replace('hisui','hisuian').replace('galar','galarian').replace('alola','alolan').replace('paldea','paldean'))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=json.loads(ap.parse_args().input.read_text())
    folders=a['unmatched_folders']; missing=a['missing_slugs']
    nf={norm(x):x for x in folders}
    aliases=[]; true_missing=[]
    for m in missing:
        cand=nf.get(norm(m))
        (aliases if cand else true_missing).append({'official':m,'dataset_folder':cand} if cand else {'official':m})
    out={'official_catchable_records':a['official_catchable_records'],'dataset_folder_count':a['dataset_folder_count'],'matched_official_catchable_slugs':a['matched_official_catchable_slugs'],'missing_official_catchable_slugs':a['missing_official_catchable_slugs'],'unmatched_dataset_folders':a['unmatched_dataset_folders'],'valid_matched_image_files':a['valid_matched_image_files'],'exact_duplicate_hash_groups':a['exact_duplicate_hash_groups'],'normalized_alias_matches':aliases,'true_missing_after_aliases':true_missing,'unmatched_folder_sample':folders[:200]}
    ap.parse_args().output.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'aliases':len(aliases),'true_missing_after_aliases':len(true_missing),'true_missing':true_missing},indent=2))
if __name__=='__main__': main()
