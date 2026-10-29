"""Freeze helper inputs after source review. This does not build or run native code."""
import argparse
import hashlib
import json
from pathlib import Path

SKIP = {'target','libobj','liblabrador.a','.git','__pycache__'}
PROTECTED = ['.yukon/setup.sh','.yukon/benchmark.sh','.yukon/run-prover.sh',
             '.yukon/lib/sandbox.sh','.yukon/lib/score.py','verifier/proof_verify/src/main.rs']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def freeze(solution, protected_root, context_out):
    solution, protected_root = Path(solution), Path(protected_root)
    guard = solution/'native_diagnostics/frozen-source.tsv'
    rows = []
    for path in sorted(solution.rglob('*')):
        rel = path.relative_to(solution)
        if any(part in SKIP for part in rel.parts) or path == guard:
            continue
        if path.is_symlink():
            raise ValueError('source symbolic link: '+str(rel))
        if path.is_file():
            data=path.read_bytes()
            rows.append(f'{rel}\t{len(data)}\t{digest(data)}\n')
    guard.write_text(''.join(rows))
    protected = {name:digest((protected_root/name).read_bytes()) for name in PROTECTED}
    seed_line='  LABINIUS_SEED=$(openssl rand -hex 32)'
    if (protected_root/'.yukon/benchmark.sh').read_text().splitlines().count(seed_line) != 1:
        raise ValueError('unexpected protected seed source')
    context = {'schema':'candidate-c-native-v1','jobs':['canonical-1','canonical-2'],
               'frozen_solution_sha256':digest(guard.read_bytes()),'protected_sha256':protected,
               'fresh_seed_source':seed_line,'fresh_seed_source_sha256':digest(seed_line.encode()),
               'base_commit':'48f2a6d36f6a4828dffd515eb84dd6959adac17f',
               'base_tree':'9f712058f73a87e870099e589d3f3a4c1f531f43',
               'common_baseline':{'status':'FINAL','choice':'exact PR652','candidate_b_retained':False,
                 'b_fixed_parser_sha256':'86a1befe44e114349e2e27ad6799bfb972b54b557d496f45bf8b0a86934e510e'}}
    Path(context_out).write_text(json.dumps(context,indent=2)+'\n')
    print(json.dumps({'source_files':len(rows),'source_bytes':sum(int(r.split('\t')[1]) for r in rows),
                      'guard_sha256':context['frozen_solution_sha256'],'context':str(context_out)}))


if __name__ == '__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--solution',required=True)
    a.add_argument('--protected-root',required=True)
    a.add_argument('--context-out',required=True)
    x=a.parse_args()
    freeze(x.solution,x.protected_root,x.context_out)
