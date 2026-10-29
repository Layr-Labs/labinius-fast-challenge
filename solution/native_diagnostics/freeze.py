"""Freeze helper inputs after source review. This does not build or run native code."""
import argparse
import hashlib
import json
import re
from pathlib import Path

SKIP = {'target','libobj','liblabrador.a','.git','__pycache__'}
PROTECTED = ['.yukon/setup.sh','.yukon/benchmark.sh','.yukon/run-prover.sh',
             '.yukon/lib/sandbox.sh','.yukon/lib/score.py','verifier/proof_verify/src/main.rs']
SHARED = {
    'best48f2': {
        'simd/ntt/gen_small.rs':'40508a583f9209678041aa61db468f8b0acba4eb102a9ed00085122f0c74bba8',
        'simd/sms_commit.rs':'e0c6e36b72ce651bb78d7678b82fd03855b80db375391b8ff771f642a1ccd593',
        'simd/sms_slot_gen.rs':'1541be110ec0ba192531aa3977477f802af910aa86b372658019bd41231d3ce2',
    },
    'candidate-a-pr804': {
        'simd/ntt/gen_small.rs':'ce04e601a6b3d721993e37dee26eac131d94c1b5c7e162d02f651fc89f675810',
        'simd/sms_commit.rs':'a42894363e9285075edbd9ce663f300d43a9e229e8408cd2ce90ca152e329647',
        'simd/sms_slot_gen.rs':'994d4dec78a2ca264d07f01227c5bd26cfa8c75135540c4652c3d5fecd6f3f7a',
    },
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_baseline(solution, selection):
    if (type(selection) is not dict or
            set(selection) != {'schema','status','choice','verdict_sha256'} or
            selection['schema'] != 'candidate-b-baseline-selection-v1' or
            selection['status'] != 'FINAL' or selection['choice'] not in SHARED or
            type(selection['verdict_sha256']) is not str or
            not re.fullmatch('[0-9a-f]{64}',selection['verdict_sha256'])):
        raise ValueError('final coordinator baseline selection is required')
    for name, wanted in SHARED[selection['choice']].items():
        path=Path(solution)/'labinius/crates/pcs/src'/name
        if digest(path.read_bytes()) != wanted:
            raise ValueError('shared B baseline differs from selection: '+name)
    return digest(json.dumps(selection,sort_keys=True,separators=(',',':')).encode())


def freeze(solution, protected_root, context_out, baseline_selection):
    solution, protected_root = Path(solution), Path(protected_root)
    selection=json.loads(Path(baseline_selection).read_text())
    baseline_hash=checked_baseline(solution,selection)
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
    context = {'schema':'candidate-b-native-v1','jobs':['canonical-1','canonical-2'],
               'frozen_solution_sha256':digest(guard.read_bytes()),'protected_sha256':protected,
               'fresh_seed_source':seed_line,'fresh_seed_source_sha256':digest(seed_line.encode()),
               'base_commit':'48f2a6d36f6a4828dffd515eb84dd6959adac17f',
               'base_tree':'9f712058f73a87e870099e589d3f3a4c1f531f43'}
    context.update(execution_baseline='FINAL',baseline_selection=selection,
                   baseline_selection_sha256=baseline_hash)
    Path(context_out).write_text(json.dumps(context,indent=2)+'\n')
    print(json.dumps({'source_files':len(rows),'source_bytes':sum(int(r.split('\t')[1]) for r in rows),
                      'guard_sha256':context['frozen_solution_sha256'],'context':str(context_out)}))


if __name__ == '__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--solution',required=True)
    a.add_argument('--protected-root',required=True)
    a.add_argument('--context-out',required=True)
    a.add_argument('--baseline-selection',required=True,
                   help='final coordinator selection after the candidate A verdict')
    x=a.parse_args()
    freeze(x.solution,x.protected_root,x.context_out,x.baseline_selection)
