"""Generate B row source views. This module does not build or run Rust."""
import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

CANDIDATE_EVAL = 'a945fdebd4dcbb3c79bef86378a03bc566b509bdb362a2626d4e653c22ff6897'
CONTROL_EVAL = '09ae065e388c67c24656c565f0f50193d4f6a2d92c2023c0677127e4c5d5416e'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def doc_view(raw):
    # Preserve line endings and every non-documentation byte in the eval module.
    return b''.join(b'// ' + line[3:] if line.startswith(b'//!') else line
                    for line in raw.splitlines(keepends=True))


def library_view(raw, pcs, eval_path):
    result = []
    lines = doc_view(raw).decode().splitlines(keepends=True)
    for i, line in enumerate(lines):
        text = line.rstrip('\r\n')
        if text.startswith('#![allow(clippy::needless_range_loop)]'):
            result.append('// ' + line)
        elif text.startswith('pub mod ') and text.endswith(';'):
            name = text[len('pub mod '):-1]
            if name == 'eval':
                path = eval_path
            elif i and lines[i - 1].startswith('#[path'):
                path = pcs / lines[i - 1].split('"')[1]
            elif (pcs / (name + '.rs')).is_file():
                path = pcs / (name + '.rs')
            else:
                path = pcs / name / 'mod.rs'
            assert path.is_file() or path == eval_path, path
            result.append('#[path = ' + json.dumps(str(path)) + ']\n' + text + '\n')
        elif not text.startswith(('#[path=', '#[path =')):
            result.append(line)
    return ''.join(result).encode()


def prepare(solution, out, fixture):
    solution, out = Path(solution).resolve(), Path(out).resolve()
    helper = solution / 'native_diagnostics'
    pcs = solution / 'labinius/crates/pcs/src'
    sources = {
        'fixture-eval-raw': pcs / 'eval.rs',
        'control-eval-raw': helper / 'control/eval.rs',
        'fixture-row-bridge': helper / 'src/row_quad_view.rs',
        'control-row-bridge': helper / 'src/row_control_view.rs',
        'row-lib-raw': pcs / 'lib.rs',
        'row-fixture': helper / 'tests/candidate_b_focused.rs',
        'row-generator': Path(__file__).resolve(),
    }
    raw = {name: path.read_bytes() for name, path in sources.items()}
    assert digest(raw['fixture-eval-raw']) == CANDIDATE_EVAL
    assert digest(raw['control-eval-raw']) == CONTROL_EVAL
    assert raw['fixture-eval-raw'].count(b'unsafe fn dot_words4(') == 1
    assert b'unsafe fn dot_words4(' not in raw['control-eval-raw']
    out.mkdir(parents=True, exist_ok=True)
    views, rows = {}, []
    for name, path in sources.items():
        rows.append({'kind': 'source', 'name': name, 'path': str(path),
                     'bytes': len(raw[name]), 'sha256': digest(raw[name])})
    for role in ('control', 'fixture'):
        converted = doc_view(raw[role + '-eval-raw'])
        views[role + '-eval-doc'] = converted
        views[role + '-eval'] = converted + (raw[role + '-row-bridge'] if fixture else b'')
        eval_path = out / (role + '-eval.rs')
        views[role + '-lib'] = library_view(raw['row-lib-raw'], pcs, eval_path)
    for name, data in views.items():
        path = out / (name + '.rs')
        path.write_bytes(data)
        rows.append({'kind': 'view', 'name': name, 'path': str(path),
                     'bytes': len(data), 'sha256': digest(data)})
    binding = {'schema': 'candidate-b-row-views-v1', 'fixture': fixture, 'rows': rows,
               'limit': 'Generated source only; no compile, native, proof, or runtime verdict.'}
    (out / 'row-bindings.json').write_text(json.dumps(binding, indent=2) + '\n')
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solution', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--mode', choices=('correctness', 'performance'), required=True)
    args = parser.parse_args()
    try:
        binding = prepare(args.solution, args.out, args.mode == 'correctness')
    except Exception as error:
        args.records.mkdir(parents=True, exist_ok=True)
        failure = args.records / 'first-row-helper-failure.json'
        if not failure.exists():
            with failure.open('x') as record:
                json.dump({'stage': 'row-source-view-preparation', 'error': repr(error),
                           'traceback': traceback.format_exc(), 'argv': sys.argv}, record, indent=2)
                record.write('\n')
        raise
    print(json.dumps({'stage': 'row-source-view-preparation', 'result': 'PASS',
                      'bindings': str(args.out / 'row-bindings.json'),
                      'source_rows': sum(r['kind'] == 'source' for r in binding['rows']),
                      'view_rows': sum(r['kind'] == 'view' for r in binding['rows'])}))


if __name__ == '__main__':
    main()
