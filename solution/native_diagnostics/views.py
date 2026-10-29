"""Rebuild B views from full frozen sources, without compiling."""
import json
from pathlib import Path, PurePosixPath


def doc(raw):
    return ''.join(('// '+line[3:] if line.startswith('//!') else line)+'\n'
                   for line in raw.splitlines()
                   if not line.startswith('#![allow(clippy::needless_range_loop)]'))


def row_doc(raw):
    return ''.join('// '+line[3:] if line.startswith('//!') else line
                   for line in raw.splitlines(keepends=True))


def expected_views(solution, rows, fixture):
    solution = Path(solution)
    selected = {r['name']: r for r in rows if r['kind'] == 'view'}
    suffix = '/labinius/crates/pcs/src/eval.rs'
    remote = selected['fixture-eval-raw']['path']
    assert remote.endswith(suffix)
    remote = remote[:-len(suffix)]
    out = str(PurePosixPath(selected['control-eval']['path']).parent)
    assert remote.isascii() and out.isascii()
    local_pcs = solution/'labinius/crates/pcs/src'
    pcs = remote+'/labinius/crates/pcs/src'
    result = {}

    def read(relative):
        return (solution/relative).read_bytes().decode()

    def module(raw, local_root, remote_root, overrides):
        lines = doc(raw).splitlines()
        text = ''
        for i, line in enumerate(lines):
            if not (line.startswith('pub mod ') and line.endswith(';')):
                if not line.startswith(('#[path=', '#[path =')):
                    text += line+'\n'
                continue
            name = line[len('pub mod '):-1]
            if name in overrides:
                path = overrides[name]
            elif i and lines[i-1].startswith('#[path'):
                path = remote_root+'/'+lines[i-1].split('"')[1]
            elif (local_root/(name+'.rs')).is_file():
                path = remote_root+'/'+name+'.rs'
            else:
                path = remote_root+'/'+name+'/mod.rs'
            text += '#[path = '+json.dumps(path)+']\npub mod '+name+';\n'
        return text

    for role in ('control', 'fixture'):
        relative = 'native_diagnostics/control/eval.rs' if role == 'control' else 'labinius/crates/pcs/src/eval.rs'
        bridge = 'native_diagnostics/src/row_control_view.rs' if role == 'control' else 'native_diagnostics/src/row_quad_view.rs'
        raw = (solution/relative).read_bytes()
        converted = row_doc(raw.decode()).encode()
        appended = (solution/bridge).read_bytes()
        result[role+'-eval-raw'] = raw
        result[role+'-row-bridge'] = appended
        result[role+'-eval-doc'] = converted
        result[role+'-eval'] = converted+(appended if fixture else b'')
        result[role+'-lib'] = module(read('labinius/crates/pcs/src/lib.rs'), local_pcs,
                                    pcs, {'eval': out+'/'+role+'-eval.rs'}).encode()
    for name in ('keygen','modulus_choice','parameters','rq'):
        result['control-'+name] = read('proof_generate/src/'+name+'.rs').replace('labinius::','labinius_control::').encode()
    raw = read('proof_generate/src/main.rs')
    for role in ('candidate-b','baseline-control'):
        text = doc(raw if role == 'candidate-b' else raw.replace('labinius::','labinius_control::'))
        private = remote+'/proof_generate/src' if role == 'candidate-b' else out
        for name in ('keygen','modulus_choice','parameters','rq'):
            needle='mod '+name+';'
            assert text.count(needle)==1
            text=text.replace(needle,'#[path = '+json.dumps(private+'/'+name+'.rs')+']\n'+needle)
        result[role]=text.encode()
    return result
