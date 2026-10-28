"""Rebuild expected generated view bytes from the frozen source, without compiling."""
import json
from pathlib import Path, PurePosixPath


def doc(raw):
    return ''.join(('// '+line[3:] if line.startswith('//!') else line)+'\n'
                   for line in raw.splitlines()
                   if not line.startswith('#![allow(clippy::needless_range_loop)]'))


def expected_views(solution, rows, fixture):
    solution = Path(solution)
    selected = {r['name']: r for r in rows if r['kind'] == 'view'}
    suffix = '/labinius/crates/pcs/src/simd/sms_commit.rs'
    remote = selected['fixture-sms-raw']['path']
    assert remote.endswith(suffix)
    remote = remote[:-len(suffix)]
    out = str(PurePosixPath(selected['control-sms']['path']).parent)
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
        sms_rel = 'native_diagnostics/control/sms_commit.rs' if role == 'control' else 'labinius/crates/pcs/src/simd/sms_commit.rs'
        slot_rel = 'native_diagnostics/control/sms_slot_gen.rs' if role == 'control' else 'labinius/crates/pcs/src/simd/sms_slot_gen.rs'
        raw = doc(read(sms_rel))
        needle = 'include!("sms_slot_gen.rs");'
        assert raw.count(needle) == 1
        raw = raw.replace(needle, 'include!('+json.dumps(remote+'/'+slot_rel)+');')
        if fixture:
            for needle, insertion in (
                ('let chunk = eh * cph + (local - 1) / CHUNK_PAIRS;', 'diagnostic_bias(ws.bias[0]);'),
                ('if half == 1 || b + LOCK == range.end {', 'diagnostic_pending(&ws.pend, half);'),
                ('gen_small::intt_gen_batch32::<P>(pd);', 'diagnostic_inverse(pd, d, both);'),
            ):
                assert raw.count(needle) == 1
                raw = raw.replace(needle, insertion+'\n'+needle if needle.startswith('if half') else needle+'\n'+insertion)
            raw += read('native_diagnostics/src/sms_private_view.rs')
        result[role+'-sms'] = raw.encode()
        ntt = module((local_pcs/'simd/ntt/mod.rs').read_text(), local_pcs/'simd/ntt', pcs+'/simd/ntt',
                     {'gen_small': remote+'/native_diagnostics/control/gen_small.rs' if role == 'control' else pcs+'/simd/ntt/gen_small.rs'})
        result[role+'-ntt'] = ntt.encode()
        commit = module((local_pcs/'simd/commit.rs').read_text(), local_pcs/'simd', pcs+'/simd', {'sms': out+'/'+role+'-sms.rs'})
        result[role+'-commit'] = commit.encode()
        simd = module((local_pcs/'simd/mod.rs').read_text(), local_pcs/'simd', pcs+'/simd',
                      {'ntt': out+'/'+role+'-ntt.rs', 'commit': out+'/'+role+'-commit.rs'})
        result[role+'-simd'] = simd.encode()
        result[role+'-lib'] = module((local_pcs/'lib.rs').read_text(), local_pcs, pcs, {'simd': out+'/'+role+'-simd.rs'}).encode()
        result[role+'-sms-raw'] = (solution/sms_rel).read_bytes()
        result[role+'-sms-slot'] = (solution/slot_rel).read_bytes()
    for name in ('keygen','modulus_choice','parameters','rq'):
        result['control-'+name] = read('proof_generate/src/'+name+'.rs').replace('labinius::','labinius_control::').encode()
    raw = read('proof_generate/src/main.rs')
    for role in ('candidate-a','best-control'):
        text = doc(raw if role == 'candidate-a' else raw.replace('labinius::','labinius_control::'))
        private = remote+'/proof_generate/src' if role == 'candidate-a' else out
        for name in ('keygen','modulus_choice','parameters','rq'):
            needle='mod '+name+';'
            assert text.count(needle)==1
            text=text.replace(needle,'#[path = '+json.dumps(private+'/'+name+'.rs')+']\n'+needle)
        result[role]=text.encode()
    return result
