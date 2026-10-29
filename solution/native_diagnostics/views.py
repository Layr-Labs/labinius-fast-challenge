"""Rebuild C view bytes independently of the Rust build script. No compilation."""
import json
from pathlib import Path, PurePosixPath


def doc(raw):
    return ''.join(('// '+line[3:] if line.startswith('//!') else line)+'\n'
                   for line in raw.splitlines()
                   if not line.startswith('#![allow(clippy::needless_range_loop)]'))


def block(raw, anchor):
    assert raw.count(anchor) == 1
    start = raw.index(anchor)
    depth = 0
    for at in range(raw.index('{', start), len(raw)):
        depth += (raw[at] == '{') - (raw[at] == '}')
        if depth == 0:
            return raw[start:at+1]
    raise ValueError('unclosed source block')


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
        slot_rel = 'labinius/crates/pcs/src/simd/sms_slot_gen.rs'
        raw = doc(read(sms_rel))
        needle = 'include!("sms_slot_gen.rs");'
        assert raw.count(needle) == 1
        raw = raw.replace(needle, 'include!('+json.dumps(remote+'/'+slot_rel)+');')
        if fixture:
            bridge = 'sms_control_view.rs' if role == 'control' else 'sms_canonical_view.rs'
            raw += read('native_diagnostics/src/'+bridge)
        result[role+'-sms'] = raw.encode()
        result[role+'-commit'] = module(read('labinius/crates/pcs/src/simd/commit.rs'), local_pcs/'simd', pcs+'/simd', {'sms': out+'/'+role+'-sms.rs'}).encode()
        result[role+'-simd'] = module(read('labinius/crates/pcs/src/simd/mod.rs'), local_pcs/'simd', pcs+'/simd', {'commit': out+'/'+role+'-commit.rs'}).encode()
        result[role+'-lib'] = module(read('labinius/crates/pcs/src/lib.rs'), local_pcs, pcs, {'simd': out+'/'+role+'-simd.rs'}).encode()
        result[role+'-sms-raw'] = (solution/sms_rel).read_bytes()
        result[role+'-sms-slot'] = (solution/slot_rel).read_bytes()
    for name in ('keygen','modulus_choice','parameters','rq'):
        result['control-'+name] = read('proof_generate/src/'+name+'.rs').replace('labinius::','labinius_control::').encode()
    for role, relative in [('candidate-c','proof_generate/src/main.rs'),('baseline-control','native_diagnostics/control/main.rs')]:
        raw = read(relative)
        result[role+'-main-raw'] = raw.encode()
        text = doc(raw if role == 'candidate-c' else raw.replace('labinius::','labinius_control::'))
        private = remote+'/proof_generate/src' if role == 'candidate-c' else out
        for name in ('keygen','modulus_choice','parameters','rq'):
            needle = 'mod '+name+';'
            assert text.count(needle) == 1
            text = text.replace(needle,'#[path = '+json.dumps(private+'/'+name+'.rs')+']\n'+needle)
        result[role] = text.encode()
    if fixture:
        raw = read('proof_generate/src/main.rs').replace('\r\n','\n')
        aligned = '#[repr(C, align(64))]\nstruct Aligned64<T>(T);'
        assert raw.count(aligned) == 1
        text = aligned+'\n'+block(raw,'const fn sms_canonical_means')+'\n'
        for prefix in ('const SMS_MUR_LEN:', 'static SMS_MUR:'):
            lines = [line for line in raw.splitlines() if line.startswith(prefix)]
            assert len(lines) == 1
            text += lines[0]+'\n'
        result['mean-static'] = (text+read('native_diagnostics/src/mean_static_view.rs')).encode()
    return result
