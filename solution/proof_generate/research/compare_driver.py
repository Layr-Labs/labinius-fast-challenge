#!/usr/bin/env python3
"""Offline build-stage full-process diagnostic; never changes the official score."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import stat
import statistics
import subprocess
import tempfile
import threading
import time

LIMIT = 64 * 1024 * 1024


class ProcessFailure(RuntimeError):
    def __init__(self, label, result):
        super().__init__(f'{label} failed; see {result["stdout"]}/{result["stderr"]}')
        self.result = result


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def emit(event, **data):
    data.update(event=event, official_score=False)
    print('cargo:warning=labinius-full-compare ' + json.dumps(data, sort_keys=True), flush=True)


def hardware():
    assert platform.system() == 'Linux' and platform.machine() == 'x86_64', 'unsupported host'
    cpus = sorted(os.sched_getaffinity(0))
    cpuinfo = Path('/proc/cpuinfo').read_text()
    model = next(line.split(':', 1)[1].strip() for line in cpuinfo.splitlines()
                 if line.startswith('model name'))
    flags = set(next(line.split(':', 1)[1].split() for line in cpuinfo.splitlines()
                     if line.startswith('flags')))
    required = {'avx512f', 'avx512bw', 'avx512vl', 'avx512vbmi', 'avx512_vbmi2', 'avx512_vnni', 'gfni'}
    assert '8488C' in model and len(cpus) == 8 and required <= flags, 'unsupported CPU/affinity/features'
    return dict(cpu_model=model, allowed_cpu_ids=cpus, required_flags=sorted(required),
                build_jobs=os.environ.get('NUM_JOBS'))


def execute(argv, env, cwd, logs, label, timeout=900):
    """Blocking wait; the independent watchdog never calls wait(timeout=...)."""
    stdout, stderr = logs / (label + '.stdout'), logs / (label + '.stderr')
    expired = threading.Event()
    state = {'proc': None}
    lock = threading.Lock()

    def kill():
        expired.set()
        with lock:
            proc = state['proc']
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    with stdout.open('wb') as out, stderr.open('wb') as err:
        watchdog = threading.Timer(timeout, kill)
        watchdog.daemon = True
        watchdog.start()
        started = time.perf_counter_ns()
        proc = None
        try:
            proc = subprocess.Popen(argv, env=env, cwd=cwd, stdout=out, stderr=err,
                                    start_new_session=True)
            with lock:
                state['proc'] = proc
                if expired.is_set():
                    os.killpg(proc.pid, signal.SIGKILL)
            rc = proc.wait()
            elapsed = time.perf_counter_ns() - started
        except BaseException:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
            raise
        finally:
            watchdog.cancel()
            watchdog.join()
    result = dict(argv=[str(x) for x in argv], returncode=rc, elapsed_ns=elapsed,
                  timed_out=expired.is_set(), stdout=stdout.name, stderr=stderr.name)
    emit('process', label=label, **result)
    if expired.is_set() or rc != 0:
        for name, path in [('stdout', stdout), ('stderr', stderr)]:
            emit('process_failure_tail', label=label, stream=name,
                 tail=path.read_bytes()[-16384:].decode('utf-8', errors='replace'))
    if expired.is_set() or rc != 0:
        raise ProcessFailure(label, result)
    return result


def table_integrity(path, expected_hash):
    st = path.lstat()
    regular = stat.S_ISREG(st.st_mode)
    actual = digest(path) if regular else None
    result = dict(regular_file=regular, mode=oct(stat.S_IMODE(st.st_mode)),
                  bytes=st.st_size, sha256=actual)
    result['unchanged'] = (regular and stat.S_IMODE(st.st_mode) == 0o444
                           and st.st_size == 268435456 and actual == expected_hash)
    return result


def source_copy(source, dest, candidate=False):
    source = source.resolve()

    def ignore(directory, names):
        ignored = {n for n in names if n in {'.git', 'target', '__pycache__'}}
        if candidate and Path(directory).resolve() == source / 'proof_generate':
            ignored.add('research')
        for name in names:
            assert not (Path(directory) / name).is_symlink(), 'source symlink refused'
        return ignored
    shutil.copytree(source, dest, ignore=ignore)


def run(solution, out):
    assert os.environ.get('NUM_JOBS') == '1', 'parent Cargo must be single-job/quiescent'
    assert os.environ.get('HOST') == os.environ.get('TARGET') == 'x86_64-unknown-linux-gnu', 'native host=target required'
    assert os.environ.get('LABINIUS_COMPARE_CHILD') != '1', 'diagnostic recursion'
    machine = hardware()
    os.umask(0o077)
    work = Path(tempfile.mkdtemp(prefix='full-compare-', dir=out))
    logs = work / 'logs'
    logs.mkdir()
    report = dict(schema=1, official_score=False, native_diagnostic=True,
                  machine=machine, trials=[], complete=False, script_sha256=digest(__file__),
                  limits='Build sandbox full-process diagnostic, not protected runtime score. Copied source-matched verifier; normal official five proofs remain separate.')
    report_path = work / 'report.json'

    def save():
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    save()
    try:
        research = solution / 'proof_generate' / 'research'
        provenance = json.loads((research / 'source_provenance.json').read_text())
        for item in provenance['files']:
            path = research / item['path']
            assert path.resolve().is_relative_to(research.resolve()), 'provenance path escape'
            assert not path.is_symlink() and path.stat().st_size == item['bytes'], 'reference shape changed'
            assert digest(path) == item['sha256'], 'reference source hash mismatch'
        report['source_provenance_sha256'] = digest(research / 'source_provenance.json')
        report['reference_revisions'] = provenance['references']
        report['candidate_bitmat_sha256'] = digest(solution / 'labinius/crates/pcs/src/fields/bitmat.rs')
        home = work / 'cargo-home'
        home.mkdir()
        registry = Path(os.environ['CARGO_HOME']) / 'registry'
        for name in ('index', 'cache', 'src'):
            assert (registry / name).is_dir(), f'offline registry {name} missing'
            shutil.copytree(registry / name, home / 'registry' / name)
        roots = {}
        for key, name in [('A', 'reference8af'), ('R', 'reference126')]:
            roots[key] = work / name / 'solution'
            source_copy(research / name / 'solution', roots[key])
        roots['B'] = work / 'diag-candidate' / 'solution'
        source_copy(solution, roots['B'], candidate=True)
        shutil.copyfile(research / 'build_original.rs', roots['B'] / 'proof_generate' / 'build.rs')
        trusted = work / 'trusted'
        source_copy(research / 'trusted', trusted)
        manifests = {k: v / 'proof_generate' / 'Cargo.toml' for k, v in roots.items()}
        manifests['V'] = trusted / 'proof_verify' / 'Cargo.toml'
        env = dict(PATH='/opt/cargo/bin:/usr/bin:/bin', HOME=str(home), CARGO_HOME=str(home),
                   RUSTUP_HOME='/opt/rustup', CC='/usr/bin/gcc-13',
                   CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER='/usr/bin/gcc-13',
                   RUSTFLAGS='-C target-cpu=native', CARGO_BUILD_JOBS='1',
                   LABINIUS_COMPARE_CHILD='1')
        binaries = {}
        for key in ('A', 'R', 'B', 'V'):
            target = work / ('target-' + key)
            childenv = dict(env, CARGO_TARGET_DIR=str(target))
            binary = 'labinius-fast-verify' if key == 'V' else 'labinius-fast'
            execute(['/opt/cargo/bin/cargo', 'build', '--locked', '--offline', '--release',
                     '--jobs', '1', '--bin', binary, '--manifest-path', str(manifests[key])],
                    childenv, manifests[key].parent, logs, 'build-' + key, timeout=3600)
            binaries[key] = target / 'release' / binary
        report['binaries'] = {k: dict(sha256=digest(p), bytes=p.stat().st_size) for k, p in binaries.items()}
        save()
        emit('builds_complete', binaries=report['binaries'], **machine)
        orders = [('A', 'R', 'B'), ('R', 'B', 'A'), ('B', 'A', 'R'),
                  ('B', 'R', 'A'), ('A', 'B', 'R')]
        for trial, order in enumerate(orders):
            seed = os.urandom(32).hex()
            block = work / f'trial-{trial}'
            block.mkdir()
            table = block / 'witness-table.bin'
            runenv = dict(PATH='/usr/bin:/bin', HOME='/tmp', TMPDIR='/tmp', LABINIUS_SEED=seed)
            execute([str(binaries['V']), '--write-table', str(table)], runenv, block, logs, f'table-{trial}')
            assert table.stat().st_size == 268435456 and not table.is_symlink(), 'invalid table'
            table.chmod(0o444)
            record = dict(seed=seed, order=list(order), table_sha256=digest(table), samples={})
            report['trials'].append(record)
            save()
            for key in order:
                instance = block / key
                instance.mkdir()
                proof = instance / 'proof.bin'
                try:
                    sample = execute([str(binaries[key]), '--prove', str(proof), '--table', str(table)],
                                     runenv, instance, logs, f'prove-{trial}-{key}')
                except ProcessFailure as exc:
                    record['samples'][key] = exc.result
                    save()
                    raise
                record['samples'][key] = sample
                # Same-UID children could chmod a 0444 table. Check after the timed
                # process, before verification. This is not a read-only bwrap mount.
                sample['table_integrity_after_prove'] = table_integrity(table, record['table_sha256'])
                save()
                assert sample['table_integrity_after_prove']['unchanged'], 'timed input table changed'
                st = proof.lstat()
                assert stat.S_ISREG(st.st_mode) and 0 < st.st_size <= LIMIT, 'invalid proof shape'
                sample.update(proof_bytes=st.st_size, proof_sha256=digest(proof))
                verify = execute([str(binaries['V']), '--verify', str(proof)], runenv, instance,
                                 logs, f'verify-{trial}-{key}')
                sample['verified_by_copied_reference'] = verify['returncode'] == 0
                verified = json.loads((logs / f'verify-{trial}-{key}.stdout').read_text())
                assert verified['field'] == 'GHASH' and verified['polynomialSize'] == 1 << 31 and verified['packedElements'] == 1 << 24, 'verifier shape mismatch'
                sample['verification'] = verified
                save()
            proofs = [block / k / 'proof.bin' for k in ('A', 'R', 'B')]
            assert proofs[0].read_bytes() == proofs[1].read_bytes() == proofs[2].read_bytes(), 'proof byte mismatch'
            record['proofs_bitwise_equal'] = True
            emit('trial', trial=trial, seed=seed, order=list(order), samples=record['samples'])
            save()
        gains = {key: [(1 - t['samples']['B']['elapsed_ns'] / t['samples'][key]['elapsed_ns']) * 100
                       for t in report['trials']] for key in ('A', 'R')}
        report.update(complete=True, paired_gains_percent=gains,
                      median_paired_gains_percent={k: statistics.median(v) for k, v in gains.items()},
                      median_ns={k: statistics.median(t['samples'][k]['elapsed_ns'] for t in report['trials']) for k in ('A', 'R', 'B')})
        save()
        emit('summary', **report)
    except BaseException as exc:
        report['failure'] = repr(exc)
        if isinstance(exc, ProcessFailure):
            report['failed_process'] = exc.result
        save()
        emit('failure', error=repr(exc), completed_trials=len(report['trials']))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.solution.resolve(), args.out.resolve())
