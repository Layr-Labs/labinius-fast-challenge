"""Bounded, nonfatal build-stage compiler/ELF observations; outside prover timers."""
import base64
import hashlib
import re
import subprocess


def assembly_metrics(text):
    instructions = []
    for line in text.splitlines():
        match = re.match(r'^\s*[0-9a-f]+:\s+([a-z][a-z0-9.]*)\s*(.*)$', line)
        if match:
            instructions.append((match[1], match[2]))
    fold = {op: {'instructions': 0, 'memory_operands': 0}
            for op in ('vpmaddwd', 'vptestmd', 'vpaddd')}
    extract = []
    for op, operands in instructions:
        parts = [part.strip() for part in operands.split(',')]
        selected = ((op == 'vpmaddwd' and len(parts) == 3
                     and parts[0] == 'zmm31' and parts[1] in ('zmm29', 'zmm30'))
                    or (op == 'vptestmd' and len(parts) == 3 and parts[0] == 'k1'
                        and parts[1] in ('zmm29', 'zmm30'))
                    or (op == 'vpaddd' and len(parts) == 3
                        and parts[0].replace(' ', '') == 'zmm31{k1}'
                        and parts[1] == 'zmm31'))
        if selected:
            fold[op]['instructions'] += 1
            fold[op]['memory_operands'] += int('[' in operands)
            extract.append(op + ' ' + operands)
    metrics = dict(instructions=len(instructions),
                   stack_memory_instructions=sum('[' in args and re.search(r'\b(?:rsp|rbp)\b', args) is not None
                                                 for _, args in instructions),
                   calls=sum(op in ('call', 'callq') for op, _ in instructions),
                   frame_subtractions=[args for op, args in instructions
                                       if op == 'sub' and args.startswith('rsp,')],
                   prologue=[op + ' ' + args for op, args in instructions[:12]],
                   pair_fold=fold)
    metrics['pair_fold_classified'] = all(item['instructions'] == 54 for item in fold.values())
    return metrics, '\n'.join(extract)


def collect(env, cwd, logs, binaries, emit):
    report = dict(official_score=False, tool_calls=[], toolchain={}, elf_codegen={},
                  limits='Diagnostic R/B ELF only. Selected objdump stdout is hashed, saved privately and copied in base64 log chunks. This is function disassembly, not the raw ELF or normalized machine-code identity. No runtime or score changes.')

    def command(argv, label):
        out = logs / (label + '.stdout')
        err = logs / (label + '.stderr')
        item = dict(argv=[str(x) for x in argv], label=label,
                    stdout=out.name, stderr=err.name)
        try:
            with out.open('wb') as stdout, err.open('wb') as stderr:
                result = subprocess.run(argv, env=env, cwd=cwd, stdout=stdout,
                                        stderr=stderr, timeout=30, check=False)
            item.update(returncode=result.returncode, stdout_bytes=out.stat().st_size,
                        stdout_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
            text = out.read_text(errors='replace')
        except Exception as exc:
            item['error'] = repr(exc)
            text = ''
        report['tool_calls'].append(item)
        return item, text

    try:
        for name, argv in [('rustc', ['/opt/cargo/bin/rustc', '-vV']),
                           ('cargo', ['/opt/cargo/bin/cargo', '-V'])]:
            item, text = command(argv, 'observe-' + name)
            report['toolchain'][name] = dict(item, text_prefix=text[:4096], truncated=len(text) > 4096)
        for key in ('R', 'B'):
            _, nm = command(['/usr/bin/nm', '-S', '--defined-only', '--no-demangle', str(binaries[key])],
                            'observe-nm-' + key)
            symbols = []
            for line in nm.splitlines():
                parts = line.split()
                if len(parts) == 4 and parts[2] in ('t', 'T'):
                    name = parts[3]
                    if ('ntt_bin_batch32_pair' in name or 'partials_row' in name):
                        symbols.append((name, int(parts[1], 16)))
            report['elf_codegen'][key] = dict(matching_symbols=len(symbols),
                                               symbols_truncated=len(symbols) > 8, functions=[])
            for index, (name, size) in enumerate(symbols[:8]):
                item, text = command(['/usr/bin/objdump', '-d', '-M', 'intel', '--no-show-raw-insn',
                                      '--disassemble=' + name, str(binaries[key])],
                                     f'observe-asm-{key}-{index}')
                metrics, extract = assembly_metrics(text)
                entry = dict(symbol=name, symbol_bytes=size, observation=item, **metrics)
                raw = (logs / item['stdout']).read_bytes() if item.get('returncode') == 0 else b''
                chunks = (len(raw) + 12287) // 12288
                entry['asm_transport'] = dict(encoding='base64', chunks=chunks,
                                              bytes=len(raw), sha256=item.get('stdout_sha256'))
                report['elf_codegen'][key]['functions'].append(entry)
                emit('native_codegen', binary=key, **entry,
                     text_prefix=text[:1024], text_prefix_truncated=len(text) > 1024,
                     fold_extract=extract, fold_extract_complete=True)
                for index in range(chunks):
                    emit('native_asm_chunk', binary=key, symbol=name, index=index,
                         chunks=chunks, bytes=len(raw), sha256=item['stdout_sha256'],
                         base64=base64.b64encode(raw[12288 * index:12288 * (index + 1)]).decode('ascii'))
    except Exception as exc:
        report['error'] = repr(exc)
    report['observability_tool_calls'] = len(report['tool_calls'])
    complete = (not report.get('error')
                and all(item.get('returncode') == 0 for item in report['tool_calls'])
                and all(report['elf_codegen'].get(key, {}).get('functions')
                        and not report['elf_codegen'][key]['symbols_truncated']
                        and sum(item['pair_fold_classified']
                                for item in report['elf_codegen'][key]['functions']) == 2
                        for key in ('R', 'B')))
    report['observation_status'] = 'complete' if complete else 'partial_or_unavailable'
    emit('native_observations', **report)
    return report
