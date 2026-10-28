"""Validate two fixed candidate A jobs from source-bound native records and official results.

A PASS here applies only to candidate-a-native-v1. This is a new parser.
Job evidence JSON must bind job_id, conclusion, records_sha256, score_sha256,
solution_sha256, protected_sha256, and submitted source_commit from the actual job.
The unchanged protected script supplies verifier success on a successful five-run job.
"""
import argparse
import hashlib
import json
import math
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

from views import expected_views

SCHEMA='candidate-a-native-v1'
ROLES=('best-control','candidate-a')
Q='40772829184'
PROOF_BYTES=13958612
TABLE_BYTES=268435456
GATES=('identity','source','role','timer','unit','context','proof','runtime')
KEY_BYTES={'key.bin':0,'tail-key.bin':0,'hybrid-key.bin':0,'pair-key.bin':0,
           'sms-pairs.bin':42467328,'sms-mur.bin':145152,'sms-mu.bin':2592}
CHANGED={'simd/ntt/gen_small.rs','simd/sms_commit.rs','simd/sms_slot_gen.rs'}
FIELDS={
 'begin':{'case','cpu','workers','mode','scheduler','parent_pool','table_bytes','modulus','calls','guards','pairs','rustflags','encoded_rustflags'},
 'artifact':{'namespace','name','path','bytes','sha256'},
 'binding':{'namespace','kind','name','path','bytes','sha256'},
 'correctness':{'case','test','exit','tests','pass','stdout'},
 'command':{'call','role','executable','proof','table','seed','environment','args'},
 'child':{'call','role','phase','pair','order','position','timer','unit','outer_ns','exit','profile','stderr','stdout_sha256','stderr_sha256'},
 'proof':{'call','role','reference','bytes','modulus','sha256','equal'},
 'pair':{'pair','order','a_ns','b_ns','delta_ns'},
 'complete':{'calls','guards','pairs','correctness_targets'},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'duplicate JSON field: '+key)
        result[key]=value
    return result


def loads(text):
    def nonfinite(value):
        raise ValueError('nonfinite JSON value: '+value)
    return json.loads(text,object_pairs_hook=unique,parse_constant=nonfinite)


def hash_value(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value),'invalid SHA-256')


def integer(value, positive=False):
    require(type(value) is int and (value>0 if positive else value>=0),'invalid integer unit or count')


def read_records(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if 'candidate-a-json ' in line:
            line=line.split('candidate-a-json ',1)[1]
        elif not line.startswith('{'):
            continue
        row=loads(line)
        require(type(row) is dict,'record must be an object')
        require(row.get('schema')==SCHEMA,'unknown schema')
        event=row.get('event')
        require(event in FIELDS,'unknown event')
        require(set(row)==FIELDS[event]|{'schema','seq','event'},'missing or extra record fields')
        require(type(row['seq']) is int and row['seq']==len(rows)+1,'missing, repeated, or out of order event')
        rows.append(row)
    require(bool(rows) and rows[0]['event']=='begin' and rows[-1]['event']=='complete','incomplete job record')
    return rows


def schedule():
    result=[(ROLES[0],'reference',0,'',1),(ROLES[1],'validation',0,'',2)]
    for round in range(18):
        order='AB' if round%2==0 else 'BA'
        for position in range(2):
            role=position if round%2==0 else 1-position
            result.append((ROLES[role],'warmup' if round<2 else 'sample',round+1 if round<2 else round-1,order,position+1))
    return result


def profile(stderr):
    lines=[line for line in stderr.splitlines() if line.startswith('labinius-profile ')]
    require(len(lines)==1,'one complete production profile required')
    fields=unique(token.split('=',1) for token in lines[0].split()[1:])
    numeric={'load_us','matrix_us','setup_us','commit_us','recombine_us','partials_us','switch_us','row_us','challenges_us','fold_us','encode_us','total_us'}
    arrays={'sms_end_us','sms_groups','fold_end_us','fold_tasks'}
    require(set(fields)==numeric|arrays|{'threads','fold_gfni','sms_decoupled','sms_decoupled_at'},'current 20-field profile required')
    for name in numeric|{'threads','sms_decoupled'}:
        require(re.fullmatch('[0-9]+',fields[name]),'noninteger profile field')
    require(fields['threads']=='8' and fields['fold_gfni'] in ('true','false'),'production worker context')
    for name in arrays:
        values=fields[name].split('/')
        require(len(values)==8 and all(re.fullmatch('[0-9]+',v) for v in values),'eight worker report values required')
    at=fields['sms_decoupled_at'].split('/') if fields['sms_decoupled_at'] else []
    require(len(at)==min(int(fields['sms_decoupled']),16),'decoupling report count')
    require(all(re.fullmatch('[0-7]@[0-9]+',v) for v in at),'decoupling report format')
    for prefix,names in [('labinius-clock ',{'main_ns','exit_ns'}),('labinius-output ',{'prove_return_us','write_us'})]:
        selected=[line for line in stderr.splitlines() if line.startswith(prefix)]
        require(len(selected)==1,'one clock/output report required')
        values=unique(token.split('=',1) for token in selected[0].split()[1:])
        require(set(values)==names and all(re.fullmatch('[0-9]+',v) for v in values.values()),'clock/output schema')
        if prefix=='labinius-clock ':
            require(int(values['exit_ns'])>=int(values['main_ns']),'main/exit clock order')
    return lines[0]


def source_check(solution,context,rows):
    guard=Path(solution)/'native_diagnostics/frozen-source.tsv'
    require(sha(guard.read_bytes())==context['frozen_solution_sha256'],'frozen source context mismatch')
    source={}
    for line in guard.read_text().splitlines():
        name,size,digest=line.split('\t')
        data=(Path(solution)/name).read_bytes()
        require(len(data)==int(size) and sha(data)==digest,'local frozen source changed: '+name)
        source[name]=(int(size),digest)
    candidate=''.join(f'{name[len("labinius/"):]}\t{size}\t{digest}\n' for name,(size,digest) in sorted(source.items()) if name.startswith('labinius/'))
    control=[]
    for name,(size,digest) in sorted(source.items()):
        if not name.startswith('labinius/'):
            continue
        rel=name[len('labinius/'):]
        for changed in CHANGED:
            if rel=='crates/pcs/src/'+changed:
                size,digest=source['native_diagnostics/control/'+Path(changed).name]
        control.append(f'{rel}\t{size}\t{digest}\n')
    library={'candidate-library':sha(candidate.encode()),'control-library':sha(''.join(control).encode())}
    keys={}
    for namespace in ('performance','focused','round-trip'):
        bindings=[r for r in rows if r['event']=='binding' and r['namespace']==namespace]
        by_key={(r['kind'],r['name']):r for r in bindings}
        require(len(by_key)==len(bindings),'duplicate source binding')
        require(by_key[('source','frozen-solution')]['sha256']==context['frozen_solution_sha256'],'unbound source snapshot')
        for name,digest in library.items():
            require(by_key[('source',name)]['sha256']==digest,'effective full library differs from frozen view')
        expected=expected_views(solution,bindings,namespace=='focused')
        views={r['name']:r for r in bindings if r['kind']=='view'}
        require(set(views)==set(expected),'missing or extra generated view')
        wanted={('source',name) for name in ('frozen-solution','candidate-library','control-library')}|{('view',name) for name in expected}|{(kind,name) for kind in ('key','key-parent') for name in KEY_BYTES}
        require(set(by_key)==wanted,'missing or extra source binding')
        for name,data in expected.items():
            require(views[name]['sha256']==sha(data) and views[name]['bytes']==len(data),'generated view bytes differ from fixed sources: '+name)
        for name,size in KEY_BYTES.items():
            image=by_key[('key',name)]; parent=by_key[('key-parent',name)]
            require(image['bytes']==parent['bytes']==size and image['sha256']==parent['sha256'],'key is not the actual parent image')
            require(name not in keys or keys[name]==image['sha256'],'key image changed between targets')
            keys[name]=image['sha256']
        artifacts={r['name']:r for r in rows if r['event']=='artifact' and r['namespace']==namespace}
        needed={'native-link-metadata','native-c-archive','native-labrador-archive','native-logq48-stamp','bindings'}
        needed|={'driver','best-control','candidate-a','build-stdout','build-stderr','table'} if namespace=='performance' else {'test-executable','command','stdout','stderr'}
        require(needed<=set(artifacts),'native or executable evidence missing')
        require(any(name.startswith('native-object-') for name in artifacts),'actual native objects are unbound')
        require(artifacts['native-logq48-stamp']['bytes']==2 and artifacts['native-logq48-stamp']['sha256']==sha(b'48'),'actual current C LOGQ stamp is unbound')
        require('candidate-a-'+namespace+'-labrador' in artifacts['native-labrador-archive']['path'],'stale native archive path')
        require('candidate-a-'+namespace+'-target' in artifacts['native-c-archive']['path'],'stale C object target')
    production={r['name']:r for r in rows if r['event']=='artifact' and r['namespace']=='production-build'}
    require(production and any(name.endswith('-metadata') for name in production) and any('-object-' in name for name in production),'outer production C builds are unbound')
    for name,row in production.items():
        if name.endswith('-logq48'):
            require(row['bytes']==2 and row['sha256']==sha(b'48'),'outer C stamp differs from current native source')
    return keys


def median(values):
    values=sorted(values)
    n=len(values)
    return Fraction(values[n//2]) if n%2 else Fraction(values[n//2-1]+values[n//2],2)


def numeric(deltas,orders):
    require(len(deltas)==16 and Counter(orders)=={'AB':8,'BA':8},'sixteen balanced sample pairs required')
    values={'median_ns':median(deltas),'mean_ns':Fraction(sum(deltas),16),
            'ab_median_ns':median([v for v,o in zip(deltas,orders) if o=='AB']),
            'ba_median_ns':median([v for v,o in zip(deltas,orders) if o=='BA'])}
    wins=sum(v>0 for v in deltas)
    return all(v>0 for v in values.values()) and wins>=12, {**{k:str(v) for k,v in values.items()},'wins':wins}


def official(score,evidence,context,identity,label,records_bytes,score_bytes):
    require(evidence['conclusion']=='success','protected job did not succeed')
    integer(evidence['job_id'],True)
    wanted={row['canonical_job']:row['job_id'] for row in identity['jobs']}
    require(evidence['job_id']==wanted[label],'official job is not the preselected job')
    require(evidence['source_commit']==identity['source_commit'] and evidence['source_tree']==identity['source_tree'],'submitted source head/tree differs from preselected source')
    require(evidence['records_sha256']==sha(records_bytes) and evidence['score_sha256']==sha(score_bytes),'job does not bind actual record/score files')
    require(evidence['solution_sha256']==context['frozen_solution_sha256'],'protected job source mismatch')
    require(evidence['protected_sha256']==context['protected_sha256'],'protected source changed')
    require(context['fresh_seed_source']=='  LABINIUS_SEED=$(openssl rand -hex 32)' and context['fresh_seed_source_sha256']==sha(context['fresh_seed_source'].encode()),'fresh-seed source is unbound')
    runs=score['runs']; require(len(runs)==5,'five protected proofs required')
    times=[]
    for i,run in enumerate(runs,1):
        require(run['run']==i and run['proofBytes']==PROOF_BYTES,'protected proof count/shape')
        value=run['provingTimeMs']; require(type(value) in (int,float) and math.isfinite(value) and value>0,'invalid official duration')
        times.append(value)
    require(score['score']==sorted(times)[2],'official median mismatch')
    metrics=score['metrics']
    require(metrics['modulus']==Q and metrics['nproc']==8 and metrics['avx512f'] is True and 'Xeon(R) Platinum 8488C' in metrics['cpuModel'],'official runtime context mismatch')
    require(metrics['proofBytes']==PROOF_BYTES and metrics['packedElements']==16777216 and metrics['polynomialSize']==2147483648,'official shape mismatch')
    return evidence['job_id']


def validate_job(solution,context,identity,label,records_path,score_path,evidence_path):
    require(label in context['jobs'],'unselected canonical job')
    raw=Path(records_path).read_bytes();score_raw=Path(score_path).read_bytes()
    rows=read_records(records_path);score=loads(score_raw);evidence=loads(Path(evidence_path).read_bytes())
    require(Counter(r['event'] for r in rows)['begin']==1 and Counter(r['event'] for r in rows)['complete']==1,'duplicate job envelope')
    for r in rows:
        for name in {'call','pair','position','exit','workers','mode','calls','guards','pairs','tests','table_bytes','correctness_targets'}&set(r):
            integer(r[name])
        if 'delta_ns' in r:
            require(type(r['delta_ns']) is int,'delta must be a signed integer')
        if r['event'] in ('artifact','binding'):
            integer(r['bytes']);hash_value(r['sha256'])
            require(type(r['path']) is str and r['path'].startswith('/'),'unbound artifact path')
    namespaces={'performance','focused','round-trip','production-build','children'}
    for row in rows:
        if row['event'] in ('binding','artifact'):
            require(row['namespace'] in namespaces,'extra artifact namespace')
            if row['event']=='binding':
                require(row['namespace'] in ('performance','focused','round-trip'),'extra source namespace')
            elif row['namespace']=='children':
                require(re.fullmatch('child-(0[1-9]|[12][0-9]|3[0-8])-(stdout|stderr)',row['name']),'extra child artifact')
            elif row['namespace']=='production-build':
                require(re.fullmatch('native-[0-9]+-(metadata|c-archive|labrador-archive|logq48|object-[0-9]+)',row['name']),'extra production artifact')
            else:
                common={'native-link-metadata','native-c-archive','native-labrador-archive','native-logq48-stamp','bindings'}
                allowed=common|({'driver','best-control','candidate-a','build-stdout','build-stderr','table'} if row['namespace']=='performance' else {'test-executable','command','stdout','stderr'})
                require(row['name'] in allowed or re.fullmatch('native-object-[0-9]+',row['name']),'extra target artifact')
    keys=source_check(solution,context,rows)
    begin=rows[0];complete=rows[-1]
    require(begin['case']=='candidate-a-whole-process' and begin['workers']==8 and begin['mode']==0 and begin['scheduler']=='production-main' and begin['parent_pool']=='absent','incorrect production context')
    require(begin['table_bytes']==TABLE_BYTES and begin['modulus']==Q and 'Xeon(R) Platinum 8488C' in begin['cpu'],'incorrect native shape/hardware')
    require('target-cpu=native' in begin['rustflags']+begin['encoded_rustflags'],'missing native compiler context')
    require(all(begin[k]==v for k,v in {'calls':38,'guards':37,'pairs':16}.items()),'changed fixed call budget')
    require(all(complete[k]==v for k,v in {'calls':38,'guards':37,'pairs':16,'correctness_targets':2}.items()),'incomplete fixed work')
    correct=[r for r in rows if r['event']=='correctness']
    require([(r['case'],r['test']) for r in correct]==[('focused','candidate_a_checks::inverse_9721_and_sms_equivalence'),('round-trip','tests::round_trip_and_reject_wrong_instance')],'wrong correctness targets')
    artifacts={(r['namespace'],r['name']):r for r in rows if r['event']=='artifact'}
    require(len(artifacts)==sum(r['event']=='artifact' for r in rows),'duplicate artifact role')
    require(all(r['exit']==0 and r['tests']==1 and r['pass'] is True for r in correct),'native correctness failed or skipped')
    for row in correct:
        summaries=[line for line in row['stdout'].splitlines() if line.startswith('test result:')]
        named='test '+row['test']+' ... ok'
        require(len(summaries)==1 and summaries[0].startswith('test result: ok. 1 passed; 0 failed;') and row['stdout'].splitlines().count(named)==1,'one exact named native success and one summary required')
        require(sha(row['stdout'].encode())==artifacts[(row['case'],'stdout')]['sha256'],'unbound native correctness output')
    commands=[r for r in rows if r['event']=='command'];children=[r for r in rows if r['event']=='child'];proofs=[r for r in rows if r['event']=='proof']
    require(len(commands)==len(children)==len(proofs)==38,'missing, repeated, or extra production child')
    require(correct[-1]['seq']<commands[0]['seq'],'correctness must precede the performance namespace')

    paths=[];samples={};reference=proofs[0]['sha256']
    for call,(want,command,child,proof) in enumerate(zip(schedule(),commands,children,proofs),1):
        role,phase,pair,order,position=want
        require(command['call']==child['call']==proof['call']==call,'child call identity')
        require(command['role']==child['role']==proof['role']==role,'swapped child roles')
        require((child['phase'],child['pair'],child['order'],child['position'])==(phase,pair,order,position),'changed fixed AB/BA schedule')
        require(command['seq']<child['seq']<proof['seq'],'timer/proof event order')
        require(child['timer']=='child-output-outer' and child['unit']=='ns','wrong metric or unit')
        integer(child['outer_ns'],True); require(child['exit']==0,'production child failed')
        require(command['seed']=='51'*32 and command['args']=='--prove,proof,--table,table' and command['environment']=='HOME=/tmp;TMPDIR=/tmp;LABINIUS_SEED','changed production child context')
        require(command['executable']==artifacts[('performance',role)]['path'],'unbound executable role')
        require(command['table']==artifacts[('performance','table')]['path'],'unbound input table')
        require(profile(child['stderr'])==child['profile'],'incomplete current production profile')
        require(sha(child['stderr'].encode())==child['stderr_sha256']==artifacts[('children',f'child-{call:02}-stderr')]['sha256'],'unbound raw profile bytes')
        require(child['stdout_sha256']==artifacts[('children',f'child-{call:02}-stdout')]['sha256'],'unbound raw stdout')
        hash_value(proof['sha256'])
        require(proof['bytes']==PROOF_BYTES and proof['modulus']==Q and proof['sha256']==reference,'complete proof guard failed')
        require(proof['reference'] is (call==1) and proof['equal'] is (call!=1),'missing full proof byte check')
        paths.append(command['proof'])
        if phase=='sample': samples.setdefault(pair,{})[role]=child['outer_ns']
    require(len(set(paths))==38,'proof path reused')
    require(artifacts[('performance','table')]['bytes']==TABLE_BYTES,'incomplete input table')
    pair_rows=[r for r in rows if r['event']=='pair'];require(len(pair_rows)==16,'missing or extra sample pairs')
    deltas=[];orders=[]
    for i,row in enumerate(pair_rows,1):
        integer(row['a_ns'],True);integer(row['b_ns'],True)
        pair=samples[i];a,b=pair[ROLES[0]],pair[ROLES[1]];order='AB' if i%2 else 'BA'
        require((row['pair'],row['order'],row['a_ns'],row['b_ns'],row['delta_ns'])==(i,order,a,b,a-b),'pair summary differs from raw integer durations')
        deltas.append(a-b);orders.append(order)
    passed,statistics=numeric(deltas,orders)
    job_id=official(score,evidence,context,identity,label,raw,score_raw)
    return {'canonical_job':label,'job_id':job_id,'gates':{gate:(passed if gate=='runtime' else True) for gate in GATES},'numeric':statistics,'key_sha256':keys,'official_median_ms':score['score']}


def selected_identity(identity,context,context_bytes):
    require(set(identity)=={'schema','context_sha256','source_commit','source_tree','jobs'},'identity record schema')
    require(identity['schema']==SCHEMA and identity['context_sha256']==sha(context_bytes),'identity must bind the frozen context')
    require(type(identity['source_commit']) is str and re.fullmatch('[0-9a-f]{40}',identity['source_commit']),'preselected submitted head required')
    require(type(identity['source_tree']) is str and re.fullmatch('[0-9a-f]{40}',identity['source_tree']),'preselected submitted tree required')
    require(type(identity['jobs']) is list and len(identity['jobs'])==2,'two preselected official jobs required')
    require([row['canonical_job'] for row in identity['jobs']]==context['jobs'],'fixed canonical job roles required')
    for row in identity['jobs']:
        require(set(row)=={'canonical_job','job_id'},'identity job schema')
        integer(row['job_id'],True)
    require(len({row['job_id'] for row in identity['jobs']})==2,'official and companion must be different jobs')


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--solution',required=True)
    a.add_argument('--context',required=True)
    a.add_argument('--identity',required=True,help='fixed after request creation, before either job result/log read')
    a.add_argument('--job',nargs=4,action='append',metavar=('CANONICAL','RECORDS','SCORE','EVIDENCE'),required=True)
    x=a.parse_args();context_bytes=Path(x.context).read_bytes();context=loads(context_bytes)
    identity=loads(Path(x.identity).read_bytes());selected_identity(identity,context,context_bytes)
    require(context['schema']==SCHEMA and context['jobs']==['canonical-1','canonical-2'],'unselected job plan')
    require([job[0] for job in x.job]==context['jobs'],'exactly the two fixed jobs are required')
    results=[validate_job(x.solution,context,identity,*job) for job in x.job]
    require(len({r['job_id'] for r in results})==2,'same actual job used twice')
    require(results[0]['key_sha256']==results[1]['key_sha256'],'public images changed between canonical jobs')
    passed=all(all(r['gates'].values()) for r in results)
    print(json.dumps({'schema':SCHEMA,'verdict':'PASS' if passed else 'FAIL','jobs':results},indent=2))
    return 0 if passed else 1


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,KeyError,IndexError,TypeError,AssertionError) as error:
        print(json.dumps({'schema':SCHEMA,'verdict':'FAIL','gates':{gate:False for gate in GATES},'error':str(error)}))
        raise SystemExit(1)
