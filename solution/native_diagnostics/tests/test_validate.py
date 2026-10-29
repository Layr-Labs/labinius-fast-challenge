"""Invalid-record checks for the new schema; these are not native proof results."""
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import validate as v


def envelope():
    common={'schema':v.SCHEMA}
    begin={**common,'seq':1,'event':'begin','case':'candidate-b-whole-process','cpu':'Intel(R) Xeon(R) Platinum 8488C',
           'workers':8,'mode':0,'scheduler':'production-main','parent_pool':'absent','table_bytes':v.TABLE_BYTES,
           'modulus':v.Q,'calls':38,'guards':37,'pairs':16,'rustflags':'-C target-cpu=native','encoded_rustflags':''}
    complete={**common,'seq':2,'event':'complete','calls':38,'guards':37,'pairs':16,'correctness_targets':2}
    return [begin,complete]


def valid_profile():
    numeric='load_us matrix_us setup_us commit_us recombine_us partials_us switch_us row_us challenges_us fold_us encode_us total_us'.split()
    text='labinius-profile '+' '.join(name+'=1' for name in numeric)
    text+=' threads=8 fold_gfni=true sms_end_us=1/1/1/1/1/1/1/1 sms_groups=16/16/16/16/16/16/16/16 sms_decoupled=0 sms_decoupled_at='
    text+=' fold_end_us=1/1/1/1/1/1/1/1 fold_tasks=32/32/32/32/32/32/32/32\n'
    return text+'labinius-output prove_return_us=1 write_us=1\nlabinius-clock main_ns=1 exit_ns=2\n'


def protocol_rows():
    rows=envelope()[:1]
    def add(event,**fields): rows.append({'schema':v.SCHEMA,'seq':len(rows)+1,'event':event,**fields})
    for role in v.ROLES: add('artifact',namespace='performance',name=role,path='/bin/'+role,bytes=1,sha256='a'*64)
    add('artifact',namespace='performance',name='table',path='/table',bytes=v.TABLE_BYTES,sha256='b'*64)
    for case,test in [('focused','candidate_b_checks::row_equivalence'),('round-trip','tests::round_trip_and_reject_wrong_instance')]:
        stdout='test '+test+' ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 1s\n'
        add('artifact',namespace=case,name='stdout',path='/correct/'+case+'/stdout',bytes=len(stdout),sha256=v.sha(stdout.encode()))
        add('correctness',case=case,test=test,exit=0,tests=1,pass_=True,stdout=stdout)
        rows[-1]['pass']=rows[-1].pop('pass_')
    times={}
    for call,(role,phase,pair,order,position) in enumerate(v.schedule(),1):
        stderr=valid_profile()
        for stream,content in [('stdout',''),('stderr',stderr)]:
            add('artifact',namespace='children',name=f'child-{call:02}-{stream}',path=f'/raw/{call}/{stream}',bytes=len(content),sha256=v.sha(content.encode()))
        add('command',call=call,role=role,executable='/bin/'+role,proof=f'/proof-{call}',table='/table',seed='51'*32,environment='HOME=/tmp;TMPDIR=/tmp;LABINIUS_SEED',args='--prove,proof,--table,table')
        ns=200 if role==v.ROLES[0] else 100
        add('child',call=call,role=role,phase=phase,pair=pair,order=order,position=position,timer='child-output-outer',unit='ns',outer_ns=ns,exit=0,profile=stderr.splitlines()[0],stderr=stderr,stdout_sha256=v.sha(b''),stderr_sha256=v.sha(stderr.encode()))
        add('proof',call=call,role=role,reference=call==1,bytes=v.PROOF_BYTES,modulus=v.Q,sha256='c'*64,equal=call!=1)
        if phase=='sample':
            times.setdefault(pair,{})[role]=ns
            if position==2:
                add('pair',pair=pair,order=order,a_ns=times[pair][v.ROLES[0]],b_ns=times[pair][v.ROLES[1]],delta_ns=100)
    add('complete',calls=38,guards=37,pairs=16,correctness_targets=2)
    return rows


class InvalidRecords(unittest.TestCase):
    def read(self,rows):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'records.jsonl'
            p.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            return v.read_records(p)

    def test_current_empty_decoupling_report_is_valid(self):
        self.assertTrue(v.profile(valid_profile()).endswith('fold_tasks=32/32/32/32/32/32/32/32'))

    def test_reject_duplicate_json_keys(self):
        with self.assertRaises(ValueError): v.loads('{"outer_ns":1,"outer_ns":2}')

    def test_reject_nonfinite_values(self):
        for value in ('NaN','Infinity','-Infinity'):
            with self.subTest(value=value),self.assertRaises(ValueError): v.loads('{"outer_ns":'+value+'}')

    def test_reject_missing_extra_and_repeated_events(self):
        cases=[]
        missing=envelope();del missing[0]['mode'];cases.append(missing)
        extra=envelope();extra[0]['tolerance_ns']=1;cases.append(extra)
        repeat=envelope();repeat.insert(1,copy.deepcopy(repeat[0]));cases.append(repeat)
        cases.append(envelope()[:-1])
        for rows in cases:
            with self.subTest(rows=rows),self.assertRaises(ValueError): self.read(rows)

    def test_reject_old_profile_missing_worker_fields(self):
        old=' '.join(valid_profile().splitlines()[0].split()[:15])+'\n'+ '\n'.join(valid_profile().splitlines()[1:])
        with self.assertRaises(ValueError): v.profile(old)

    def test_reject_duplicate_profile_and_bad_empty_report(self):
        for raw in (valid_profile()+valid_profile().splitlines()[0]+'\n',
                    valid_profile().replace('sms_decoupled=0','sms_decoupled=1'),
                    valid_profile().replace('threads=8','threads=4')):
            with self.subTest(raw=raw),self.assertRaises(ValueError): v.profile(raw)

    def test_integer_units_do_not_accept_float_or_boolean(self):
        for value in (1.0,True,float('inf'),0,-1):
            with self.subTest(value=value),self.assertRaises(ValueError): v.integer(value,True)

    def test_fixed_numeric_rules_reject_boundary_cases(self):
        orders=['AB','BA']*8
        self.assertTrue(v.numeric([1]*16,orders)[0])
        self.assertFalse(v.numeric([0]*16,orders)[0])
        # Median, mean and both order medians pass, but only eleven pairs win.
        self.assertFalse(v.numeric([4]*11+[-1]*5,orders)[0])
        # Twelve wins and a positive median, but a negative mean.
        self.assertFalse(v.numeric([1]*12+[-10]*4,orders)[0])
        # Twelve wins, positive overall mean/median, but AB median equals zero.
        bad=[-4 if i%2==0 and i<8 else 4 for i in range(16)]
        self.assertEqual(sum(x>0 for x in bad),12)
        self.assertEqual(v.median(bad[::2]),0)
        self.assertGreater(v.median(bad[1::2]),0)
        self.assertFalse(v.numeric(bad,orders)[0])
        mirrored=[-4 if i%2==1 and i<8 else 4 for i in range(16)]
        self.assertEqual(sum(x>0 for x in mirrored),12)
        self.assertGreater(v.median(mirrored[::2]),0)
        self.assertEqual(v.median(mirrored[1::2]),0)
        self.assertFalse(v.numeric(mirrored,orders)[0])
        with self.assertRaises(ValueError):v.numeric([1]*16,['AB']*16)
        with self.assertRaises(ValueError):v.numeric([1]*17,orders+['AB'])

    def test_large_ns_keep_exact_integer_sign(self):
        huge=2**80
        passed,stats=v.numeric([huge+1]*12+[-huge]*4,['AB','BA']*8)
        self.assertTrue(passed)
        self.assertEqual(stats['mean_ns'],str((12*(huge+1)-4*huge)//16) if (12*(huge+1)-4*huge)%16==0 else str(v.Fraction(12*(huge+1)-4*huge,16)))

    def validate_protocol(self,rows):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'records').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            (d/'score').write_text('{"score":1}');(d/'job').write_text('{}')
            with patch.object(v,'source_check',return_value={}),patch.object(v,'official',return_value=101):
                return v.validate_job(d,{'jobs':['canonical-1']},{},'canonical-1',d/'records',d/'score',d/'job')

    def test_full_protocol_baseline_is_valid_without_native_claim(self):
        self.assertTrue(all(self.validate_protocol(protocol_rows())['gates'].values()))

    def test_reject_shared_malformed_proof_hash(self):
        rows=protocol_rows()
        for row in rows:
            if row['event']=='proof':row['sha256']='not-a-hash'
        with self.assertRaises(ValueError):self.validate_protocol(rows)

    def test_reject_duplicate_or_wrong_named_native_success(self):
        for kind in ('duplicate-summary','wrong-name'):
            rows=protocol_rows();correct=next(row for row in rows if row['event']=='correctness')
            if kind=='duplicate-summary':correct['stdout']+=correct['stdout'].splitlines()[-1]+'\n'
            else:correct['stdout']=correct['stdout'].replace(correct['test'],'other_test')
            # Bind the modified bytes so only the exact-success rule can reject them.
            for row in rows:
                if row['event']=='artifact' and row['namespace']==correct['case'] and row['name']=='stdout':
                    row['sha256']=v.sha(correct['stdout'].encode());row['bytes']=len(correct['stdout'])
            with self.subTest(kind=kind),self.assertRaises(ValueError):self.validate_protocol(rows)

    def test_identity_rejects_unselected_job_head_and_context(self):
        context={'jobs':['canonical-1','canonical-2']}
        raw=json.dumps(context).encode()
        identity={'schema':v.SCHEMA,'context_sha256':v.sha(raw),'source_commit':'1'*40,'source_tree':'2'*40,
                  'jobs':[{'canonical_job':'canonical-1','job_id':101},{'canonical_job':'canonical-2','job_id':102}]}
        v.selected_identity(identity,context,raw)
        for field,value in [('context_sha256','0'*64),('source_commit','bad'),('source_tree','bad')]:
            changed=copy.deepcopy(identity);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):v.selected_identity(changed,context,raw)
        for value in (True,101.0,101):
            changed=copy.deepcopy(identity);changed['jobs'][1]['job_id']=value
            with self.subTest(value=value),self.assertRaises(ValueError):v.selected_identity(changed,context,raw)
        evidence={'conclusion':'success','job_id':999,'source_commit':'1'*40,'source_tree':'2'*40}
        with self.assertRaises(ValueError):v.official({},evidence,context,identity,'canonical-1',b'',b'')
        evidence['job_id']=101;evidence['source_commit']='3'*40
        with self.assertRaises(ValueError):v.official({},evidence,context,identity,'canonical-1',b'',b'')

    def test_view_preserves_private_crlf_bytes(self):
        from views import expected_views
        solution=Path(__file__).resolve().parents[2]
        selected=[{'kind':'view','name':'fixture-eval-raw','path':'/frozen/labinius/crates/pcs/src/eval.rs'},
                  {'kind':'view','name':'control-eval','path':'/out/control-eval.rs'}]
        generated=expected_views(solution,selected,False)
        for name in ('keygen','modulus_choice','parameters','rq'):
            raw=(solution/'proof_generate/src'/f'{name}.rs').read_bytes()
            self.assertIn(b'\r\n',raw)
            self.assertEqual(generated['control-'+name],raw.replace(b'labinius::',b'labinius_control::'))

    def test_open_or_mixed_shared_baseline_is_rejected(self):
        from freeze import SHARED, checked_baseline
        solution=Path(__file__).resolve().parents[2]
        choice=next(choice for choice,files in SHARED.items()
                    if all(v.sha((solution/'labinius/crates/pcs/src'/name).read_bytes())==digest
                           for name,digest in files.items()))
        receipt={'schema':'candidate-b-baseline-selection-v1','status':'FINAL',
                 'choice':choice,'verdict_sha256':'a'*64}
        self.assertRegex(checked_baseline(solution,receipt),'^[0-9a-f]{64}$')
        for key,value in [('status','OPEN'),('verdict_sha256','unobserved'),
                          ('choice',next(c for c in SHARED if c!=choice))]:
            changed={**receipt,key:value}
            with self.subTest(key=key),self.assertRaises(ValueError):
                checked_baseline(solution,changed)
        with self.assertRaises(ValueError):
            v.source_check(solution,{'execution_baseline':'OPEN'},[])

    def test_reject_role_timer_proof_and_pair_changes(self):
        # Source/native/official checks have separate gates. These mutations isolate
        # the protocol fields in the full 38-call parser, without running native code.
        mutations=[
            ('child','role','candidate-b'),('child','timer','total-us'),
            ('child','outer_ns',1.0),('child','exit',False),
            ('proof','bytes',v.PROOF_BYTES-1),('proof','sha256','d'*64),
            ('pair','a_ns',200.0),('pair','b_ns',True),('pair','delta_ns',101),('command','environment','LABINIUS_NORMAL_SCHED=1'),
        ]
        for event,key,value in mutations:
            rows=protocol_rows()
            # Change call 1 except the proof hash, where call 2 tests byte guards.
            selected=[r for r in rows if r['event']==event]
            row=selected[1] if event=='proof' and key=='sha256' else selected[0]
            row[key]=value
            with self.subTest(event=event,key=key),tempfile.TemporaryDirectory() as d:
                d=Path(d);(d/'records').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
                (d/'score').write_text('{}');(d/'job').write_text('{}')
                with patch.object(v,'source_check',return_value={}),patch.object(v,'official',return_value='unit-only-job'):
                    with self.assertRaises(ValueError):v.validate_job(d,{'jobs':['canonical-1']},{},'canonical-1',d/'records',d/'score',d/'job')

    def test_protected_failure_cannot_supply_verifier_success(self):
        context={'protected_sha256':{},'frozen_solution_sha256':'a'*64}
        with self.assertRaises(ValueError):v.official({}, {'conclusion':'failure'},context,{},'canonical-1',b'raw',b'score')


if __name__=='__main__':
    unittest.main()
