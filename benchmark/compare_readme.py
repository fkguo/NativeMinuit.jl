#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Collect a frozen five-pair, five-case MIGRAD comparison; preserve failed rows."""
import argparse, copy, datetime, hashlib, json, math, os, pathlib, statistics, subprocess, time

def atomic(path, obj):
    path=pathlib.Path(path); temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)

def sha(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def source_hashes(root):
    root=pathlib.Path(root)
    paths=[root/'Project.toml',root/'benchmark/Project.toml',root/'benchmark/Manifest.toml',root/'benchmark/fixtures/migrad_inputs.json',root/'benchmark/bench_migrad_suite.jl',root/'benchmark/readme_julia_bench.jl',root/'benchmark/compare_readme.py',root/'benchmark/cpp/cpp_readme_bench.cxx',root/'benchmark/cpp/CMakeLists.txt']
    for folder in ['src','ext','reference/Minuit2_cpp/inc','reference/Minuit2_cpp/src']:
        paths += [p for p in (root/folder).rglob('*') if p.is_file()]
    return {str(p.relative_to(root)):sha(p) for p in sorted(set(paths))}

def delta(a,b,tol):
    aa=a if isinstance(a,list) else [a];bb=b if isinstance(b,list) else [b]
    if len(aa)!=len(bb) or not aa:return {'passed':False,'reason':'shape or empty'}
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in aa+bb):return {'passed':False,'reason':'nonfinite or nonnumeric'}
    diffs=[abs(x-y) for x,y in zip(aa,bb)]
    budgets=[max(tol['atol'],tol['rtol']*max(abs(x),abs(y))) for x,y in zip(aa,bb)]
    margins=[d/b if b else (0 if d==0 else math.inf) for d,b in zip(diffs,budgets)]
    return {'passed':all(d<=b for d,b in zip(diffs,budgets)),'max_abs':max(diffs),'max_budget_fraction':max(margins),'signed_deltas':[x-y for x,y in zip(aa,bb)],'tolerance':tol}

def row_usable(row,n,samples):
    if not isinstance(row,dict):return False
    r=row.get('result',{})
    fields=['params','errors','covariance','fval','edm']
    values=[]
    for k in fields:
        v=r.get(k);values.extend(v if isinstance(v,list) else [v])
    return (row.get('samples')==samples and row.get('evals')==1 and row.get('stable_outputs') is True
        and len(row.get('sample_times_ns',[]))==samples
        and all(isinstance(v,(int,float)) and math.isfinite(v) for v in values)
        and all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in row.get('sample_times_ns',[]))
        and len(r.get('params',[]))==n and len(r.get('errors',[]))==n and len(r.get('covariance',[]))==n*n
        and r.get('is_valid') is True and r.get('has_covariance') is True
        and r.get('above_max_edm') is False and r.get('reached_call_limit') is False
        and isinstance(r.get('nfcn'),int) and r.get('nfcn')==r.get('actual_calls') and r.get('nfcn')>0)

def compare_pair(j,c,protocol):
    result={}
    for name,n in zip(protocol['cases'],[4,2,10,2,10]):
        jr=[r for r in (j or {}).get('rows',[]) if r.get('name')==name]
        cr=[r for r in (c or {}).get('rows',[]) if r.get('name')==name]
        row={'julia_rows':jr,'cpp_rows':cr,'usable':len(jr)==len(cr)==1 and row_usable(jr[0],n,protocol['samples']) and row_usable(cr[0],n,protocol['samples'])}
        if row['usable']:
            a,b=jr[0]['result'],cr[0]['result']
            row['strict_checks']={k:delta(a[k],b[k],t) for k,t in protocol['strict_tolerances'].items()}
            row['strict_parity']=all(v['passed'] for v in row['strict_checks'].values())
            row['nfcn_delta']=a['nfcn']-b['nfcn']
            if name in protocol['rosenbrock_diagnostic_tolerances']:
                tol=protocol['rosenbrock_diagnostic_tolerances'][name]
                row['diagnostic_checks']={k:delta(a[k],b[k],t) for k,t in tol.items() if k!='nfcn_slack'}
                row['diagnostic_pass']=all(v['passed'] for v in row['diagnostic_checks'].values()) and abs(row['nfcn_delta'])<=tol['nfcn_slack']
            else:row['diagnostic_pass']=None
            row['julia_median_ns']=statistics.median(jr[0]['sample_times_ns'])
            row['cpp_median_ns']=statistics.median(cr[0]['sample_times_ns'])
            row['julia_over_cpp']=row['julia_median_ns']/row['cpp_median_ns']
        result[name]=row
    return result

def summarize(pairs,protocol):
    result={}
    for name in protocol['cases']:
        rows=[p[name] for p in pairs]
        usable=len(rows)==protocol['rounds'] and all(r['usable'] for r in rows)
        r={'complete_usable_pairs':sum(x['usable'] for x in rows),'usable':usable,'strict_parity':usable and all(x['strict_parity'] for x in rows),'equivalent_computation_speed_claim':False}
        if name in protocol['rosenbrock_diagnostic_tolerances']:
            r['diagnostic_pass']=usable and all(x['diagnostic_pass'] for x in rows)
            r['status']='DIAGNOSTIC_ONLY' if r['diagnostic_pass'] else 'FAIL'
        else:r['status']='STRICT_PARITY_PASS' if r['strict_parity'] else 'FAIL'
        if usable:
            ratios=[x['julia_over_cpp'] for x in rows]
            r.update({'julia_median_ns_by_pair':[x['julia_median_ns'] for x in rows], 'cpp_median_ns_by_pair':[x['cpp_median_ns'] for x in rows], 'julia_over_cpp_by_pair':ratios,'median_paired_ratio':statistics.median(ratios),'observed_ratio_range':[min(ratios),max(ratios)],'nfcn_delta_by_pair':[x['nfcn_delta'] for x in rows]})
            # Exact reproducibility of each implementation across independent processes.
            r['julia_identical_across_processes']=all(x['julia_rows'][0]['result']==rows[0]['julia_rows'][0]['result'] for x in rows)
            r['cpp_identical_across_processes']=all(x['cpp_rows'][0]['result']==rows[0]['cpp_rows'][0]['result'] for x in rows)
            if not r['julia_identical_across_processes'] or not r['cpp_identical_across_processes']:r['status']='FAIL'
        result[name]=r
    return {'schema_version':1,'cases':result,'claim_verified':False,'uncertainty':protocol['uncertainty'],'scope':'One machine, one pinned Julia environment, five fixed unbounded objectives, Strategy(0), warmed core MIGRAD only.'}

def run_child(command,path,env,timeout):
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        with path.with_suffix('.log').open('w') as log:
            p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=timeout)
        receipt={'command':command,'started_utc':start,'exit_code':p.returncode,'output':str(path)}
    except subprocess.TimeoutExpired:
        receipt={'command':command,'started_utc':start,'exit_code':None,'timeout':True,'output':str(path)}
    receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic(path.with_suffix('.receipt.json'),receipt)
    return json.loads(path.read_text()) if path.exists() and receipt.get('exit_code')==0 else None

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--cpp-binary',required=True);p.add_argument('--out',required=True);p.add_argument('--protocol',required=True);p.add_argument('--julia',default='julia');p.add_argument('--timeout',type=int,default=900)
    a=p.parse_args();root=pathlib.Path(a.source).resolve();out=pathlib.Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(pathlib.Path(a.protocol).read_text());atomic(out/'protocol.json',protocol)
    fixture=root/'benchmark/fixtures/migrad_inputs.json'
    if sha(fixture)!=protocol['fixture_sha256']:raise RuntimeError('fixture identity mismatch')
    original=json.loads(fixture.read_text());assert original['settings']['tol']==0.1 and original['settings']['maxfcn']==10000 and original['settings']['storage_level']==0
    values=original['gauss_2_100']+[x for r in original['gauss_10_1000'] for x in r]
    text=' '.join(format(x,'.17g') for x in values)+'\n';assert list(map(float,text.split()))==values
    transport=out/'gaussian-transport.txt';transport.write_text(text)
    before=source_hashes(root);binary_before=sha(a.cpp_binary);atomic(out/'source-hashes-before.json',before)
    env=os.environ.copy();env.update(JULIA_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    atomic(out/'execution.json',{'cpp_binary':str(pathlib.Path(a.cpp_binary).resolve()),'cpp_binary_sha256':binary_before,'fixture_sha256':sha(fixture),'transport_sha256':sha(transport),'protocol_sha256':sha(out/'protocol.json'),'child_environment':{k:env.get(k) for k in ['JULIA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','DYLD_LIBRARY_PATH','JULIA_DEPOT_PATH']}})
    pairs=[];deadline=time.monotonic()+a.timeout
    for i in range(1,protocol['rounds']+1):
        records={};order=['julia','cpp'] if i%2 else ['cpp','julia']
        for language in order:
            path=out/f'pair-{i}-{language}.json'
            command=[a.julia,'--startup-file=no',f'--project={root/"benchmark"}',str(root/'benchmark/readme_julia_bench.jl'),str(path),str(protocol['samples'])] if language=='julia' else [a.cpp_binary,str(transport),str(path),str(protocol['samples'])]
            remaining=deadline-time.monotonic()
            records[language]=run_child(command,path,env,min(180,remaining)) if remaining>0 else None
        pair=compare_pair(records.get('julia'),records.get('cpp'),protocol);pairs.append(pair)
        atomic(out/f'pair-{i}-comparison.json',pair)
        atomic(out/'summary.json',summarize(pairs,protocol))
        print(json.dumps({'pair':i,'usable':{k:v['usable'] for k,v in pair.items()},'strict_parity':{k:v.get('strict_parity') for k,v in pair.items()}}),flush=True)
        if not all(v['usable'] for v in pair.values()):break
    after=source_hashes(root);atomic(out/'source-hashes-after.json',after)
    frozen=before==after and binary_before==sha(a.cpp_binary)
    report=summarize(pairs,protocol);report['frozen_sources']=frozen
    report['measurement_complete']=frozen and all(r['usable'] for r in report['cases'].values())
    report['verdict']='PASS_WITH_DIAGNOSTIC_LIMITS' if report['measurement_complete'] and all(r['status']!='FAIL' for r in report['cases'].values()) else 'FAIL'
    atomic(out/'summary.json',report)
    return 0 if report['verdict']=='PASS_WITH_DIAGNOSTIC_LIMITS' else 1
if __name__=='__main__':raise SystemExit(main())
