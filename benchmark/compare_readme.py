#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Collect a frozen five-pair, five-case MIGRAD comparison; preserve failed rows."""
import argparse, copy, datetime, hashlib, json, math, os, pathlib, statistics, subprocess, time

CASE_DIMENSIONS = dict(zip(
    ['quad_4d', 'rosenbrock_2d', 'rosenbrock_10d', 'gauss_ll_2_100', 'gauss_ll_10_1000'],
    [4, 2, 10, 2, 10]))

def number(value):
    return type(value) in (int, float) and math.isfinite(value)

def positive_integer(value):
    return type(value) is int and value > 0

def validate_protocol(protocol):
    """Reject unsupported solver contracts; honor configurable sampling limits."""
    if protocol.get('cases') != list(CASE_DIMENSIONS):
        raise ValueError('the five fixed cases and their order are required')
    for key, expected in {'strategy': 0, 'tol': 0.1, 'maxfcn': 10000,
                          'storage_level': 0, 'evals': 1, 'warmups': 5}.items():
        value = protocol.get(key)
        if not number(value) or value != expected:
            raise ValueError(f'unsupported {key}; expected {expected}')
    for key in ('samples', 'rounds'):
        if not positive_integer(protocol.get(key)):
            raise ValueError(f'{key} must be a positive integer')
    order = protocol.get('order')
    if not isinstance(order, list) or len(order) != protocol['rounds'] or any(
            item not in ('julia,cpp', 'cpp,julia') for item in order):
        raise ValueError('order must declare both languages once in every round')
    if any(protocol.get('threads', {}).get(key) != 1 or
           type(protocol['threads'][key]) is not int
           for key in ('julia', 'blas', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise ValueError('the fixed single-thread contract is required')
    timing = protocol.get('timing_reliability', {})
    for key in ('fixed_warmup_seconds', 'block_min_seconds'):
        if not number(timing.get(key)) or timing[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not positive_integer(timing.get('blocks')):
        raise ValueError('blocks must be a positive integer')
    for key in ('maximum_language_block_spread_fraction', 'maximum_paired_ratio_spread_fraction'):
        if not number(timing.get(key)) or timing[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if timing.get('all_blocks_retained') is not True or (
            type(timing.get('initial_samples_retained')) is not int or
            timing['initial_samples_retained'] != protocol['samples']):
        raise ValueError('all blocks and the declared initial samples must be retained')
    if set(protocol.get('strict_tolerances', {})) != {'params', 'fval', 'edm', 'errors', 'covariance'}:
        raise ValueError('all five strict numerical comparisons are required')
    for tolerance in protocol['strict_tolerances'].values():
        if any(not number(tolerance.get(key)) or tolerance[key] < 0 for key in ('rtol', 'atol')):
            raise ValueError('numerical tolerances must be finite and nonnegative')
    return timing

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
    if not all(number(v) for v in aa+bb):return {'passed':False,'reason':'nonfinite or nonnumeric'}
    diffs=[abs(x-y) for x,y in zip(aa,bb)]
    budgets=[max(tol['atol'],tol['rtol']*max(abs(x),abs(y))) for x,y in zip(aa,bb)]
    margins=[d/b if b else (0 if d==0 else math.inf) for d,b in zip(diffs,budgets)]
    return {'passed':all(d<=b for d,b in zip(diffs,budgets)),'max_abs':max(diffs),'max_budget_fraction':max(margins),'signed_deltas':[x-y for x,y in zip(aa,bb)],'tolerance':tol}

def valid_block(block,seconds):
    ts=block.get('sample_times_ns',[]) if isinstance(block,dict) else []
    return (len(ts)>0 and all(number(x) and x>0 for x in ts)
        and block.get('stable_outputs') is True and positive_integer(block.get('validated_outputs'))
        and block['validated_outputs']==len(ts)
        and number(block.get('elapsed_seconds')) and block['elapsed_seconds']>=seconds)

def timing_record_complete(row, timing):
    return (valid_block(row.get('continuous_warmup'),timing['fixed_warmup_seconds'])
        and isinstance(row.get('timing_blocks'),list)
        and len(row['timing_blocks'])==timing['blocks']
        and all(valid_block(b,timing['block_min_seconds']) for b in row['timing_blocks']))

def spread(values): return max(values)/min(values)-1

def row_usable(row,n,protocol):
    if not isinstance(row,dict):return False
    samples=protocol['samples']
    r=row.get('result',{})
    fields=['params','errors','covariance','fval','edm']
    values=[]
    for k in fields:
        v=r.get(k);values.extend(v if isinstance(v,list) else [v])
    return (timing_record_complete(row,protocol['timing_reliability']) and row.get('samples')==samples and row.get('evals')==1 and row.get('stable_outputs') is True
        and len(row.get('sample_times_ns',[]))==samples
        and all(number(v) for v in values)
        and all(number(v) and v>0 for v in row.get('sample_times_ns',[]))
        and len(r.get('params',[]))==n and len(r.get('errors',[]))==n and len(r.get('covariance',[]))==n*n
        and r.get('is_valid') is True and r.get('has_covariance') is True
        and r.get('above_max_edm') is False and r.get('reached_call_limit') is False
        and positive_integer(r.get('nfcn')) and positive_integer(r.get('actual_calls'))
        and r['nfcn']==r['actual_calls'])

def compare_pair(j,c,protocol):
    validate_protocol(protocol)
    timing = protocol['timing_reliability']
    def metadata_matches(record):
        if not isinstance(record, dict) or len(record.get('rows', [])) != len(CASE_DIMENSIONS):
            return False
        actual = record.get('timing_reliability', {})
        return (type(actual.get('blocks')) is int and actual['blocks'] == timing['blocks']
                and all(number(actual.get(key)) and actual[key] == timing[key]
                        for key in ('fixed_warmup_seconds', 'block_min_seconds')))
    metadata_ok = metadata_matches(j) and metadata_matches(c)
    result={}
    for name,n in CASE_DIMENSIONS.items():
        jr=[r for r in (j or {}).get('rows',[]) if r.get('name')==name]
        cr=[r for r in (c or {}).get('rows',[]) if r.get('name')==name]
        row={'julia_rows':jr,'cpp_rows':cr,'usable':metadata_ok and len(jr)==len(cr)==1 and row_usable(jr[0],n,protocol) and row_usable(cr[0],n,protocol)}
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
            row['initial_julia_median_ns']=statistics.median(jr[0]['sample_times_ns'])
            row['julia_block_medians_ns']=[statistics.median(b['sample_times_ns']) for b in jr[0]['timing_blocks']]
            row['julia_median_ns']=statistics.median(row['julia_block_medians_ns'])
            row['initial_cpp_median_ns']=statistics.median(cr[0]['sample_times_ns'])
            row['cpp_block_medians_ns']=[statistics.median(b['sample_times_ns']) for b in cr[0]['timing_blocks']]
            row['cpp_median_ns']=statistics.median(row['cpp_block_medians_ns'])
            row['julia_over_cpp']=row['julia_median_ns']/row['cpp_median_ns']
        result[name]=row
    return result

def summarize(pairs,protocol):
    validate_protocol(protocol)
    result={}
    for name in protocol['cases']:
        rows=[p[name] for p in pairs]
        usable=len(rows)==protocol['rounds'] and all(r['usable'] for r in rows)
        r={'strict_numeric_admitted':False,'timing_stability_admitted':False,'benchmark_admitted':False,'complete_usable_pairs':sum(x['usable'] for x in rows),'usable':usable,'strict_parity':usable and all(x['strict_parity'] for x in rows),'equivalent_computation_speed_claim':False}
        if name in protocol['rosenbrock_diagnostic_tolerances']:
            r['diagnostic_pass']=usable and all(x['diagnostic_pass'] for x in rows)
            r['status']='DIAGNOSTIC_ONLY' if r['diagnostic_pass'] else 'FAIL'
        else:r['status']='STRICT_PARITY_PASS' if r['strict_parity'] else 'FAIL'
        if usable:
            ratios=[x['julia_over_cpp'] for x in rows]
            r.update({'julia_median_ns_by_pair':[x['julia_median_ns'] for x in rows], 'cpp_median_ns_by_pair':[x['cpp_median_ns'] for x in rows], 'julia_over_cpp_by_pair':ratios,'median_paired_ratio':statistics.median(ratios),'observed_ratio_range':[min(ratios),max(ratios)],'nfcn_delta_by_pair':[x['nfcn_delta'] for x in rows]})
            r['block_medians_ns_by_pair']={lang:[x[lang+'_block_medians_ns'] for x in rows] for lang in ['julia','cpp']}
            r['language_block_spread_fraction']={lang:spread([v for x in rows for v in x[lang+'_block_medians_ns']]) for lang in ['julia','cpp']}
            r['paired_ratio_spread_fraction']=spread(ratios)
            limits=protocol['timing_reliability']
            r['timing_stability_admitted']=(all(x<=limits['maximum_language_block_spread_fraction'] for x in r['language_block_spread_fraction'].values()) and r['paired_ratio_spread_fraction']<=limits['maximum_paired_ratio_spread_fraction'])
            r['strict_numeric_admitted']=r['strict_parity']
            r['benchmark_admitted']=r['strict_numeric_admitted'] and r['timing_stability_admitted']
            # Exact reproducibility of each implementation across independent processes.
            r['julia_identical_across_processes']=all(x['julia_rows'][0]['result']==rows[0]['julia_rows'][0]['result'] for x in rows)
            r['cpp_identical_across_processes']=all(x['cpp_rows'][0]['result']==rows[0]['cpp_rows'][0]['result'] for x in rows)
            if not r['julia_identical_across_processes'] or not r['cpp_identical_across_processes']:
                r['status']='FAIL'
                r['strict_numeric_admitted']=False
                r['benchmark_admitted']=False
        r['numerical_status']=r['status']
        if not usable:
            r['status']='INCOMPLETE_OR_INVALID_RECORD'
        elif not r['julia_identical_across_processes'] or not r['cpp_identical_across_processes']:
            r['status']='PROCESS_OUTPUT_VARIATION'
        elif r['benchmark_admitted']:
            r['status']='STRICT_AND_TIMING_PASS'
        elif not r['strict_numeric_admitted'] and not r['timing_stability_admitted']:
            r['status']='NUMERIC_FAIL_AND_TIMING_FAIL'
        elif not r['strict_numeric_admitted']:
            r['status']='NUMERIC_FAIL'
        else:
            r['status']='TIMING_FAIL'
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

def collector_command(language, root, binary, julia, path, transport, protocol):
    timing = validate_protocol(protocol)
    settings = [str(timing['fixed_warmup_seconds']), str(timing['blocks']),
                str(timing['block_min_seconds'])]
    if language == 'julia':
        return [julia, '--startup-file=no', f'--project={root/"benchmark"}',
                str(root/'benchmark/readme_julia_bench.jl'), str(path),
                str(protocol['samples']), str(root), *settings]
    if language == 'cpp':
        return [str(binary), str(transport), str(path), str(protocol['samples']), *settings]
    raise ValueError('unknown collector language')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--cpp-binary',required=True);p.add_argument('--out',required=True);p.add_argument('--protocol',required=True);p.add_argument('--julia',default='julia');p.add_argument('--timeout',type=int,default=900)
    a=p.parse_args();root=pathlib.Path(a.source).resolve();out=pathlib.Path(a.out).resolve()
    protocol=json.loads(pathlib.Path(a.protocol).read_text());validate_protocol(protocol)
    if a.timeout <= 0: raise ValueError('timeout must be positive')
    out.mkdir(parents=True,exist_ok=False);atomic(out/'protocol.json',protocol)
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
        records={};order=protocol['order'][i-1].split(',')
        for language in order:
            path=out/f'pair-{i}-{language}.json'
            command=collector_command(language,root,a.cpp_binary,a.julia,path,transport,protocol)
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
    report['all_cases_benchmark_admitted']=report['measurement_complete'] and all(r['benchmark_admitted'] for r in report['cases'].values())
    report['verdict']='STRICT_AND_TIMING_PASS' if report['all_cases_benchmark_admitted'] else ('COMPLETED_WITH_UNADMITTED_ROWS' if report['measurement_complete'] else 'FAIL')
    atomic(out/'summary.json',report)
    return 0 if report['all_cases_benchmark_admitted'] else (2 if report['measurement_complete'] else 1)
if __name__=='__main__':raise SystemExit(main())
