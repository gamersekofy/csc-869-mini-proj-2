#!/usr/bin/env python3
"""Read-only Python oracle and mechanical CSV comparison for the native port.
Quick means first 8000 source rows, first two ORIGINAL frozen folds (projected,
not repartitioned), 40 RF trees, DT depth 5, uncalibrated CV/test SVM.
The stock --quick runner would create data/folds_quick.json and reports;
this harness instead calls its unchanged CV/test implementations directly.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.path.insert(0, str(ROOT / 'components/mp2-py'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def protected():
    return {str(p.relative_to(ROOT)): digest(p) for d in ['components/mp2-py','data','reports']
            for p in (ROOT / d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}

def reference(args):
    import numpy as np
    from csc869.config import load_config, apply_quick, resolved_paths
    from csc869.data import load_dataset
    from csc869.cv import run_cv
    from csc869.test_eval import run_test_eval
    from csc869.preprocess import Preprocessor
    from csc869.sampling import resample_indices
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    before = protected()
    cfg = load_config(ROOT / 'components/mp2-py/configs/main.yaml')
    if args.quick:
        cfg = apply_quick(cfg, True)
    cfg['folds'] = str(ROOT / 'data/folds.json')
    cfg['results_dir'] = str(out)
    cfg['figures_dir'] = str(out / 'unused-figures')
    cfg['cv']['jobs'] = args.jobs
    cfg = resolved_paths(cfg)
    train, test = load_dataset(cfg)
    if args.quick:
        train = train.iloc[:int(cfg['quick']['max_rows'])].reset_index(drop=True)
    frozen = json.loads((ROOT / 'data/folds.json').read_text())
    n = len(train)
    folds = [(np.array([i for i in f['train'] if i < n]),
              np.array([i for i in f['val'] if i < n]))
             for f in frozen['folds'][:2 if args.quick else len(frozen['folds'])]]
    y = train['class'].to_numpy(dtype=int)
    oracle = []
    for si, scenario in enumerate(cfg['scenarios']):
        for fi, (tr, va) in enumerate(folds):
            seed = int(cfg['seed']) + fi * 17 + si * 101
            idx, counts = resample_indices(y[tr], scenario, seed)
            pre = Preprocessor(**cfg['missing'])
            xp = pre.fit_transform(train.iloc[tr[idx]])
            xv = pre.transform(train.iloc[va])
            oracle.append(dict(scenario=scenario, fold=fi, seed=seed,
                               train=tr.tolist(), val=va.tolist(), sampled=tr[idx].tolist(),
                               feature_names=pre.feature_names_,
                               mean=pre.num_mean_, std=pre.num_std_,
                               train_head=xp[:3].tolist(), val_head=xv[:3].tolist(), counts=counts))
    (out / 'input_oracle.json').write_text(json.dumps(oracle))
    t = time.perf_counter()
    run_cv(cfg, train, y, folds)
    run_test_eval(cfg, train, test)
    elapsed = time.perf_counter() - t
    after = protected()
    assert before == after, 'Protected files changed during reference evaluation'
    (out / 'parity_metadata.json').write_text(json.dumps(dict(
        elapsed_seconds=elapsed, quick=args.quick, train_rows=n, test_rows=len(test),
        frozen_folds_sha256=digest(ROOT/'data/folds.json'),
        protocol='first-8000/first-two-frozen-folds' if args.quick else 'full-frozen-folds',
        protected_files_unchanged=True), indent=2))
    print(f'PYTHON_SECONDS={elapsed:.6f}', flush=True)

def compare(args):
    import numpy as np
    import pandas as pd
    py, rs = ROOT / args.python, ROOT / args.rust
    reports = {}
    for filename, keys in [('cv_tidy.csv',['model','scenario','fold','metric']),
                           ('test_metrics.csv',['model','scenario','metric'])]:
        a,b = pd.read_csv(py/filename),pd.read_csv(rs/filename)
        assert list(a.columns)==list(b.columns), (filename,list(a.columns),list(b.columns))
        assert not a.duplicated(keys).any() and not b.duplicated(keys).any()
        joined=a.merge(b,on=keys,how='outer',suffixes=('_python','_rust'),indicator=True)
        assert (joined['_merge']=='both').all(), 'Missing result cells'
        joined['delta']=joined.value_rust-joined.value_python
        joined.to_csv(rs/('parity_'+filename),index=False)
        reports[filename]=dict(cells=len(joined),max_abs_delta=float(joined.delta.abs().max()))
        if filename=='cv_tidy.csv':
            means=joined.groupby(['model','scenario','metric'])[['value_python','value_rust','delta']].mean().reset_index()
            means.to_csv(rs/'parity_means.csv',index=False)
            for (model,scenario), group in means.groupby(['model','scenario']):
                d=group.set_index('metric')
                print(model,scenario,' | '.join(f'{m} {d.loc[m,"value_python"]:.5f}/{d.loc[m,"value_rust"]:.5f}/{d.loc[m,"delta"]:+.5f}' for m in ['f1','roc_auc','pr_auc']))
    a,b=pd.read_csv(py/'sampling_counts.csv'),pd.read_csv(rs/'sampling_counts.csv')
    cols=[c for c in a.columns if c not in ['lang','positive_rate_before','positive_rate_after']]
    pd.testing.assert_frame_equal(a[cols],b[cols],check_dtype=False)
    oracle=json.loads((py/'input_oracle.json').read_text())
    rust=json.loads((rs/'input_audit.json').read_text())
    exact={k:True for k in ['folds','resampling','features','preprocessing_head']}
    for p,r in zip(oracle,rust,strict=True):
        exact['folds'] &= p['train']==r['train'] and p['val']==r['val']
        exact['resampling'] &= p['sampled']==r['sampled']
        exact['features'] &= p['feature_names']==r['feature_names']
        exact['preprocessing_head'] &= bool(np.allclose(p['train_head'],r['train_head'],atol=1e-10,rtol=1e-10) and np.allclose(p['val_head'],r['val_head'],atol=1e-10,rtol=1e-10))
    reports['input_parity']=exact
    (rs/'parity_checks.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports,indent=2))
    assert all(exact.values()), 'Frozen inputs or preprocessing parity failed'

def fixtures(args):
    import numpy as np
    cases=[]
    for seed in [0,1,42,59,143,244,345,2**32+17]:
        rng=np.random.default_rng(seed)
        cases.append(dict(seed=seed,raw=[int(x) for x in rng.bit_generator.random_raw(8)],
                          shuffle=np.random.default_rng(seed).permutation(17).tolist(),
                          replace=np.random.default_rng(seed).choice(13,17,replace=True).tolist(),
                          choice=np.random.default_rng(seed).choice(101,31,replace=False).tolist(),
                          large_choice=np.random.default_rng(seed).choice(20001,501,replace=False).tolist()))
    path=ROOT/'components/mp2-rust/tests/rng_oracle.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(cases,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('reference'); a.add_argument('--quick',action='store_true'); a.add_argument('--out',default='results_quick/python'); a.add_argument('--jobs',type=int,default=2)
    a=sub.add_parser('compare'); a.add_argument('--python',default='results_quick/python'); a.add_argument('--rust',default='results_quick/rust')
    sub.add_parser('fixtures')
    a=p.parse_args(); globals()[a.action](a)
