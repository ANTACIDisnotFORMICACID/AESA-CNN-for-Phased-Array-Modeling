"""Post-registered statistics and high-budget HFSS numerical reference."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import beam_campaign as campaign
import four_point_optimizer_study as study


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def wilson(k: int, n: int):
    z = 1.96; p = k / n; den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [100*(center-half), 100*(center+half)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    raw = json.loads((root/'results.json').read_text(encoding='utf-8'))
    arrays = np.load(root/'study_arrays.npz')
    theta, phi, qfull, hs = campaign.ffd('exportelement_new')
    hfull = campaign.cartesian(theta, phi, hs).astype(np.complex64)
    indices = arrays['original_indices']; labels = arrays['strata']
    true_h = torch.tensor(hfull[indices], device='cuda')
    initial = torch.tensor(arrays['baseline_softmin_weights'], device='cuda')
    start = time.perf_counter()
    oracle_w, _ = campaign.optimize(true_h, mode='phase', starts=32, steps=320,
                                    seed=study.SEED+900, initial=initial)
    torch.cuda.synchronize(); oracle_seconds = time.perf_counter()-start
    oracle_np = oracle_w.cpu().numpy()
    oracle, odb, orel, olocal = study.evaluate_method(
        oracle_np, indices, indices, hfull, hfull[indices], qfull, labels)
    np.savez_compressed(root/'hfss_reference_arrays.npz', weights=oracle_np,
                        true_gain_dBi=odb, relative_global_dB=orel, local_peak_gap_dB=olocal)
    names = list(study.METHODS)
    baseline_success = arrays['baseline_softmin_relative_global_dB'].min(1) >= -10
    baseline_min = arrays['baseline_softmin_true_gain_dBi'].min(1)
    rng = np.random.default_rng(study.SEED+1); n=len(labels); draws=10000
    sample = rng.integers(0,n,(draws,n)); comparisons={}
    for name in names:
        success = arrays[name+'_relative_global_dB'].min(1) >= -10
        mingain = arrays[name+'_true_gain_dBi'].min(1)
        delta = 100*(success[sample].mean(1)-baseline_success[sample].mean(1))
        med_delta = np.median(mingain[sample]-baseline_min[sample],axis=1)
        comparisons[name] = {
            'success_count': int(success.sum()),
            'success_wilson95_percent': wilson(int(success.sum()),n),
            'paired_success_delta_vs_baseline_percent': float(100*(success.mean()-baseline_success.mean())),
            'paired_bootstrap_success_delta95_percent': np.percentile(delta,[2.5,97.5]).tolist(),
            'per_case_min_gain_delta_median_dB': float(np.median(mingain-baseline_min)),
            'paired_bootstrap_median_gain_delta95_dB': np.percentile(med_delta,[2.5,97.5]).tolist(),
            'success_flips_vs_baseline': {
                'fail_to_pass': int(np.sum(~baseline_success & success)),
                'pass_to_fail': int(np.sum(baseline_success & ~success)),
            },
        }
    oracle_min = odb.min(1); baseline_linear=10**(baseline_min/10); oracle_linear=10**(oracle_min/10)
    regret = np.maximum(0,1-baseline_linear/np.maximum(oracle_linear,1e-30))
    result = {
        'comparisons': comparisons,
        'hfss_numerical_reference': {
            'description': 'true-HFSS phase-only max-min, 32 starts x 320 steps, includes baseline weight; numerical feasible reference, not global optimum',
            'optimizer_seconds': oracle_seconds,
            'metrics': oracle,
            'baseline_min_gain_regret_median_p95_percent': (100*np.percentile(regret,[50,95])).tolist(),
        },
        'interpretation': [
            'Confidence intervals are trajectory-set resampling only; they do not cover new HFSS states or hardware errors.',
            'Relocate changes coordinates and is not directly comparable for fixed stations.',
            'Runtime is one warm GPU pass, not a latency benchmark.',
        ],
    }
    dump(root/'statistical_analysis.json',result)
    report=(root/'RESULTS.md').read_text(encoding='utf-8').rstrip()
    report += '\n\n## 配对统计与数值参考\n\n'
    report += '|方法|成功数/180|成功率95%区间|相对baseline变化及配对bootstrap 95%区间|最小增益逐例中位变化(dB)|翻转：失败→成功 / 成功→失败|\n'
    report += '|---|---:|---:|---:|---:|---:|\n'
    for name in names:
        x=comparisons[name]; ci=x['success_wilson95_percent']; di=x['paired_bootstrap_success_delta95_percent']; fl=x['success_flips_vs_baseline']
        report += f"|{name}|{x['success_count']}/180|[{ci[0]:.1f}%, {ci[1]:.1f}%]|{x['paired_success_delta_vs_baseline_percent']:+.1f}pp [{di[0]:+.1f}, {di[1]:+.1f}]|{x['per_case_min_gain_delta_median_dB']:+.3f}|{fl['fail_to_pass']} / {fl['pass_to_fail']}|\n"
    report += f"\n高预算HFSS数值max-min参考的最小增益P5/中位/P95为 {oracle['min_gain_dBi_p5_median_p95'][0]:.2f}/{oracle['min_gain_dBi_p5_median_p95'][1]:.2f}/{oracle['min_gain_dBi_p5_median_p95'][2]:.2f} dBi；baseline相对该可行参考的最小增益regret中位/P95为 {100*np.percentile(regret,50):.2f}%/{100*np.percentile(regret,95):.2f}%。参考解不是全局最优。\n"
    report += '\n置信区间只反映这180组随机坐标的抽样不确定性，不覆盖新的HFSS状态、制造误差或硬件误差。运行时间仅为一次GPU顺序执行，不能作为FPGA时延。\n'
    (root/'RESULTS.md').write_text(report,encoding='utf-8')
    print('ANALYSIS COMPLETE',root,flush=True)


if __name__=='__main__':
    main()
