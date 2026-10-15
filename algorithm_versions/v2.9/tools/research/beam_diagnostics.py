"""Post-freeze analysis only. Does not alter or choose any trained weights."""
from pathlib import Path
import argparse, csv, json, math
import numpy as np
import torch
import beam_campaign as c
from scipy.ndimage import maximum_filter

def summarize(x):
    x=np.asarray(x)
    return dict(count=int(x.size),mae_db=float(np.mean(abs(x))),bias_db=float(x.mean()),p95_abs_db=float(np.percentile(abs(x),95))) if x.size else dict(count=0)

def extra(root):
    if (root/'extended_diagnostics.json').exists():return
    final=c.read(root/'final_report.json');sel=final['selection'];scale=c.read(root/'protocol.json')['scale']
    data=c.data_load(root,'holdout');model=c.CompactVectorCNN().cuda()
    model.load_state_dict(torch.load(root/'runs'/f"{sel['recipe']}_{sel['representative_seed']}"/'best.pt',weights_only=False)['model']);model.eval()
    t,p,q,h2=c.ffd('exportelement_new');h=c.cartesian(t,p,h2)
    qt=torch.tensor(q,dtype=torch.float32,device='cuda');hp=c.predict(model,qt,scale)
    wt=torch.tensor(data['commands'],device='cuda');true=c.gains(torch.tensor(h,dtype=torch.complex64,device='cuda'),wt).cpu().numpy()
    pred=c.gains(hp,wt).cpu().numpy();error=c.db(pred)-c.db(true)
    full=c.metrics_numpy(pred,true,t)
    # Measured local half-power cuts for the 24 macro commands (not ideal AF estimates).
    macro_widths=[]
    exportroot=c.BASE/'data/external_next/hfss_direct_validation/exports/realized_gain'
    def cutwidth(cut,index,periodic=False):
        threshold=cut[index]-3.;dist=[]
        for sign in [-1,1]:
            last=float(cut[index]);found=None
            for step in range(1,len(cut)):
                j=index+sign*step
                if not periodic and not 0<=j<len(cut):break
                value=float(cut[j%len(cut)])
                if value<threshold:
                    found=step-1+(last-threshold)/(last-value);break
                last=value
            if found is None:return None
            dist.append(found)
        return sum(dist)
    for path in sorted(exportroot.glob('*_realized_gain.csv')):
        values=np.loadtxt(path,skiprows=1,delimiter=',');grid=values[:,3].reshape(180,360);ti,pi=np.unravel_index(np.argmax(grid),grid.shape)
        tw=cutwidth(grid[:,pi],ti);pw=cutwidth(grid[ti],pi,True)
        greatcircle=None if pw is None else float(np.rad2deg(np.arccos(np.clip(np.cos(np.deg2rad(ti+.5))**2+np.sin(np.deg2rad(ti+.5))**2*np.cos(np.deg2rad(pw)),-1,1))))
        macro_widths.append(dict(command_id=path.name.split('_')[0],theta_cut_hpbw_deg=tw,phi_coordinate_cut_hpbw_deg=pw,phi_cut_spherical_width_deg=greatcircle))
    relative=c.db(true/true.max(1,keepdims=True));prelative=c.db(pred/pred.max(1,keepdims=True))
    rb=np.digitize(-relative,[10,20,30]);pb=np.digitize(-prelative,[10,20,30])
    confusion=np.zeros((4,4),dtype=int)
    for i in range(4):
        for j in range(4):confusion[i,j]=int(np.sum((rb==i)&(pb==j)))
    predicted_bins={name:summarize(error[pb==i]) for i,name in enumerate(['strong','medium','weak','deep'])}
    neighborhood={name:[] for name in ['global_peak_neighborhood','competing_peak_neighborhood','other_strong']}
    for k in range(len(true)):
        grid=true[k].reshape(180,360);peak=int(np.argmax(true[k]));strong=relative[k]>=-10
        candidates=np.flatnonzero((grid==maximum_filter(grid,size=3,mode=('nearest','wrap'))).ravel() & strong)
        candidates=candidates[np.argsort(true[k,candidates])[::-1]];chosen=[]
        for j in candidates:
            if not chosen or np.max(q[chosen]@q[j])<np.cos(np.deg2rad(5)):chosen.append(int(j))
        main=(q@q[peak]>=np.cos(np.deg2rad(5)))&strong
        other=np.zeros(len(q),bool)
        for j in chosen[1:]:other|=q@q[j]>=np.cos(np.deg2rad(5))
        other &= strong & ~main
        for name,mask in [('global_peak_neighborhood',main),('competing_peak_neighborhood',other),('other_strong',strong&~main&~other)]:neighborhood[name].append(error[k,mask])
    neighborhood={k:summarize(np.concatenate(v)) for k,v in neighborhood.items()}
    # Same inputs and Ephi-only observable for old/new comparison (no renaming Ephi as total gain).
    from v29.models.amplitude_forward import model_from_config
    oldpath=c.BASE/'artifacts/research_20260905/runs/final/R1_seed29003/best.pt'
    checkpoint=torch.load(oldpath,weights_only=False);old=model_from_config(checkpoint['model_config']).cuda();old.load_state_dict(checkpoint['model_state']);old.eval()
    w=wt.reshape(-1,4,4);wiq=torch.stack([w.real,w.imag],1)
    ix=data['ids'];qheld=torch.tensor(data['q'],device='cuda')
    with torch.no_grad():oldamp=torch.cat([old.forward_command_grid(wiq,qheld[i:i+1024]) for i in range(0,len(qheld),1024)],1).cpu().numpy()
    ep=np.stack([-np.sin(np.deg2rad(p)),np.cos(np.deg2rad(p)),np.zeros_like(p)],1)
    pp=(hp*torch.tensor(ep,dtype=torch.float32,device='cuda')[:,:,None]).sum(1)[:,None,:]
    newphi=c.gains(pp[ix],wt).cpu().numpy();refphi=c.gains(torch.tensor(h2[ix,1:2,:],dtype=torch.complex64,device='cuda'),wt).cpu().numpy()
    oldnew=dict(old_Ephi=c.metrics_numpy(oldamp**2/60,refphi,data['theta']),new_Ephi=c.metrics_numpy(newphi,refphi,data['theta']),
                limitation='old model saw different angular split and budgets; common observable comparison is diagnostic, not clean causal architecture ablation')
    # Same-new-model A ablation evaluated post-freeze, seed chosen using A development results only.
    aseed=min(c.read(root/'dev_A.json'),key=lambda x:(x['p95_regret'],x['validation_nmse_db']))['seed']
    am=c.CompactVectorCNN().cuda();am.load_state_dict(torch.load(root/'runs'/f'A_{aseed}'/'best.pt',weights_only=False)['model']);am.eval()
    abaseline,_=c.eval_targets(root,am,data,scale,'baseline_A_holdout')
    ah=c.predict(am,qheld,scale);atrue=c.gains(torch.tensor(data['h'],device='cuda'),wt).cpu().numpy()
    abforward=c.metrics_numpy(c.gains(ah,wt).cpu().numpy(),atrue,data['theta'])
    # Geometry-dependent conditioning and phase quantization sensitivity on frozen commands.
    arrays=np.load(root/'holdout_target_arrays.npz'); hw=data['trajectory_h'].reshape(-1,4,3,16)
    singular=np.linalg.svd(hw.reshape(-1,12,16),compute_uv=False);energy=singular**2
    erank=energy.sum(1)**2/(energy**2).sum(1)
    conditioning={}
    for name in np.unique(data['trajectory_strata']):
        mask=np.repeat(data['trajectory_strata']==name,3)
        conditioning[str(name)]=dict(effective_rank_quantiles=np.percentile(erank[mask],[5,50,95]).tolist(),frame_count=int(mask.sum()))
    # Three final seeds' target performance, with no post-holdout model selection.
    seed_targets={str(sel['representative_seed']):final['targets']}
    for seed in c.SEEDS:
        if seed==sel['representative_seed']:continue
        sm=c.CompactVectorCNN().cuda();sm.load_state_dict(torch.load(root/'runs'/f"{sel['recipe']}_{seed}"/'best.pt',weights_only=False)['model']);sm.eval()
        seed_targets[str(seed)]=c.eval_targets(root,sm,data,scale,f'holdout_seed{seed}')[0]
    # Controlled same-budget continuity comparison, independently of exploratory faster warm starts.
    tq=torch.tensor(data['trajectory_q'],device='cuda');th=torch.tensor(data['trajectory_h'],device='cuda')
    predh=c.predict(model,tq.reshape(-1,3),scale).reshape(len(tq),3,4,3,16)
    cold=[];warm=[];last=None
    for frame in range(3):
        cw,_=c.optimize(predh[:,frame],starts=8,steps=160,seed=30500+frame)
        ww,_=c.optimize(predh[:,frame],starts=8,steps=160,seed=30500+frame,initial=last)
        cold.append(cw);warm.append(ww);last=ww
    cold=torch.stack(cold,1);warm=torch.stack(warm,1)
    def true_min(w):return torch.einsum('bfkpn,bfn->bfkp',th,w).abs().square().sum(-1).min(-1).values/60
    def jump(w):return (2-2*(w[:,1:].conj()*w[:,:-1]).sum(-1).abs()).clamp_min(0).sqrt()
    delta=(c.tdb(true_min(warm))-c.tdb(true_min(cold))).cpu().numpy()
    continuity=dict(budget='8starts x160steps, identical random starts except final warm start slot',
                    true_min_gain_change_dB_quantiles=np.percentile(delta,[5,50,95]).tolist(),
                    cold_min_gain_p5_dBi=float(torch.quantile(c.tdb(true_min(cold)).flatten(),.05)),
                    warm_min_gain_p5_dBi=float(torch.quantile(c.tdb(true_min(warm)).flatten(),.05)),
                    cold_phase_invariant_jump_median=float(jump(cold).median()),warm_phase_invariant_jump_median=float(jump(warm).median()))
    ww=arrays['phase_weights'];base=arrays['phase_real_gain'].min(-1);quant={}
    for bits in [3,4,6,8]:
        phase=np.angle(ww);quantw=np.exp(1j*np.round(phase/(2*np.pi/2**bits))*(2*np.pi/2**bits))/4
        g=np.sum(abs(np.einsum('bkpn,bn->bkp',hw,quantw))**2,-1)/60
        change=c.db(g.min(-1))-c.db(base)
        quant[str(bits)]=dict(delta_min_gain_db_quantiles=np.percentile(change,[5,50,95]).tolist(),assumption='ideal uniform phase codebook sensitivity only')
    # Trajectory-block bootstrap; adjacent frames are never treated as independent draws.
    regret=arrays['phase_regret'].reshape(-1,3);rng=np.random.default_rng(295551);med=[];tail=[]
    for _ in range(500):
        a=regret[rng.integers(0,len(regret),len(regret))].ravel();med.append(np.median(a));tail.append(np.percentile(a,95))
    result=dict(full_sphere_forward=full,macro_measured_local_widths=macro_widths,prediction_defined_intensity_bins=predicted_bins,intensity_confusion_counts=confusion.tolist(),
                peak_regions=neighborhood,peak_region_definition='full-grid local maxima, separated by >=5 deg; neighborhoods within5 deg and reference>=peak-10dB; not certified grating-lobe identities',
                old_new_common_Ephi=oldnew,baseline_A=dict(seed=aseed,forward=abforward,targets=abaseline),
                target_effective_rank=conditioning,phase_quantization=quant,target_seed_summaries=seed_targets,controlled_continuity=continuity,
                regret_trajectory_bootstrap95=dict(median=np.percentile(med,[2.5,97.5]).tolist(),p95=np.percentile(tail,[2.5,97.5]).tolist()),
                whole_sphere_diagnostic='Includes development angular blocks. Final acceptance remains held-out blocks from final_report.json.')
    c.dump(root/'extended_diagnostics.json',result)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for recipe in 'ABCDE':
        for si,seed in enumerate(c.SEEDS):
            hist=c.read(root/'runs'/f'{recipe}_{seed}'/'history.json');axs[0].plot([x['epoch'] for x in hist],[x['validation_nmse_db'] for x in hist],alpha=.6,label=recipe if si==0 else None)
    axs[0].legend();axs[0].set_xlabel('Epoch within recipe');axs[0].set_ylabel('Validation amplitude NMSE (dB)')
    take=rng.choice(error.size,50000,replace=False);axs[1].hexbin(relative.ravel()[take],error.ravel()[take],gridsize=80,mincnt=1,bins='log');axs[1].set_xlabel('True relative gain (dB)');axs[1].set_ylabel('Predicted - true gain (dB)');axs[1].set_xlim(-50,0)
    fig.savefig(root/'convergence_and_intensity.png',dpi=150);plt.close(fig)

def report(root):
    r=c.read(root/'final_report.json');x=c.read(root/'extended_diagnostics.json');m=c.read(root/'macro/summary.json');p=c.read(root/'protocol.json');sel=r['selection']
    table=['| 配方 | 父配方 | seed | 实际轮数 / 最佳轮数 | 收敛 | 验证 NMSE(dB) |','|---|---|---:|---:|---|---:|']
    for recipe in 'ABCDE':
        for seed in c.SEEDS:
            a=c.read(root/'runs'/f'{recipe}_{seed}'/'result.json');table.append(f"| {recipe} | {a['parent'] or '从头'} | {seed} | {a['epochs']} / {a['best_epoch']} | {a['converged']} | {a['best_validation_nmse_db']:.3f} |")
    stats=['|参考强度区间|三 seed 平均 MAE(dB)|三 seed 平均偏差(dB)|三 seed 平均绝对误差P95(dB)|','|---|---:|---:|---:|']
    for name in ['all','strong','medium','weak','deep']:
        rows=[a['bins'][name] for a in r['forward']];stats.append(f"|{name}|{np.mean([a['mae_db'] for a in rows]):.3f}|{np.mean([a['bias_db'] for a in rows]):.3f}|{np.mean([a['p95_abs_db'] for a in rows]):.3f}|")
    a01=m['commands'][0];phase=r['targets']['phase'];amp=r['targets']['amplitude_phase'];baseline=x['baseline_A']['targets']['phase']
    text=f'''# V2.9 双极化、波束统计与新训练记录

## 本轮结论与证据边界

本轮完成 24 组宏观统计、A–E 五种 loss 的三 seed 训练、独立命令/角块保留评估，以及 120 条合成轨迹（每条3帧、每帧4站）的优化分析。胜出配方为 {sel['recipe']}，代表 seed 为 {sel['representative_seed']}。参数 {p['model_parameters']:,}，卷积 MAC/方向 {p['conv_mac_per_direction']:,}，相对旧34973参数下降17.5%。当前全验收通过状态：{all(r['acceptance'].values())}。

数据均来自同一个47GHz HFSS解，24组为嵌入式方向图线性合成，只有A01直接报表对齐已完成。此前报告的 1e-13 级误差是算术闭环，不是CNN预测精度，也不是多状态独立验证。新的训练/验证/校准/保留命令和轨迹ID分离；角块训练文件也分离，但完整方向图在历史诊断中已暴露，因此本轮不能称为从未看过电磁状态的严格外部盲测。C/D历史优化命令只作宏观诊断，未回灌训练。

## 阵列性质及其后果

阵列为4×4，法向+x，行+z、列+y。波长6.379mm，行距15mm=2.352λ，列距6.8mm=1.066λ；阵元中心跨度45×20.4mm。理想均匀阵列因子法向HPBW估计约5.40°/11.90°，仅作孔径尺度，不能替代HFSS实测峰宽。行方向波束更窄，峰间干涉和扫描混叠更敏感。相位扫描会同时移动/重排多个波峰，CNN不应把所有竞争波峰平滑掉。

A01最高增益 {a01['peak_gain_dBi']:.3f}dBi，位于 θ={a01['peak_theta_deg']}°、φ={a01['peak_phi_deg']}°；另有{a01['competing_peak_count_minus3dB']}个相隔至少5°的峰高于峰值−3dB。全球高于峰值−3dB的立体角比例仅{100*a01['coverage_minus3dB']:.3f}%。等幅随机相位B组峰值约10.4–12.2dBi，分散出更多局部峰，较大“相对−10dB覆盖”部分源于其峰值降低，不能解释为绝对高增益覆盖必然改善。

A01 Eθ辐射能量比例 {100*a01['theta_polarized_energy_fraction']:.2f}%，所以Eφ-only不适合作为RealizedGainTotal。C/D成对比较表明调幅会改变波束功率分配与零陷，例如C03有效激励端口约11.14，D03为16；其最高峰14.36/14.49dBi，不能据最高峰判断四点最小值谁更好。统一放大所有激励时场幅度按sqrt(P)增长，而RealizedGainTotal按入射功率归一化后不变；比较接收功率时才应再乘总发射功率。

全球辐射算子有效秩约{m['radiation_effective_rank']:.3f}/16，表示全局辐射自由度没有明显坍缩。固定总功率且忽略公共相位时，连续幅相约30个实自由度、phase-only约15个实自由度。这不保证任意四站均能获得高增益：近邻目标的响应相关、远距组合的能量分摊、元素方向性及栅瓣竞争仍限制可达性。报告另给每种轨迹的局部有效秩、数值可行参考和谱上界差距。谱上界是松上界，32起点参考并非全局最优。辐射Gram最大特征值约1.018，提示积分/数值/归一化误差仍需核查，不能据此声称无源效率超过100%。

## 模型、loss与优化如何吸收这些性质

共享4×4孔径CNN接受方向Fourier特征和阵元行列/边缘/角点信息，输出Cartesian复场修正；固定横向投影确保场与方向正交，避免球坐标极点的Eθ/Eφ基底奇异；固定exp(jkq·r)保留传播相位；最后严格按H^T w线性叠加，并算 (|rEθ|²+|rEφ|²)/(60Pinc)。CNN只编码方向到复算子，不把w交给任意非线性层。完整方向图表仅用于离线标签，不随CNN部署；当前尚未验证FPGA量化、真实资源和时延。

只有总增益监督时，横向场因子存在方向相关的酉变换不唯一性：H与U(q)H产生相同总功率。因此A以及继承A且未启用复算子监督的候选，只能将内部输出解释为等效场因子；不能保证其单独Eθ/Eφ或相位是真实HFSS分量。总增益和对应四点优化仍有意义；未来若接收站要求特定极化或Jones投影，需要额外极化/复场监督。旧/新Eφ诊断用于暴露这一差别，不作为本轮选型依据。

基础loss为幅度NMSE+0.1功率NMSE+0.03dB Huber（−50dB数值底）；A仅基础，B增加0.5复算子监督，C增加0.08训练轨迹困难样本Huber，D增加0.01单侧高估平方惩罚（0.1dB容差），E增加0.02弱点排序KL。复算子监督只消去所有端口/Cartesian分量共同的方向相关相位，保留相对相位及极化关系。训练困难样本来自训练轨迹的参考命令及定期模型优化命令，测试轨迹不回灌。

所有配方使用3 seed、相同规则，至少600轮；连续200轮最佳验证NMSE改善<0.03dB则早停。第一条A在3000轮仍改善，因此依据P5统一允许延长至6000轮，且修订发生在保留集开启前。A/B从头；C/D/E从开发集目前最佳父配方继续，采用顺序消融，非完全因子实验。每次只在开发集以phase-only四点P95 regret选择配方，并设置全场NMSE不得退步>0.3dB的保护条件。最终checkpoint按各运行验证NMSE选择，代表seed按开发四点结果选择。

顺序微调没有额外设置“相同新增训练预算但保持基础loss”的继续训练对照，因此C相对A的收益不能完全归因于loss本身；三seed均值改善也不等于统计显著。JSON保留各seed的结果。B当前0.5权重的失败不能推导出所有复算子预训练方案都无效。

四点优化使用功率dB soft-min，温度1.2→0.12dB、Adam投影/归一化迭代；每步保留hard-min最好的可行候选。分别报告phase-only和连续幅相；模型8起点×160步，参考32起点×160步并包含模型可行解，从而避免错误的负regret。连续性实验为4起点×80步加上一帧权值；预算不同，作为热启动效率探索，不作为公平算法优越性证明。统一LCB减常数只校准预测报值，不会改变最优w；64条校准轨迹用于三帧四站联合单侧分位数，真实覆盖区间单列。

## 实际训练记录

{chr(10).join(table)}

## 新保留集前向误差

强度相对每条命令在保留角块上的参考峰值：strong≥−10dB，medium=[−20,−10)，weak=[−30,−20)，deep<−30dB。全网格统计另列在extended_diagnostics，包含开发角块，不能混入主验收。以下MAE按等角点计算，同时JSON提供sinθ加权MAE。

{chr(10).join(stats)}

三 seed幅度NMSE样本标准差 {r['seed_nmse_std_db']:.4f}dB。弱场的dB误差对小的绝对复场误差十分敏感，应同时观察功率NMSE、偏差与误差P95。分桶分别以参考和预测定义，附混淆计数；避免仅按模型自报强度证明模型准确。全局峰邻域、其他竞争峰邻域、其余强场分别统计，竞争峰未被全部命名为栅瓣。

## 四点效果与自由度

代表模型phase-only regret中位数/P95：{100*phase['median_regret']:.2f}%/{100*phase['p95_regret']:.2f}%；四点MAE {phase['fourpoint_mae_db']:.3f}dB，真实瓶颈高估P95 {phase['bottleneck_over_p95_db']:.3f}dB，瓶颈点身份识别率{100*phase['bottleneck_identity_accuracy']:.1f}%。

同一新结构的A基础loss代表模型phase-only中位/P95 regret为{100*baseline['median_regret']:.2f}%/{100*baseline['p95_regret']:.2f}%，因此本轮loss改进是否有收益可直接对照。旧Eφ模型另按同一Eφ观测和同一批命令重算，仅作诊断，不能将旧Eφ NMSE与新总增益NMSE直接比较。

C的保留集P95比A更差（18.75%对17.78%），所以开发集上的小幅收益未形成稳健的尾部改进证据。冻结后的代表模型仍按事先选择保留C，不根据保留集重新改选A。

三seed的phase-only regret中位/P95分别为：29511：3.86%/20.33%；29512：4.05%/24.52%；29513：3.07%/18.75%。三条都满足中位数门槛，但都未达到尾部10%门槛。

前半球分散组P95 regret约9.81%，聚集组约6.62%，全球压力组约24.68%。说明困难度明显依赖目标几何/可见域。后续需要真实可见域，不能在看过保留结果后删除压力组来宣称本轮全部通过。

连续幅相regret中位/P95：{100*amp['median_regret']:.2f}%/{100*amp['p95_regret']:.2f}%。两种控制约束的参考解不同，不直接相互比较regret以宣称硬件增益。三个轨迹族为局部聚集、前半球分散和全球压力，后者包含实际卫星不可见方向，仅测试数值边界，不能声称代表真实地面站过境覆盖率。3/4/6/8bit相位量化仅为未知硬件下敏感性分析。

联合LCB覆盖率 {100*r['calibration']['holdout_joint_coverage']:.2f}%，95% Wilson区间{[round(100*v,2) for v in r['calibration']['wilson95']]}%；仅120条独立轨迹不足以满足原计划“覆盖率区间宽度≤2个百分点”，相邻帧不能充作独立样本。regret区间按整轨迹bootstrap统计。

## 波峰、连续性与极化的补充解读

A01实际局部半功率切面宽度约θ方向6.20°、φ坐标方向12.59°（对应球面角距11.36°）。宏观统计与理想孔径估计相符，但多峰竞争使全局峰可能跳到另一分支。相位应控制扫描与多点干涉；调幅用于功率分配和零陷/旁峰控制，且总入射功率约束必须保持一致。

冻结模型全网格诊断的最高峰邻域MAE约0.131dB、其他竞争峰邻域0.149dB，两者都比弱场更准确，但最高峰并非比所有竞争峰显著优越。这些统计含开发角块，仅作为误差特性诊断。

同一输入的Eφ对照：旧模型NMSE约−22.39dB/MAE1.10dB，新模型Eφ投影NMSE约−4.71dB/MAE5.64dB。这证实只用总增益监督不能恢复真实单独极化分量。新模型可用于本轮总增益目标，但不能直接替换要求真实Eφ或特定接收极化的接口。

相同8起点×160步的热启动对照，公共相位对齐后的权值跳变中位数由0.219降至0.207，但真实最小增益P5由0.156dBi变为0.129dBi，并未证明增益提高。3/4/6/8bit直接取整的最小增益变化中位数约−0.493/−0.132/−0.010/−0.001dB；仅是理想码字敏感性，未做量化后再优化，不能等同于实际硬件误差。

下一轮应优先研究：总功率二次型Q=HᴴH/60的监督（对极化酉变换不敏感）、训练分布上的瓶颈高估与近零场稳健加权、真实可见域轨迹及同预算微调对照；如需要真实极化分量，则保留真正的极化监督并重新消融权重。任何新方案需新活动，不能在本轮保留集上继续调参。

## 验收与后续记录

验收逐项：`{json.dumps(r['acceptance'],ensure_ascii=False)}`。失败指标如实保留；保留集已经打开，本活动拒绝继续训练调参。若改loss或结构，需另建活动和新的保留命令/轨迹。

实际应用仍需：多组非同相命令的直接HFSS闭环、真实轨道姿态/地面站可见域、接收极化与链路参数、实物幅相误差、相移器码字及FPGA资源/位宽验证。本次结果服务于固定HFSS状态的CNN和优化器研究。

## 文件与复现

入口 `tools/research/beam_campaign.py`，依次 `macro`、`prepare`、`train`、`evaluate`，均传 `--root artifacts/vector_beam_20260905`；图表与补充诊断由 `tools/research/beam_diagnostics.py --root ...` 生成。环境 `D:/Miniconda3/envs/rl_env/python.exe`，CUDA GPU。

核心文件：`protocol.json`、`split_audit.json`、`runs/`、`selection_frozen.json`、`final_report.json`、`extended_diagnostics.json`、`*_target_arrays.npz`；无不必要哈希。`macro/command_statistics.csv`保留24组逐项数据。

物理定义参考：[Ansys Realized Gain](https://ansyshelp.ansys.com/public/Views/Secured/Electronics/v242/en/Subsystems/HFSS/Content/HFSS/PeakRealizedGain.htm)、[Analog Devices grating lobes](https://www.analog.com/en/resources/analog-dialogue/articles/phased-array-antenna-patterns-part2.html)。
'''
    (root/'RESULTS.md').write_text(text,encoding='utf-8')
    print('REPORT WRITTEN',root/'RESULTS.md',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    extra(args.root);report(args.root)
