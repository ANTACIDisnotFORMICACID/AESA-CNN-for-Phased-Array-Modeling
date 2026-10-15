"""Prospective dual-polarization experiment; historical experiments remain frozen.

Actions separate macro diagnostics, preparation, development-only training and
one-shot held-out evaluation. No per-file hashing is performed.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.ndimage import maximum_filter, label

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / 'src'))
from v29.models.blocks_2d import ApertureCorrectionBranch, DirectionFourierEncoder
from v29.models.nominal_pattern import _geometry_maps
from v29.contracts import ArrayGeometry

SEEDS = [29511, 29512, 29513]
SPLITS = ['train', 'validation', 'calibration', 'holdout']
CFG = dict(seeds=SEEDS, max_epochs=3000, extension_max_epochs=6000, min_epochs=600, plateau_epochs=200,
           plateau_db=0.03, batch=512, steps_per_epoch=4, validate_every=10,
           learning_rate=0.001, learning_rate_floor=1e-5, op_weight=0.5,
           hard_weight=0.08, over_weight=0.01, rank_weight=0.02,
           recipes=['A','B','C','D','E'], device='cuda', split_seed=2952026,
           gain_floor_db=-50, optimizer_steps=160, model_starts=8, reference_starts=32,
           parameter_budget=34973)

def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()), encoding='utf-8')
    tmp.replace(p)

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def db(x): return 10*np.log10(np.maximum(x,1e-30))
def dirs(theta, phi):
    t,p=np.deg2rad(theta),np.deg2rad(phi)
    return np.stack([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)],-1)
def ffd(prefix):
    folder=BASE.parent/'v2.8/HFSS/exports/hfss_47ghz'
    values=[]
    for port in range(1,17):
        path=folder/f'{prefix}_{port}.ffd'
        with path.open() as stream: head=[next(stream) for _ in range(4)]
        ts,ps=[list(map(float,s.split())) for s in head[:2]]
        theta=np.linspace(ts[0],ts[1],int(ts[2])); phi=np.linspace(ps[0],ps[1],int(ps[2]))
        a=np.loadtxt(path,skiprows=4)
        assert a.shape==(len(theta)*len(phi),4) and np.isfinite(a).all()
        values.append(np.stack([a[:,0]+1j*a[:,1],a[:,2]+1j*a[:,3]],-1))
    t,p=np.meshgrid(theta,phi,indexing='ij'); q=dirs(t.ravel(),p.ravel())
    h=np.stack(values,-1) # Q,polarization,port
    return t.ravel(),p.ravel(),q,h

def cartesian(t,p,h):
    tr,pr=np.deg2rad(t),np.deg2rad(p)
    et=np.stack([np.cos(tr)*np.cos(pr),np.cos(tr)*np.sin(pr),-np.sin(tr)],-1)
    ep=np.stack([-np.sin(pr),np.cos(pr),np.zeros_like(pr)],-1)
    return et[:,:,None]*h[:,0,None,:]+ep[:,:,None]*h[:,1,None,:]

def macro(root):
    target=root/'macro'
    if (target/'summary.json').exists(): return
    t,p,q,h=ffd('exportelement_new'); n=180*360
    area=np.sin(np.deg2rad(t))*np.deg2rad(1)**2
    forward=q[:,0]>0
    exports=BASE/'data/external_next/hfss_direct_validation'
    rows=[]; peaks={}; maps={}
    for path in sorted((exports/'commands/sources').glob('cmd_*_sources.csv')):
        with path.open(encoding='utf-8-sig') as f: src=list(csv.DictReader(f))
        cid=src[0]['command_id']; power=np.array([float(r['power_w']) for r in src])
        w=np.sqrt(power)*np.exp(1j*np.deg2rad([float(r['phase_deg']) for r in src]))
        e=np.einsum('qpn,n->qp',h,w); g=np.sum(abs(e)**2,axis=1)/60/power.sum()
        values=np.loadtxt(exports/f'exports/realized_gain/{cid}_realized_gain.csv',delimiter=',',skiprows=1)
        assert len(values)==n and np.allclose(values[:,:2],np.stack([t,p],-1))
        discrepancy=float(np.max(abs(db(g)-values[:,3])))
        peak=int(np.argmax(g)); gd=db(g); relative=gd-gd[peak]; grid=gd.reshape(180,360)
        maxima=(grid==maximum_filter(grid,size=3,mode=('nearest','wrap')))
        candidates=np.flatnonzero(maxima.ravel() & (relative>=-10))
        order=candidates[np.argsort(g[candidates])[::-1]]; chosen=[]
        for j in order:
            if not chosen or np.max(q[chosen]@q[j])<np.cos(np.deg2rad(5)):
                chosen.append(int(j))
        # Half-power component around global peak; seam at +/-180 is explicitly joined.
        mask=(relative>=-3).reshape(180,360)
        tiled=np.tile(mask,(1,3)); lab,_=label(tiled)
        ti,pi=divmod(peak,360); component=lab==lab[ti,pi+360]
        central=component[:,360:720]; main_fraction=float(area[central.ravel()].sum()/area.sum())
        other=[j for j in chosen if not central.ravel()[j]]
        eff=float(np.sum(g*area)/(4*np.pi))
        phase=np.angle(w); circular=float(abs(np.mean(np.exp(1j*phase))))
        row=dict(command_id=cid,peak_gain_dBi=float(gd[peak]),peak_theta_deg=float(t[peak]),peak_phi_deg=float(p[peak]),
                 competing_peak_count_minus3dB=sum(relative[j]>=-3 for j in chosen)-1,
                 distinct_peak_count_minus10dB=len(chosen),main_halfpower_solid_angle_fraction=main_fraction,
                 other_peak_relative_db=float(relative[other[0]]) if other else None,
                 coverage_minus3dB=float(area[relative>=-3].sum()/area.sum()),
                 coverage_minus10dB=float(area[relative>=-10].sum()/area.sum()),
                 weak_below_minus30dB=float(area[relative<-30].sum()/area.sum()),
                 front_radiated_fraction=float(np.sum(g[forward]*area[forward])/np.sum(g*area)),
                 sphere_integrated_realized_efficiency=eff,
                 theta_polarized_energy_fraction=float(np.sum(abs(e[:,0])**2*area)/np.sum(abs(e)**2*area[:,None])),
                 effective_excited_ports=float(power.sum()**2/np.sum(power**2)),
                 phase_circular_coherence=circular,total_incident_W=float(power.sum()),export_db_max_difference=discrepancy)
        rows.append(row); peaks[cid]=[dict(theta=float(t[j]),phi=float(p[j]),gain_dBi=float(gd[j]),relative_dB=float(relative[j])) for j in chosen[:24]]
        maps[cid]=grid
    hc=cartesian(t,p,h)
    gram=np.einsum('qpn,qpm,q->nm',hc.conj(),hc,area)/(4*np.pi*60)
    eig=np.linalg.eigvalsh(gram).real
    geometry=ArrayGeometry(); wl=geometry.wavelength_m
    summary=dict(geometry=dict(wavelength_mm=wl*1000,row_pitch_lambda=.015/wl,col_pitch_lambda=.0068/wl,
                              center_span_mm=[45,20.4],nominal_array_factor_hpbw_deg=[float(np.rad2deg(.886*wl/(4*.015))),float(np.rad2deg(.886*wl/(4*.0068)))],
                              boresight='+x',row='+z',column='+y'),
                 complex_excitation_dof=32,normalized_power_pattern_amp_phase_real_dof=30,phase_only_power_pattern_real_dof=15,
                 radiation_gram_eigenvalues=eig.tolist(),radiation_effective_rank=float(eig.sum()**2/np.sum(eig**2)),
                 commands=rows,peaks=peaks,interpretation='Peak clusters separated by >=5 deg; competing lobes are not all certified grating lobes. Solid-angle weights sin(theta).',
                 data_boundary='24 diagnostic commands are synthesized from one exposed HFSS state; they are not new independent direct simulations.')
    dump(target/'summary.json',summary)
    with (target/'command_statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,4,figsize=(16,9),layout='constrained')
    for ax,cid in zip(axs.ravel(),['A01','A03','A04','A06','A08','B01','B02','B03','C03','D03','C04','D04']):
        im=ax.imshow(maps[cid]-maps[cid].max(),origin='lower',extent=[-180,180,0,180],vmin=-40,vmax=0,aspect='auto',cmap='turbo')
        ax.set_title(cid); ax.set_xlabel('phi (deg)'); ax.set_ylabel('theta (deg)')
    fig.colorbar(im,ax=axs,label='Gain relative to command peak (dB)'); fig.savefig(target/'beam_patterns.png',dpi=150);plt.close(fig)
    print('MACRO',json.dumps({k:v for k,v in summary.items() if k not in ['commands','peaks']},ensure_ascii=False),flush=True)

class CompactVectorCNN(nn.Module):
    """Shared 4x4 aperture CNN; Cartesian IQ avoids polar-coordinate singularities.

    Analytic propagation and transverse projection have no trainable parameters.
    Every command is applied exclusively by a linear complex contraction.
    """
    def __init__(self):
        super().__init__()
        self.encoder=DirectionFourierEncoder(4)
        self.branch=ApertureCorrectionBranch(31,24,depth=2,output_channels=6)
        # A nonzero nominal field avoids the zero-field stationary point of amplitude-only supervision.
        with torch.no_grad(): self.branch.head.bias[2]=1.
        self.register_buffer('geometry',_geometry_maps())
        self.register_buffer('positions',torch.tensor(ArrayGeometry().positions_m.reshape(16,3),dtype=torch.float32))
    def forward(self,q):
        feat=self.encoder(q[None])[0,:, :,None,None].expand(-1,-1,4,4)
        raw=self.branch(torch.cat([feat,self.geometry[None].expand(len(q),-1,-1,-1)],1))
        raw=raw.reshape(-1,3,2,16)
        field=torch.complex(raw[:,:,0],raw[:,:,1])
        field=field-q[:,:,None]*torch.sum(q[:,:,None]*field,dim=1,keepdim=True)
        phase=(2*math.pi/ArrayGeometry().wavelength_m)*(q@self.positions.T)
        return field*torch.exp(1j*phase)[:,None,:]

def commands(rng,n):
    result=[]; fam=[]
    pos=ArrayGeometry().positions_m.reshape(16,3); k=2*np.pi/ArrayGeometry().wavelength_m
    for i in range(n):
        f=i%4
        if f==0: w=np.exp(1j*rng.uniform(-np.pi,np.pi,16))
        elif f==1: w=np.exp(1j*(rng.integers(0,16,16)*2*np.pi/16))
        elif f==2: w=rng.normal(size=16)+1j*rng.normal(size=16)
        else:
            q=rng.normal(size=3);q/=np.linalg.norm(q);w=np.exp(-1j*k*(pos@q))
        w=w/np.linalg.norm(w);w*=np.exp(-1j*np.angle(w[0]));result.append(w);fam.append(f)
    return np.asarray(result,dtype=np.complex64),np.asarray(fam)

def prepare(root):
    if (root/'protocol.json').exists(): return
    t,p,q,hs=ffd('exportelement_new');h=cartesian(t,p,hs).astype(np.complex64)
    tiles=(t//10).astype(int)*36+((p+180)//10).astype(int)
    rng=np.random.default_rng(CFG['split_seed'])
    # Stratify tiles in 30deg theta bands and 60deg phi bands; never split trajectories across sets.
    assignment={}
    for tb in range(6):
        for pb in range(6):
            blocks=[a*36+b for a in range(tb*3,tb*3+3) for b in range(pb*6,pb*6+6)]
            rng.shuffle(blocks)
            for si,chosen in enumerate([blocks[:10],blocks[10:13],blocks[13:15],blocks[15:]]):
                assignment.update({x:si for x in chosen})
    part=np.array([assignment[x] for x in tiles]); assert len(assignment)==648
    scale=float(np.sqrt(np.mean(abs(h[part==0])**2)))
    signatures=[]; splitinfo={}
    for si,name in enumerate(SPLITS):
        ids=np.flatnonzero(part==si); w,fam=commands(np.random.default_rng(29570+si),256 if si==0 else 96)
        sig={np.round(np.stack([x.real,x.imag])*1e5).astype(int).tobytes() for x in w}
        assert all(not(sig & old) for old in signatures); signatures.append(sig)
        # Whole synthetic trajectory: 3 frames; each station stays within one held tile.
        trng=np.random.default_rng(29700+si); count=[64,24,64,120][si]; trajectory=[]; strata=[]
        allowed=np.array(sorted(x for x,v in assignment.items() if v==si))
        front=[x for x in allowed if abs((x%36)*10-175)<80 and 20<=(x//36)*10<=150]
        for ti in range(count):
            if ti%3==0:
                chosen=[int(trng.choice(front))]*4; strata.append('clustered')
            elif ti%3==1:
                chosen=trng.choice(front,4,replace=False);strata.append('distributed_front')
            else:
                chosen=trng.choice(allowed,4,replace=False);strata.append('full_sphere_stress')
            base=[];vel=[]
            for tile in chosen:
                a=tile//36*10+int(trng.integers(2,8))+.5
                b=tile%36*10-180+int(trng.integers(2,8))+.5
                base.append((a,b));vel.append(trng.integers(-1,2,size=2))
            frames=[]
            for frame in [-1,0,1]:
                angles=np.array(base)+frame*np.array(vel)
                ix=((angles[:,0]-.5).astype(int)*360+(angles[:,1]+179.5).astype(int))
                assert (part[ix]==si).all();frames.append(ix)
            trajectory.append(frames)
        trajectory=np.array(trajectory)
        folder=root/('dev' if si<2 else 'reserved');folder.mkdir(parents=True,exist_ok=True)
        np.savez(folder/f'{name}.npz',q=q[ids].astype(np.float32),h=h[ids],theta=t[ids],phi=p[ids],ids=ids,
                 commands=w,family=fam,trajectory_q=q[trajectory].astype(np.float32),trajectory_h=h[trajectory],
                 trajectory_indices=trajectory,trajectory_strata=np.array(strata))
        splitinfo[name]=dict(directions=len(ids),tiles=len(set(tiles[ids])),commands=len(w),trajectories=count,frames=3)
    model=CompactVectorCNN(); count=sum(v.numel() for v in model.parameters());assert count<=CFG['parameter_budget']
    mac=sum(m.weight.numel()*16 for m in model.modules() if isinstance(m,nn.Conv2d))
    protocol=dict(config=CFG,scale=scale,split=splitinfo,model_parameters=count,conv_mac_per_direction=mac,
                  largest_single_feature_map_floats=31*16,architecture='Cartesian transverse dual-polarization operator, ideal phase x shared aperture CNN',
                  model_output='[Q,3,16] complex; equivalent two transverse polarizations; gain=sum(abs(Hw)^2)/(60*sum(abs(w)^2))',
                  isolation='trainer only opens dev/train.npz and dev/validation.npz; reserved labels opened after selection freeze',
                  historical_exposure='same complete HFSS table already used for diagnostics; holdout is prospective command/trajectory and internal angular-block generalization, not unseen physical state',
                  diagnostic_exports_used_for_training=False,hash_policy='no hashes per user; separate paths, fixed config, split intersection checks and freeze markers',
                  acceptance=dict(global_db_mae=.5,strong_db_mae=.35,fourpoint_db_mae=.3,bottleneck_over_p95_db=.75,regret_median=.05,regret_p95=.10,seed_nmse_std_db=.25))
    dump(root/'protocol.json',protocol);dump(root/'split_audit.json',dict(disjoint_angle_tiles=True,disjoint_commands_mod_global_phase=True,trajectories_stay_in_partition=True,**splitinfo))
    print('PREPARED',count,mac,splitinfo,flush=True)

def gains(h,w):
    # h [Q,3,16]; w [C,16] -> [C,Q]
    e=torch.einsum('qpn,cn->cqp',h,w)
    return e.abs().square().sum(-1)/60/w.abs().square().sum(-1)[:,None]

def tdb(g):return 10*torch.log10(g.clamp_min(1e-5))
def objective_loss(pred,truth):
    amp=(pred.clamp_min(1e-12).sqrt()-truth.clamp_min(1e-12).sqrt()).square().mean()/truth.mean().clamp_min(1e-8)
    power=(pred-truth).square().mean()/truth.square().mean().clamp_min(1e-8)
    robust=nn.functional.huber_loss(tdb(pred),tdb(truth),delta=1.)
    return amp+.1*power+.03*robust

def optimize(h,mode='phase',starts=8,steps=160,seed=0,initial=None):
    """Batched numerical feasible search, no global-optimum claim. h [cases,4,3,16]."""
    h=h.detach();gen=torch.Generator(device=h.device).manual_seed(seed); b=len(h)
    z=torch.randn((b,starts,16),device=h.device,generator=gen)+1j*torch.randn((b,starts,16),device=h.device,generator=gen)
    # Analytic matched-polarization starts for each target; include sum of matched directions.
    matched=h[:, :,1,:].conj()
    z[:,:min(4,starts)]=matched[:,:min(4,starts)]
    if initial is not None:z[:,-1]=initial.detach()
    if mode=='phase':param=nn.Parameter(torch.angle(z)); to_w=lambda:torch.exp(1j*param)/4
    else:
        param=nn.Parameter(torch.view_as_real(z).clone())
        def to_w():
            w=torch.view_as_complex(param);return w/w.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    opt=torch.optim.Adam([param],lr=.065)
    best=torch.full((b,),-1e30,device=h.device);bw=torch.zeros((b,16),device=h.device,dtype=torch.complex64)
    for step in range(steps+1):
        w=to_w(); e=torch.einsum('bkpn,bsn->bskp',h,w);g=e.abs().square().sum(-1)/60
        gd=tdb(g);minimum=gd.min(-1).values;sv,ix=minimum.max(1)
        improve=sv>best;best=torch.maximum(best,sv);bw[improve]=w.detach()[torch.arange(b,device=h.device),ix][improve]
        if step==steps: break
        tau=1.2*(1-step/steps)+.12
        loss=(tau*torch.logsumexp(-gd/tau,-1)).mean()
        opt.zero_grad();loss.backward();opt.step()
    return bw.detach(),best.detach()

@torch.no_grad()
def predict(model,q,scale,batch=1024):
    return torch.cat([model(q[i:i+batch])*scale for i in range(0,len(q),batch)])

def data_load(root,name):
    folder='dev' if name in ['train','validation'] else 'reserved'
    with np.load(root/folder/f'{name}.npz') as d:return {k:d[k] for k in d.files}

def train_one(root,recipe,seed,parent=None):
    if (root/'selection_frozen.json').exists() or (root/'holdout_opened.json').exists():raise RuntimeError('Training is closed after freeze')
    out=root/'runs'/f'{recipe}_{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        old_result=read(out/'result.json')
        if old_result['converged'] or old_result['epochs']>=CFG['extension_max_epochs']:return old_result
    protocol=read(root/'protocol.json');scale=protocol['scale']; torch.manual_seed(seed);np.random.seed(seed)
    model=CompactVectorCNN().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=1e-5)
    schedule=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=12,factor=.5,threshold=.01,threshold_mode='abs',min_lr=1e-5)
    if parent:model.load_state_dict(torch.load(root/'runs'/f'{parent}_{seed}'/'best.pt',weights_only=False)['model'])
    operator_enabled = recipe=='B'
    if parent:
        operator_enabled=read(root/'runs'/f'{parent}_{seed}'/'result.json').get('operator_enabled',parent!='A')
    tr=data_load(root,'train');va=data_load(root,'validation')
    q=torch.tensor(tr['q'],device='cuda');h=torch.tensor(tr['h'],device='cuda');w=torch.tensor(tr['commands'],device='cuda')
    # Fixed stratified validation subset selected without labels.
    vix=np.random.default_rng(29590).choice(len(va['q']),2048,replace=False)
    vq=torch.tensor(va['q'][vix],device='cuda');vh=torch.tensor(va['h'][vix],device='cuda');vw=torch.tensor(va['commands'],device='cuda')
    truth=gains(vh,vw); gen=torch.Generator(device='cuda').manual_seed(seed+1000)
    tq=torch.tensor(tr['trajectory_q'].reshape(-1,4,3),device='cuda');th=torch.tensor(tr['trajectory_h'].reshape(-1,4,3,16),device='cuda')
    hard_w=None
    if recipe in 'CDE':hard_w=optimize(th,starts=8,steps=120,seed=seed)[0]
    history=[];best=1e30;best_epoch=0;elapsed_prior=0.;start=time.perf_counter();converged=False
    if (out/'latest.pt').exists():
        ck=torch.load(out/'latest.pt',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);schedule.load_state_dict(ck['scheduler']);gen.set_state(ck['generator']);history=ck['history'];best=ck['best'];best_epoch=ck['best_epoch'];elapsed_prior=ck['seconds']
        if ck.get('hard_w') is not None:hard_w=ck['hard_w']
    print('TRAIN',recipe,seed,'resume',history[-1]['epoch'] if history else 0,flush=True)
    for epoch in range((history[-1]['epoch'] if history else 0)+1,CFG['extension_max_epochs']+1):
        model.train();total=0.
        for step in range(CFG['steps_per_epoch']):
            ix=torch.randint(len(q),(CFG['batch'],),generator=gen,device='cuda');ci=torch.randint(len(w),(64,),generator=gen,device='cuda')
            hp=model(q[ix])*scale;truth_batch=gains(h[ix],w[ci]); pred=gains(hp,w[ci]);loss=objective_loss(pred,truth_batch)
            if operator_enabled:
                inner=(hp.conj()*h[ix]).sum((1,2));phase=inner/inner.abs().clamp_min(1e-8)
                aligned=hp*phase[:,None,None]
                loss=loss+CFG['op_weight']*(aligned-h[ix]).abs().square().mean()/h[ix].abs().square().mean().clamp_min(1e-8)
            if hard_w is not None:
                ixh=torch.randint(len(tq),(16,),generator=gen,device='cuda');ht=model(tq[ixh].reshape(-1,3)).reshape(-1,4,3,16)*scale
                # Both teacher commands and periodically model-optimized training commands.
                hw=hard_w[ixh];gp=torch.einsum('bkpn,bn->bkp',ht,hw).abs().square().sum(-1)/60
                gt=torch.einsum('bkpn,bn->bkp',th[ixh],hw).abs().square().sum(-1)/60
                error=tdb(gp)-tdb(gt);loss=loss+CFG['hard_weight']*nn.functional.huber_loss(tdb(gp),tdb(gt))
                if recipe in 'DE':loss=loss+CFG['over_weight']*torch.relu(error-.1).square().mean()
                if recipe=='E':
                    target=torch.softmax(-tdb(gt),-1);loss=loss+CFG['rank_weight']*nn.functional.kl_div(torch.log_softmax(-tdb(gp),-1),target,reduction='batchmean')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();total+=loss.item()
        if epoch%100==0 and hard_w is not None:
            hp=predict(model,tq.reshape(-1,3),scale).reshape(-1,4,3,16)
            mined=optimize(hp,starts=4,steps=80,seed=seed+epoch)[0]
            hard_w[::2]=mined[::2]
        if epoch%CFG['validate_every']!=0:continue
        model.eval()
        with torch.no_grad():
            pg=gains(predict(model,vq,scale),vw)
            nmse=float(10*torch.log10((pg.sqrt()-truth.sqrt()).square().sum()/truth.sum()))
            mae=float((tdb(pg)-tdb(truth)).abs().mean());train_nmse=float(10*torch.log10((pred.sqrt()-truth_batch.sqrt()).square().sum()/truth_batch.sum()))
        schedule.step(nmse)
        if nmse<best:
            best=nmse;best_epoch=epoch;torch.save(dict(model=model.state_dict(),epoch=epoch,scale=scale,recipe=recipe,seed=seed),out/'best.pt')
        history.append(dict(epoch=epoch,loss=total/CFG['steps_per_epoch'],validation_nmse_db=nmse,validation_db_mae=mae,train_batch_nmse_db=train_nmse,best=best,lr=opt.param_groups[0]['lr']))
        old=[x for x in history if x['epoch']<=epoch-CFG['plateau_epochs']]
        improvement=old[-1]['best']-best if old else 999.
        converged=epoch>=CFG['min_epochs'] and improvement<CFG['plateau_db']
        if epoch%100==0 or converged or epoch==CFG['extension_max_epochs']:
            elapsed=elapsed_prior+time.perf_counter()-start
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),generator=gen.get_state(),history=history,best=best,best_epoch=best_epoch,seconds=elapsed,hard_w=hard_w),out/'latest.pt')
            dump(out/'history.json',history);dump(root/'progress.json',dict(recipe=recipe,seed=seed,epoch=epoch,best_nmse_db=best,elapsed=elapsed))
            print(recipe,seed,epoch,round(best,3),'seconds',round(elapsed,1),flush=True)
        if converged:break
    result=dict(recipe=recipe,seed=seed,parent=parent,operator_enabled=operator_enabled,epochs=epoch,best_epoch=best_epoch,best_validation_nmse_db=best,converged=converged,last_200_improvement_db=improvement,seconds=elapsed_prior+time.perf_counter()-start)
    dump(out/'result.json',result);return result

def evaluate_dev(root,recipe):
    scale=read(root/'protocol.json')['scale'];va=data_load(root,'validation');rows=[]
    tq=torch.tensor(va['trajectory_q'].reshape(-1,4,3),device='cuda');th=torch.tensor(va['trajectory_h'].reshape(-1,4,3,16),device='cuda')
    teacher,reference=optimize(th,starts=32,seed=29799,steps=160)
    for seed in SEEDS:
        model=CompactVectorCNN().cuda();model.load_state_dict(torch.load(root/'runs'/f'{recipe}_{seed}'/'best.pt',weights_only=False)['model']);model.eval()
        hp=predict(model,tq.reshape(-1,3),scale).reshape(-1,4,3,16)
        w,predminimum=optimize(hp,starts=8,seed=29799,steps=160)
        real=torch.einsum('bkpn,bn->bkp',th,w).abs().square().sum(-1)/60
        rp=real.min(-1).values;rg=10**(reference/10);regret=(1-rp/torch.maximum(rg,rp)).cpu().numpy()
        over=predminimum-tdb(rp)
        result=read(root/'runs'/f'{recipe}_{seed}'/'result.json')
        rows.append(dict(seed=seed,median_regret=float(np.median(regret)),p95_regret=float(np.percentile(regret,95)),over_p95_db=float(torch.quantile(over,.95)),validation_nmse_db=result['best_validation_nmse_db']))
    dump(root/f'dev_{recipe}.json',rows)
    return dict(recipe=recipe,score=float(np.mean([r['p95_regret'] for r in rows])),nmse=float(np.mean([r['validation_nmse_db'] for r in rows])),rows=rows)

def campaign(root):
    if (root/'selection_frozen.json').exists():return
    assert read(root/'protocol.json')['config']==CFG
    scores=[]
    for recipe in ['A','B']:
        for seed in SEEDS:train_one(root,recipe,seed)
        scores.append(evaluate_dev(root,recipe));dump(root/'development_selection.json',scores)
    best=scores[0]
    if scores[1]['score']<best['score'] and scores[1]['nmse']<=best['nmse']+.3:best=scores[1]
    parent=best['recipe']
    for recipe in ['C','D','E']:
        for seed in SEEDS:train_one(root,recipe,seed,parent=parent)
        candidate=evaluate_dev(root,recipe);scores.append(candidate);dump(root/'development_selection.json',scores)
        # Predeclared global guard: do not sacrifice >0.3 dB NMSE for optimization tail.
        if candidate['score']<best['score'] and candidate['nmse']<=best['nmse']+.3:best=candidate;parent=recipe
    seed=min(best['rows'],key=lambda x:(x['p95_regret'],x['validation_nmse_db']))['seed']
    dump(root/'selection_frozen.json',dict(recipe=best['recipe'],representative_seed=seed,seeds=SEEDS,scores=scores,selection='validation-only phase-only p95 regret; <=0.3 dB NMSE guard',holdout_opened=False))
    print('FROZEN',best['recipe'],seed,flush=True)

def metrics_numpy(pred,true,theta):
    e=db(pred)-db(true);peak=true.max(1,keepdims=True);relative=db(true/peak);aw=np.sin(np.deg2rad(theta))[None,:]*np.ones_like(e)
    def stat(mask):
        if not mask.any():return dict(count=0)
        a=e[mask];ww=aw[mask]
        return dict(count=int(mask.sum()),mae_db=float(np.mean(abs(a))),bias_db=float(np.mean(a)),p95_abs_db=float(np.percentile(abs(a),95)),
                    over_fraction=float(np.mean(a>0)),solid_angle_mae_db=float(np.sum(abs(a)*ww)/ww.sum()))
    masks=dict(all=np.ones_like(e,dtype=bool),strong=relative>=-10,medium=(relative<-10)&(relative>=-20),weak=(relative<-20)&(relative>=-30),deep=relative<-30)
    return dict(amplitude_nmse_db=float(db(np.sum((np.sqrt(pred)-np.sqrt(true))**2)/np.sum(true))),
                power_nmse_db=float(db(np.sum((pred-true)**2)/np.sum(true**2))),bins={name:stat(mask) for name,mask in masks.items()})

def eval_targets(root,model,data,scale,part):
    q=torch.tensor(data['trajectory_q'],device='cuda');h=torch.tensor(data['trajectory_h'],device='cuda')
    n=len(q); hp=predict(model,q.reshape(-1,3),scale).reshape(n,3,4,3,16); output={};arrays={}
    for mode in ['phase','amplitude_phase']:
        flat=h.reshape(-1,4,3,16);predflat=hp.reshape_as(flat)
        wn,pminimum=optimize(predflat,mode,8,160,30001)
        real=torch.einsum('bkpn,bn->bkp',flat,wn).abs().square().sum(-1)/60
        predicted=torch.einsum('bkpn,bn->bkp',predflat,wn).abs().square().sum(-1)/60
        wr,_=optimize(flat,mode,32,160,30001,initial=wn)
        ref=torch.einsum('bkpn,bn->bkp',flat,wr).abs().square().sum(-1)/60
        rm=real.min(-1).values;refmin=torch.maximum(ref.min(-1).values,rm);regret=(1-rm/refmin).cpu().numpy()
        err=(tdb(predicted)-tdb(real)).cpu().numpy();bottleneck=real.argmin(-1).cpu().numpy();be=err[np.arange(len(err)),bottleneck]
        # Feasible reference plus rigorous but loose spectral upper bound.
        Q=torch.einsum('bkpn,bkpm->bknm',flat.conj(),flat)/60
        upper=torch.linalg.eigvalsh(Q.mean(1)).real[:,-1]
        ub_gap=(1-refmin/upper).clamp_min(0).cpu().numpy()
        metrics=dict(median_regret=float(np.median(regret)),p95_regret=float(np.percentile(regret,95)),worst_regret=float(regret.max()),
                     fourpoint_mae_db=float(abs(err).mean()),bottleneck_bias_db=float(be.mean()),bottleneck_over_p95_db=float(np.percentile(be,95)),
                     bottleneck_over_fraction=float(np.mean(be>0)),bottleneck_identity_accuracy=float(np.mean(predicted.argmin(-1).cpu().numpy()==bottleneck)),
                     true_min_gain_dBi_quantiles=np.percentile(db(rm.cpu().numpy()),[5,50,95]).tolist(),reference_min_gain_dBi_quantiles=np.percentile(db(refmin.cpu().numpy()),[5,50,95]).tolist(),
                     spectral_bound_gap_median=float(np.median(ub_gap)),constraint_max_power_error=float(abs(wn.abs().square().sum(-1)-1).max()),
                     constraint_max_magnitude_error=float(abs(wn.abs()-.25).max()) if mode=='phase' else None)
        strata={}
        for name in np.unique(data['trajectory_strata']):
            ix=np.repeat(data['trajectory_strata']==name,3);strata[str(name)]=dict(frames=int(ix.sum()),regret_p95=float(np.percentile(regret[ix],95)),min_gain_p5_dBi=float(np.percentile(db(rm.cpu().numpy()[ix]),5)))
        metrics['strata']=strata;output[mode]=metrics
        arrays[mode+'_error_db']=err;arrays[mode+'_real_gain']=real.cpu().numpy();arrays[mode+'_pred_gain']=predicted.cpu().numpy();arrays[mode+'_reference_min']=refmin.cpu().numpy();arrays[mode+'_weights']=wn.cpu().numpy();arrays[mode+'_regret']=regret
        if mode=='phase':
            warm=[];last=None;elapsed=[]
            for frame in range(3):
                start=time.perf_counter();ww,_=optimize(hp[:,frame],mode,4,80,30001+frame,initial=last);torch.cuda.synchronize();elapsed.append((time.perf_counter()-start)/n);warm.append(ww);last=ww
            warm=torch.stack(warm,1);wt=wn.reshape(n,3,16)
            def jump(w):return torch.sqrt((2-2*(w[:,1:].conj()*w[:,:-1]).sum(-1).abs()).clamp_min(0))
            wg=torch.einsum('bfkpn,bfn->bfkp',h,warm).abs().square().sum(-1)/60
            output['continuity']=dict(warm4starts80steps_phase_jump_median=float(jump(warm).median()),cold8starts160steps_phase_jump_median=float(jump(wt).median()),
                                      warm_min_p5_dBi=float(torch.quantile(tdb(wg.min(-1).values).flatten(),.05)),cold_min_p5_dBi=float(torch.quantile(tdb(rm),.05)),
                                      amortized_gpu_seconds_per_frame=float(np.mean(elapsed)),comparison='unequal budgets, exploratory warm-start efficiency, not a controlled causal claim or FPGA latency')
    np.savez_compressed(root/f'{part}_target_arrays.npz',**arrays)
    return output,arrays

def evaluate(root):
    if (root/'final_report.json').exists():return
    selection=read(root/'selection_frozen.json');scale=read(root/'protocol.json')['scale'];recipe=selection['recipe']
    dump(root/'holdout_opened.json',dict(opened=True,selection=selection['recipe'],purpose='one-shot reporting; do not retrain in this campaign'))
    data=data_load(root,'holdout');cal=data_load(root,'calibration');results=[];representative=None
    for seed in SEEDS:
        model=CompactVectorCNN().cuda();model.load_state_dict(torch.load(root/'runs'/f'{recipe}_{seed}'/'best.pt',weights_only=False)['model']);model.eval()
        q=torch.tensor(data['q'],device='cuda');h=torch.tensor(data['h'],device='cuda');w=torch.tensor(data['commands'],device='cuda')
        pred=gains(predict(model,q,scale),w).cpu().numpy();true=gains(h,w).cpu().numpy()
        stats=metrics_numpy(pred,true,data['theta']);stats['seed']=seed
        stats['families']={str(f):metrics_numpy(pred[data['family']==f],true[data['family']==f],data['theta']) for f in range(4)}
        results.append(stats)
        if seed==selection['representative_seed']:representative=model;rep_pred=pred;rep_true=true
    targets,arr=eval_targets(root,representative,data,scale,'holdout')
    caltargets,ca=eval_targets(root,representative,cal,scale,'calibration')
    score=ca['phase_error_db'].reshape(-1,3,4).max(axis=(1,2))
    order=min(len(score),math.ceil((len(score)+1)*.95));radius=float(np.sort(score)[order-1])
    hs=arr['phase_error_db'].reshape(-1,3,4).max(axis=(1,2));covered=hs<=radius;rate=float(covered.mean());n=len(covered)
    z=1.96;den=1+z*z/n;center=(rate+z*z/(2*n))/den;half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
    convergence=[read(root/'runs'/f'{recipe}_{s}'/'result.json') for s in SEEDS]
    global_mae=np.mean([r['bins']['all']['mae_db'] for r in results]);strong=np.mean([r['bins']['strong']['mae_db'] for r in results]);std=float(np.std([r['amplitude_nmse_db'] for r in results],ddof=1));four=targets['phase']
    acceptance=dict(converged=all(r['converged'] for r in convergence),global_db_mae=bool(global_mae<=.5),strong_db_mae=bool(strong<=.35),
                    fourpoint_mae=bool(four['fourpoint_mae_db']<=.3),bottleneck_p95=bool(four['bottleneck_over_p95_db']<=.75),regret_median=bool(four['median_regret']<=.05),regret_p95=bool(four['p95_regret']<=.1),seed_std=bool(std<=.25))
    report=dict(selection=selection,forward=results,targets=targets,convergence=convergence,seed_nmse_std_db=std,acceptance=acceptance,
                calibration=dict(trajectories=len(score),joint_three_frame_four_target_radius_db=radius,holdout_joint_coverage=rate,wilson95=[center-half,center+half],
                                 caveat='fixed subtraction calibrates reporting only; it does not change the maximizing weights; trajectory exchangeability is assumed'),
                boundaries=['Fixed one-state HFSS; no OTA or FPGA results','24 diagnostic exports were not used as independent blind labels','Directions historically exposed; prospective trajectory/command IDs only','Numerical multistart reference, not globally optimal','Synthetic full-sphere stress includes satellite-infeasible directions; evaluate front strata separately'])
    dump(root/'final_report.json',report)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for r in results:
        names=['strong','medium','weak','deep'];axs[0].plot(names,[r['bins'][x]['mae_db'] for x in names],marker='o',label=str(r['seed']))
    axs[0].legend();axs[0].set_ylabel('Gain MAE (dB)');axs[0].set_title('Reference-defined intensity bins')
    axs[1].hist(arr['phase_error_db'].ravel(),bins=70);axs[1].set_xlabel('Predicted - true target gain (dB)')
    axs[2].hist(arr['phase_regret']*100,bins=40);axs[2].set_xlabel('Phase-only regret (%)')
    fig.savefig(root/'evaluation.png',dpi=160);plt.close(fig)
    print('EVALUATED',acceptance,'STD',std,flush=True)

def selftest(root):
    tr=data_load(root,'train');q=torch.tensor(tr['q'][:32],device='cuda');model=CompactVectorCNN().cuda();h=model(q)
    assert torch.max(abs((h*q[:,:,None]).sum(1)))<1e-5
    w=torch.tensor(tr['commands'][:8],device='cuda');g=gains(h,w);assert torch.allclose(g,gains(h,1j*w),atol=1e-7)
    loss=objective_loss(g,torch.ones_like(g));loss.backward();assert model.branch.head.weight.grad.norm()>0
    qt=torch.einsum('qpn,qpm->qnm',h.conj(),h)/60
    quadratic=torch.einsum('cn,qnm,cm->cq',w.conj(),qt,w).real
    assert torch.allclose(g,quadratic,atol=1e-6)
    hw=torch.tensor(tr['trajectory_h'][:2,0],device='cuda');ww,_=optimize(hw,starts=4,steps=4)
    assert torch.max(abs(ww.abs()-.25))<1e-6
    dump(root/'selftest.json',dict(transverse=True,common_phase_invariance=True,nonzero_gradient=True,quadratic_contraction=True,phase_constraints=True))
    print('SELFTEST PASSED',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['macro','prepare','train','evaluate','selftest']);parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=True
    args.root.mkdir(parents=True,exist_ok=True)
    {'macro':macro,'prepare':prepare,'train':campaign,'evaluate':evaluate,'selftest':selftest}[args.action](args.root)

if __name__=='__main__':main()
