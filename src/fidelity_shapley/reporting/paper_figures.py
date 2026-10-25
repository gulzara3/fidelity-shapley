"""Publication figures (Figs. 2–6 of the paper), rendered from result JSON files.

Every number drawn comes from the archived result files; nothing is digitised.
Each figure is saved as 600-dpi PNG and vector PDF, together with the exact data
plotted (``data_figN.json``). Fig. 1 (architecture) is a static diagram kept in
``docs/figures`` as an editable draw.io source.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

from ..config import CMAPSS_SENSORS as CN, LITERATURE_UNINFORMATIVE  # noqa: E402

SER = {"font.family": "DejaVu Serif", "font.size": 8, "axes.linewidth": 0.7,
       "axes.edgecolor": "#333333", "xtick.color": "#222222", "ytick.color": "#222222",
       "axes.labelcolor": "#111111", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
       "xtick.major.size": 2.5, "ytick.major.size": 2.5, "mathtext.fontset": "dejavuserif"}
CB = {"blue": "#0173B2", "orange": "#DE8F05", "green": "#029E73", "red": "#D55E00",
      "pink": "#CC78BC", "tan": "#CA9161", "gold": "#ECB000", "grey": "#7F7F7F"}
UNINF = set(LITERATURE_UNINFORMATIVE)
CONST = {"s1", "s5", "s10", "s16", "s18", "s19"}
R: Path = Path("results")
OUT: Path = Path("results/figures/paper")
P = S = CH = SK = CM = None


def _load(results: Path, out: Path) -> None:
    global R, OUT, P, S, CH, SK, CM
    R, OUT = Path(results), Path(out)
    OUT.mkdir(parents=True, exist_ok=True)
    P = json.loads((R / "paper_numbers_dt_final.json").read_text())
    S = json.loads((R / "scores" / "shapley.json").read_text())
    CH = list(P["shapley_skab_mean"].keys())
    seeds = [str(s) for s in P["seeds"]]
    SK = np.array([S["skab"][s] for s in seeds])
    CM = np.array([S["cmapss"][s] for s in seeds])


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=600, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    print("saved", OUT / name)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color="#E3E3E3", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plabel(ax, t, x=-0.02, y=1.04):
    ax.text(x, y, t, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="left", va="bottom")


def fig_values():
    plt.rcParams.update(SER)
    m,sd=SK.mean(0),SK.std(0); o=np.argsort(-m)
    fig=plt.figure(figsize=(6.9,2.95))
    a1=fig.add_axes([0.2,0.26,0.23,0.63]); a2=fig.add_axes([0.52,0.26,0.47,0.63])
    idx=o[::-1]
    a1.barh(range(8),m[idx],xerr=sd[idx],color=CB['blue'],height=0.62,error_kw=dict(lw=0.7,capsize=2,capthick=0.7,ecolor='#222222'),zorder=3)
    a1.axvline(0,color='#555555',lw=0.6,zorder=2)
    a1.set_yticks(range(8)); a1.set_yticklabels([CH[i] for i in idx],fontsize=7.3); a1.set_xlabel('Fidelity value $\\phi_i$'); a1.set_xlim(-0.015,0.115)
    grid(a1,'x'); plabel(a1,'(a) SKAB: exact Shapley',x=-0.75)
    cm,cs=CM.mean(0),CM.std(0)
    cols=[CB['green'] if k not in UNINF else (CB['red'] if k in CONST else CB['gold']) for k in CN]
    a2.bar(range(21),cm,yerr=cs,color=cols,width=0.7,edgecolor='none',error_kw=dict(lw=0.7,capsize=1.8,capthick=0.7,ecolor='#222222'),zorder=3)
    a2.axhline(0,color='#555555',lw=0.6,zorder=2)
    a2.set_xticks(range(21)); a2.set_xticklabels(CN,rotation=90,fontsize=7); a2.set_ylabel('Fidelity value $\\phi_i$'); a2.set_xlim(-0.7,20.7); a2.set_ylim(-0.3,4.35)
    grid(a2,'y'); plabel(a2,'(b) C-MAPSS FD001: Monte Carlo Shapley',x=-0.08)
    a2.annotate('s6: missed by variance\nscreening, recovered blind',xy=(5,cm[5]+cs[5]+0.06),xytext=(5,3.72),fontsize=6.4,ha='center',va='bottom',
                arrowprops=dict(arrowstyle='-|>',lw=0.7,color='#333333',mutation_scale=7,relpos=(0.5,0.0)))
    h=[Rectangle((0,0),1,1,color=CB['green']),Rectangle((0,0),1,1,color=CB['red']),Rectangle((0,0),1,1,color=CB['gold'])]
    fig.legend(h,['informative (literature)','constant (ledger-flagged)','s6 (uninformative, not flagged)'],loc='lower center',
               bbox_to_anchor=(0.5,-0.01),ncol=3,frameon=False,fontsize=7,handlelength=1.2,columnspacing=1.6)
    json.dump({'SKAB':{c:[float(m[i]),float(sd[i])] for i,c in enumerate(CH)},'CMAPSS':{c:[float(cm[i]),float(cs[i])] for i,c in enumerate(CN)}},open(OUT/'data_fig2.json','w'),indent=1)
    _save(fig,'Fig2_fidelity_values'); plt.rcParams.update(plt.rcParamsDefault)

def fig_validation():
    plt.rcParams.update(SER)
    tsd=[S['trace_sd'][str(s)] for s in range(5)]
    M=json.load(open(R/'scores'/'merged.json'))
    g=[M[str(s)]['Temperature']['group_value'] for s in range(5)]; sm=[M[str(s)]['Temperature']['sum_members'] for s in range(5)]
    sp=json.load(open(R/'scores'/'spotcheck.json'))
    fig=plt.figure(figsize=(6.9,2.35))
    a=fig.add_axes([0.105,0.2,0.22,0.66]); b=fig.add_axes([0.425,0.2,0.22,0.66]); c=fig.add_axes([0.755,0.2,0.225,0.66])
    a.bar(range(5),tsd,color=CB['blue'],width=0.6,zorder=3)
    a.set_xticks(range(5)); a.set_xticklabels([str(i) for i in range(1,6)]); a.set_xlabel('Seed'); a.set_ylabel('Late-trace s.d. of $\\hat\\phi$')
    a.set_ylim(0,0.025); grid(a); plabel(a,'(a) MC convergence',x=-0.3)
    a.text(0.97,0.95,'max $|\\phi|=3.76$',transform=a.transAxes,ha='right',va='top',fontsize=6.8)
    x=np.arange(5); w=0.36
    b.bar(x-w/2,g,w,color=CB['blue'],label='merged super-player',zorder=3)
    b.bar(x+w/2,sm,w,color=CB['green'],label='sum of members',zorder=3)
    b.set_xticks(x); b.set_xticklabels([str(i) for i in range(1,6)]); b.set_xlabel('Seed'); b.set_ylabel('Group value'); b.set_ylim(0,0.26)
    grid(b); plabel(b,'(b) Group merging',x=-0.3)
    b.legend(fontsize=6.3,frameon=False,loc='upper left',ncol=1,handlelength=1.0,borderaxespad=0.2)
    c.plot([0.65,1.1],[0.65,1.1],ls='--',color='#888888',lw=0.8,zorder=2)
    for sd_,mk,col,lab in (('0','o',CB['blue'],'seed 1'),('1','s',CB['orange'],'seed 2')):
        pts=[(v['mask_rmse'],v['retrain_rmse']) for k,v in sp.items() if k.split('|')[0]==sd_]
        c.scatter([p[0] for p in pts],[p[1] for p in pts],s=16,marker=mk,color=col,alpha=0.85,edgecolor='white',lw=0.4,zorder=3,label=lab)
    m=np.array([v['mask_rmse'] for v in sp.values()]); r=np.array([v['retrain_rmse'] for v in sp.values()])
    c.set_xlim(0.65,1.1); c.set_ylim(0.65,1.1); c.set_xticks([0.7,0.8,0.9,1.0,1.1]); c.set_yticks([0.7,0.8,0.9,1.0,1.1])
    c.set_xlabel('Mask-model utility (RMSE)'); c.set_ylabel('Retrained-twin RMSE'); grid(c,'both'); plabel(c,'(c) Utility validation',x=-0.3)
    c.text(0.05,0.95,f'Spearman $\\rho={stats.spearmanr(m,r)[0]:.3f}$\nPearson $r={stats.pearsonr(m,r)[0]:.3f}$',transform=c.transAxes,va='top',fontsize=6.8)
    c.legend(fontsize=6.3,frameon=False,loc='lower right',handletextpad=0.2,borderaxespad=0.2)
    json.dump({'mc_late_sd':tsd,'merged':g,'sum_members':sm,'validation':sp},open(OUT/'data_fig3.json','w'),indent=1)
    _save(fig,'Fig3_validation'); plt.rcParams.update(plt.rcParamsDefault)

def fig_closedloop():
    plt.rcParams.update(SER)
    B=json.load(open(R/'scores'/'bandwidth.json'))
    st={'FidelityShapley (ours)':(CB['blue'],'-','o','FidelityShapley (ours)'),'VoI-proxy':(CB['orange'],'--','D','VoI proxy'),
        'Mutual information':(CB['green'],'--','^','Mutual information'),'Variance':(CB['red'],'--','v','Variance'),
        'mRMR (greedy)':(CB['pink'],'--','s','mRMR (greedy)'),'Random':(CB['tan'],':','P','Random')}
    fig=plt.figure(figsize=(6.9,3.3))
    axs=[fig.add_axes([0.085,0.27,0.4,0.64]),fig.add_axes([0.585,0.27,0.4,0.64])]
    titles=['(a) SKAB: telemetry RMSE','(b) C-MAPSS FD001: RUL RMSE']; ylabs=['Fidelity RMSE (standardised units)','Fidelity RMSE (RUL cycles)']
    out={}
    for ax,ds,t,yl in zip(axs,['skab','cmapss'],titles,ylabs):
        ks=sorted(map(int,B[ds]['FidelityShapley (ours)']))
        full=np.mean([B[ds]['FidelityShapley (ours)'][str(ks[-1])][s] for s in '01234']); band=1.05*full
        ax.axhspan(0,band,color='#EDEDED',zorder=0)
        ax.axhline(full,color='#999999',lw=0.6,ls='-',zorder=1)
        nudge={m:(i-2.5)*(0.035 if ds=='skab' else 0.09) for i,m in enumerate(st)}
        out[ds]={}
        for m,(col,ls,mk,lab) in st.items():
            mu=np.array([np.mean([B[ds][m][str(k)][s] for s in '01234']) for k in ks]); sd=np.array([np.std([B[ds][m][str(k)][s] for s in '01234']) for k in ks])
            xs=np.array(ks)+nudge[m]
            ax.errorbar(xs,mu,yerr=sd,color=col,ls=ls,lw=1.3 if m.startswith('Fid') else 1.0,marker=mk,ms=3.4,capsize=1.6,elinewidth=0.6,capthick=0.6,zorder=4 if m.startswith('Fid') else 3,label=lab)
            kst=next(k for k,v in zip(ks,mu) if v<=band); j=ks.index(kst)
            ax.scatter([xs[j]],[mu[j]],marker='*',s=95,color=col,edgecolor='black',lw=0.5,zorder=6)
            out[ds][m]={'k':ks,'mean':mu.tolist(),'sd':sd.tolist(),'kstar':kst}
        ax.set_xlabel('Number of transmitted channels $k$'); ax.set_ylabel(yl); grid(ax,'both'); plabel(ax,t,x=-0.02)
        if ds=='skab': ax.set_xticks(ks); ax.set_ylim(0.68,1.12); ax.set_xlim(0.6,8.4)
        else: ax.set_xticks(ks); ax.set_ylim(12.5,40); ax.set_xlim(1,22)
    hs=[Line2D([0],[0],color=c,ls=l,lw=1.2,marker=mk,ms=3.6) for c,l,mk,_ in st.values()]+[Line2D([0],[0],ls='',marker='*',ms=9,mfc='#BBBBBB',mec='k',mew=0.5),Rectangle((0,0),1,1,color='#EDEDED')]
    fig.legend(hs,[v[3] for v in st.values()]+['$k^*$ (first $k$ within the band)','within 5% of full-observation fidelity'],loc='lower center',ncol=4,frameon=False,fontsize=7,bbox_to_anchor=(0.5,0.0),columnspacing=1.2,handlelength=2.4)
    json.dump(out,open(OUT/'data_fig4.json','w'),indent=1)
    _save(fig,'Fig4_closed_loop'); plt.rcParams.update(plt.rcParamsDefault)

def fig_interaction():
    plt.rcParams.update(SER)
    I=json.load(open(R/'scores'/'interactions.json'))
    Ls,Lc=np.array(I['skab']['per_channel_total']),np.array(I['cmapss']['per_channel_total'])
    ps,pc=np.abs(SK.mean(0)),np.abs(CM.mean(0))
    fig=plt.figure(figsize=(3.6,3.25)); ax=fig.add_axes([0.17,0.17,0.79,0.79])
    ax.scatter(Ls,ps,s=26,marker='o',color=CB['blue'],edgecolor='white',lw=0.5,zorder=3,label='SKAB (8 channels)')
    inf=np.array([k not in UNINF for k in CN])
    ax.scatter(Lc[inf],pc[inf],s=26,marker='s',color=CB['orange'],edgecolor='white',lw=0.5,zorder=3,label='C-MAPSS, informative (14)')
    ax.scatter(Lc[~inf],pc[~inf],s=26,marker='s',facecolor='white',edgecolor=CB['orange'],lw=1.0,zorder=3,label='C-MAPSS, uninformative (7)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Interaction strength $\\Lambda_i=\\sum_{j}|I_{ij}|$'); ax.set_ylabel('Absolute fidelity value $|\\phi_i|$')
    grid(ax,'both'); ax.set_ylim(8e-4,8)
    rs=stats.spearmanr(Ls,ps)[0]; rc=stats.spearmanr(Lc,pc)[0]
    ax.text(0.03,0.97,f'within SKAB: $\\rho_s={rs:.2f}$\nwithin C-MAPSS: $\\rho_s={rc:.2f}$',transform=ax.transAxes,va='top',fontsize=6.6)
    ax.legend(fontsize=6.3,frameon=False,loc='lower right',handletextpad=0.2,borderaxespad=0.2)
    json.dump({'skab':{'Lambda':Ls.tolist(),'abs_phi':ps.tolist(),'channels':CH},'cmapss':{'Lambda':Lc.tolist(),'abs_phi':pc.tolist(),'channels':CN}},open(OUT/'data_fig5.json','w'),indent=1)
    _save(fig,'Fig5_interaction'); plt.rcParams.update(plt.rcParamsDefault)

def fig_defect():
    plt.rcParams.update(SER)
    import seaborn as sns
    D=json.load(open(R/'scores'/'defect_map_v2.json'))
    conds=['noise|0.25','noise|0.5','noise|1.0','drift|1.0','drift|2.0','drift|4.0','stuck_at|0.1','stuck_at|0.3','stuck_at|0.5','staleness|2','staleness|5','staleness|10']
    lab=['noise 0.25','noise 0.5','noise 1.0','drift 1','drift 2','drift 4','stuck-at 0.1','stuck-at 0.3','stuck-at 0.5','staleness 2','staleness 5','staleness 10']
    M=np.array([[np.mean([D[s][f'{ch}|{c}'] for s in D]) for c in conds] for ch in CH])
    phi=SK.mean(0); vul=M.mean(1); rho=stats.spearmanr(phi,vul)[0]
    fig=plt.figure(figsize=(6.9,5.2))
    a=fig.add_axes([0.19,0.53,0.66,0.41]); cax=fig.add_axes([0.87,0.53,0.018,0.41]); b=fig.add_axes([0.19,0.075,0.66,0.33])
    cmap=sns.color_palette('rocket_r',as_cmap=True)
    imh=a.imshow(M,cmap=cmap,vmin=0,vmax=M.max(),aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i,j]>=0.1: a.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=6.8,color='white' if M[i,j]>0.2 else '#111111')
    a.set_xticks(range(12)); a.set_xticklabels(lab,rotation=40,ha='right',fontsize=7)
    a.set_yticks(range(8)); a.set_yticklabels(CH,fontsize=7)
    for gx in (2.5,5.5,8.5): a.axvline(gx,color='white',lw=1.5)
    a.tick_params(length=0)
    for s in a.spines.values(): s.set_visible(False)
    cb=fig.colorbar(imh,cax=cax); cb.set_label('$\\Delta$ fidelity RMSE',fontsize=7.5); cb.ax.tick_params(labelsize=6.8); cb.outline.set_visible(False)
    plabel(a,'(a) Defect-to-fidelity degradation map',x=-0.28,y=1.03)
    b.scatter(phi,vul,s=26,color=CB['blue'],zorder=3,edgecolor='white',lw=0.5)
    off={'Temperature':(-6,-9,'right'),'Accelerometer1RMS':(6,3,'left'),'VolumeFlowRateRMS':(-6,2,'right'),'Accelerometer2RMS':(6,3,'left'),
         'Thermocouple':(-6,-8,'right'),'Current':(6,3,'left'),'Voltage':(6,-8,'left'),'Pressure':(6,2,'left')}
    for i,nm in enumerate(CH):
        dx,dy,ha=off[nm]; b.annotate(nm,(phi[i],vul[i]),xytext=(dx,dy),textcoords='offset points',fontsize=6.4,ha=ha)
    b.set_xlabel('Fidelity value $\\phi_i$ (exact Shapley)'); b.set_ylabel('Mean defect vulnerability\n($\\Delta$RMSE)')
    b.set_xlim(-0.012,0.118); b.set_ylim(-0.004,0.066); grid(b,'both')
    b.text(0.03,0.95,f'Spearman $\\rho={rho:.2f}$ ($n=8$)',transform=b.transAxes,va='top',fontsize=6.8)
    plabel(b,'(b) Value–vulnerability relationship',x=-0.26,y=1.04)
    json.dump({'rows':CH,'cols':lab,'dRMSE_mean5seeds':M.tolist(),'phi':phi.tolist(),'vulnerability':vul.tolist()},open(OUT/'data_fig6.json','w'),indent=1)
    _save(fig,'Fig6_defect_value'); plt.rcParams.update(plt.rcParamsDefault)



def render_all(results: Path = Path("results"), out: Path = Path("results/figures/paper")) -> None:
    _load(results, out)
    for f in (fig_values, fig_validation, fig_closedloop, fig_interaction, fig_defect):
        f()
