"""Create publication figures exclusively from retained paper-v2 results."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("--interventions",type=Path,required=True)
p.add_argument("--cost",type=Path,required=True)
p.add_argument("--broad",type=Path)
p.add_argument("--out",type=Path,required=True)
args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
j=json.loads((args.interventions/"metrics.json").read_text())
c=json.loads((args.cost/"summary.json").read_text())
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
scenes=("nominal","low_friction","camera_shift","mass_low","mass_high","displacement_shift")
titles=("Nominal","Declared low friction","Camera mismatch","Low mass","High mass","1.35× goal")
policies=("fixed_1p6_unbound","fixed_4p8_unbound","integrated_support_gate")
labels=("Fixed 1.6 s","Fixed 4.8 s","Profile router")
fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
for ax,scene,title in zip(axes.flat,scenes,titles):
    rows=[j[scene]["policies"][k] for k in policies]
    complete=np.array([r["completion"]["count"] for r in rows]);unsafe=np.array([r["unsafe"]["count"] for r in rows]);reject=np.array([r["rejection"]["count"] for r in rows])
    y=np.arange(3)
    ax.barh(y,complete,color="#247f77",label="Completed")
    ax.barh(y,unsafe,left=complete,color="#d75b51",label="Unsafe execution")
    ax.barh(y,reject,left=complete+unsafe,color="#aeb7c4",label="Rejected (incomplete)")
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_title(title,loc="left",fontweight="bold",pad=12)
    for i,r in enumerate(rows):
        ax.text(102,i,f'{r["unsafe"]["count"]} unsafe · {r["completion"]["count"]} done',va="center",fontsize=9)
    ax.set_xlim(0,145);ax.set_xticks([0,25,50,75,100]);ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
for ax in axes[-1]:ax.set_xlabel("Roots / 100 (rejections stay in denominator)")
handles,names=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,names,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.02))
fig.suptitle("Safety and task utility under frozen support routing",fontsize=18,fontweight="bold",y=.985)
fig.text(.5,.945,"600 new roots · 5 s motion + 0.5 s hold · same-state paired candidates",ha="center",color="#5b687a")
fig.tight_layout(rect=[0,.08,1,.93])
for ext in ("svg","png"):fig.savefig(args.out/f"routing-utility.{ext}",dpi=180,bbox_inches="tight")
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
ratio=c["marginal_cost_comparison"]["paired_ratio_c_delta_over_c_full"]
value=ratio["estimate_ratio_of_means"];lo,hi=ratio["ci95_percentile"]
axes[0].errorbar(value,0,xerr=[[value-lo],[hi-value]],fmt="o",color="#247f77",capsize=7,ms=9)
axes[0].axvline(1,color="#65758b",ls="--");axes[0].set_yticks([0],["Marginal Δ / full"]);axes[0].set_xlim(.96,1.025);axes[0].set_ylim(-.5,.5)
axes[0].set_xlabel("Ratio of mean per-root median times (95% paired CI)");axes[0].set_title("Small average difference",loc="left",fontweight="bold")
ks=(1,2,4,8); full=[c["amortized_bound_incremental"][str(k)]["full_total_ms"]["p95"] for k in ks];inc=[c["amortized_bound_incremental"][str(k)]["incremental_total_ms"]["p95"] for k in ks]
x=np.arange(4);axes[1].bar(x-.17,full,width=.34,label="Full",color="#487fab");axes[1].bar(x+.17,inc,width=.34,label="Incremental",color="#d09a45")
axes[1].set_xticks(x,[str(k) for k in ks]);axes[1].set_xlabel("Analytical parent-cost amortization K");axes[1].set_ylabel("P95 across per-root median totals (ms)");axes[1].set_title("No tail-latency improvement",loc="left",fontweight="bold");axes[1].legend(frameon=False)
fig.suptitle("UR5e verification cost · 60 new roots · 3 rotated repeats",fontsize=16,fontweight="bold")
fig.tight_layout(rect=[0,.02,1,.90])
for ext in ("svg","png"):fig.savefig(args.out/f"verification-cost.{ext}",dpi=180,bbox_inches="tight")
plt.close(fig)
if args.broad:
    b=json.loads((args.broad/"metrics.json").read_text())
    rows=json.loads((args.broad/"per_root.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    ks=("fixed_1p6","fixed_4p8","floor_0p015_shortest")
    labels=("Fixed 1.6 s","Fixed 4.8 s","Declared-floor rule")
    values=[b["aggregate"][k]["completion"] for k in ks]
    x=np.arange(3)
    axes[0].bar(x,[v["rate"]*100 for v in values],color=["#d75b51","#487fab","#247f77"])
    for i,v in enumerate(values):
        lo,hi=v["wilson95"]
        axes[0].errorbar(i,v["rate"]*100,yerr=[[max(0.,v["rate"]-lo)*100],[max(0.,hi-v["rate"])*100]],fmt="none",ecolor="#253346",capsize=4)
        axes[0].text(i,106,f'{v["count"]}/324',ha="center",fontweight="bold")
    axes[0].set_xticks(x,labels,fontsize=9);axes[0].set_ylim(0,116)
    axes[0].set_ylabel("Task completion % (Wilson 95% interval)")
    axes[0].set_title("Slow fallback completes the frozen sample",loc="left",fontweight="bold",fontsize=11)
    counts=[sum(r["policy_outcomes"]["fixed_1p6"]["unsafe"] for r in rows if r["friction_bin_index"]==i) for i in range(3)]
    axes[1].bar(x,np.array(counts)/108*100,color="#d75b51")
    axes[1].set_xticks(x,["0.015–0.05","0.05–0.20","0.20–0.80"],fontsize=9)
    axes[1].set_xlabel("Friction bin (108 roots each)");axes[1].set_ylabel("Fixed 1.6 s unsafe execution %")
    axes[1].set_ylim(0,112)
    for i,value in enumerate(counts):axes[1].text(i,value/108*100+4,f'{value}/108',ha="center",fontweight="bold")
    axes[1].set_title("Fast failure concentrates at low friction",loc="left",fontweight="bold",fontsize=11)
    fig.suptitle("27-cell physical stress · 324 new roots · full 5.5 s outcomes",fontsize=15,fontweight="bold")
    fig.text(.5,.015,"Floor rule equals fixed 4.8 s on every root; 3× motion duration. Each cell n=12: zero-event upper bound 24.25%.",ha="center",fontsize=9,color="#5b687a")
    fig.tight_layout(rect=[0,.07,1,.91])
    for ext in ("svg","png"):fig.savefig(args.out/f"physical-stress.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.interventions/"metrics.json",args.cost/"summary.json")}
if args.broad:sources[str(args.broad/"metrics.json")]=hashlib.sha256((args.broad/"metrics.json").read_bytes()).hexdigest();sources[str(args.broad/"per_root.json")]=hashlib.sha256((args.broad/"per_root.json").read_bytes()).hexdigest()
outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file() and p.name!="figure_manifest.json"}
(args.out/"figure_manifest.json").write_text(json.dumps({"sources":sources,"outputs":outputs,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+"\n")
