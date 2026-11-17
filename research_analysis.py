"""Read-only research analysis; final reporting never updates a Q-table."""
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_runner import ROOT, ORIGINAL, load_json, model_index

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS={'baseline':'#333333','control':'#7f8c8d','intermediate':'#2980b9','fine':'#8e44ad','gamma':'#16a085','alpha':'#c0392b','epsilon':'#d68910','rewards':'#2471a3','combined':'#b03a2e','focused':'#117864'}


def stage_data(stage,split='validation'):
    result={}
    for p in (ROOT/'evaluations'/stage).glob(f'*/{split}/summary.json'):
        result[p.parents[1].name]=load_json(p)
    return result


def scores(stage,key,split):
    with (ROOT/'evaluations'/stage/key/split/'episodes.csv').open() as f:
        return np.array([int(row['score']) for row in csv.DictReader(f)])


def family_ranking(data):
    groups=defaultdict(list)
    for key,s in data.items():
        if key=='baseline': continue
        groups[key.rsplit('_s',1)[0]].append(s)
    ranking=[]
    for arm,replicas in groups.items():
        ranking.append(dict(arm=arm,replicas=len(replicas),
            worst_median=min(s['median'] for s in replicas),
            worst_p10=min(s['p10'] for s in replicas),
            worst_lower_tail=min(s['lower_tail_mean'] for s in replicas),
            mean_median=float(np.mean([s['median'] for s in replicas])),
            mean_score=float(np.mean([s['mean'] for s in replicas])),
            mean_success=float(np.mean([s['success'] for s in replicas]))))
    return sorted(ranking,key=lambda x:(x['worst_median'],x['worst_p10'],x['worst_lower_tail'],x['mean_median'],x['mean_score']),reverse=True)


def choose_representative(candidates):
    """Validation-only practical median equivalence band; prefer lower-tail safety."""
    best_median=max(summary['median'] for summary in candidates.values())
    eligible={key:summary for key,summary in candidates.items() if summary['median']>=.9*best_median}
    return max(eligible,key=lambda key:(eligible[key]['p10'],eligible[key]['lower_tail_mean'],eligible[key]['median'],eligible[key]['mean']))


def plot_screening():
    out=ROOT/'graphs'; out.mkdir(exist_ok=True)
    for stage in ['pilot','screening6000','screening10000']:
        data=stage_data(stage)
        if stage=='screening6000': data.update(stage_data('focused_pilot'))
        data={k:v for k,v in data.items() if k!='baseline'}
        if not data: continue
        arms=sorted(set(k.rsplit('_s',1)[0] for k in data))
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for ax,metric,title in zip(axes,['median','p10','mean'],['Median pipes','10th percentile pipes','Mean pipes']):
            for i,arm in enumerate(arms):
                vals=[s[metric] for k,s in data.items() if k.startswith(arm+'_s')]
                ax.bar(i,np.mean(vals),color=COLORS.get(arm),alpha=.7)
                ax.scatter([i]*len(vals),vals,color='black',s=18,zorder=3)
            ax.set_xticks(range(len(arms)),arms,rotation=45,ha='right'); ax.set_title(title); ax.grid(axis='y',alpha=.2)
            peak=max(s[metric] for s in data.values())
            ax.set_ylim(0,peak*1.15 if peak else 1)
            if not peak: ax.text(.5,.5,'All values = 0',ha='center',transform=ax.transAxes,color='#555555')
        fig.suptitle(f'{stage}: validation on identical 100 unseen courses; dots = independent training seeds')
        fig.tight_layout(); fig.savefig(out/f'{stage}_validation.png',dpi=160); plt.close(fig)


def histories(entry):
    all_rows=[]
    for run in entry['runs']:
        p=Path(run)/'training.csv'
        with p.open() as f: all_rows.extend(csv.DictReader(f))
    all_rows.sort(key=lambda r:int(r['episode']))
    assert [int(r['episode']) for r in all_rows]==list(range(1,entry['episodes']+1))
    return all_rows


def rolling(values,width=100):
    values=np.array(values,dtype=float); sums=np.r_[0,np.cumsum(values)]; idx=np.arange(len(values)); start=np.maximum(0,idx-width+1)
    return (sums[idx+1]-sums[start])/(idx-start+1)


def plot_learning():
    out=ROOT/'graphs'; out.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(12,6))
    by_arm=defaultdict(list)
    for entry in model_index().values():
        h=histories(entry); by_arm[entry['arm']].append(rolling([r['score'] for r in h]))
    for arm,curves in by_arm.items():
        length=min(map(len,curves)); array=np.array([v[:length] for v in curves]); x=np.arange(1,length+1)
        ax.plot(x,array.mean(axis=0),label=arm,color=COLORS[arm])
        ax.fill_between(x,array.min(axis=0),array.max(axis=0),color=COLORS[arm],alpha=.08)
    ax.set_xlabel('Completed training episodes'); ax.set_ylabel('100-episode mean pipes'); ax.set_title('Training trajectories: line = replica mean; shading = replica range')
    ax.legend(ncol=3); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'learning_curves.png',dpi=160); plt.close(fig)


def paired_bootstrap(candidate,baseline,repetitions=5000):
    rng=np.random.default_rng(20261117)
    estimates=[]
    for _ in range(repetitions):
        idx=rng.integers(0,len(candidate),len(candidate))
        a,b=candidate[idx],baseline[idx]
        estimates.append([a.mean()-b.mean(),np.median(a)-np.median(b),np.quantile(a,.1)-np.quantile(b,.1)])
    estimates=np.array(estimates)
    return dict(mean_delta=float(candidate.mean()-baseline.mean()),mean_delta_ci=np.quantile(estimates[:,0],[.025,.975]).tolist(),
        median_delta=float(np.median(candidate)-np.median(baseline)),median_delta_ci=np.quantile(estimates[:,1],[.025,.975]).tolist(),
        p10_delta=float(np.quantile(candidate,.1)-np.quantile(baseline,.1)),p10_delta_ci=np.quantile(estimates[:,2],[.025,.975]).tolist(),
        win_fraction=float(np.mean(candidate>baseline)),tie_fraction=float(np.mean(candidate==baseline)))


if __name__=='__main__':
    plot_screening(); plot_learning()
    print(json.dumps(family_ranking(stage_data('screening10000')),indent=2))

def plot_final(selection):
    out=ROOT/'graphs'; out.mkdir(exist_ok=True)
    chosen=selection['recommended_key']
    original=scores('locked_test','baseline','test')
    candidate=scores('locked_test',chosen,'test')
    keys=['baseline']+selection['finalist_keys']
    data=[scores('locked_test',key,'test') for key in keys]
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    axes[0].boxplot(data,tick_labels=keys,showmeans=True)
    axes[0].tick_params(axis='x',rotation=45); axes[0].set_ylabel('Pipes passed (capped at 100)'); axes[0].set_title('Locked test: 200 identical, previously unseen courses')
    for key,values in zip(keys,data):
        arm=key.rsplit('_s',1)[0]
        axes[1].step(np.sort(values),np.arange(1,len(values)+1)/len(values),where='post',label=key,color=COLORS.get(arm),alpha=1 if key in ('baseline',chosen) else .4,linestyle=':' if key.endswith('_s848484') else '--' if key.endswith('_s424242') else '-')
    axes[1].set_xlabel('Pipes passed'); axes[1].set_ylabel('Cumulative fraction'); axes[1].set_title('Score distribution (curves farther right are better)'); axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'final_test_distributions.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].scatter(original,candidate,alpha=.4,s=20,color='#117864'); axes[0].plot([0,100],[0,100],'--',color='black'); axes[0].set_xlabel('Original baseline score'); axes[0].set_ylabel('Recommended agent score'); axes[0].set_title('Paired scores on identical held-out courses')
    baseline=load_json(ROOT/'evaluations/locked_test/baseline/test/summary.json')
    chosen_summary=load_json(ROOT/'evaluations/locked_test'/chosen/'test/summary.json')
    metrics=['median','p10','mean']; positions=np.arange(3)
    axes[1].bar(positions-.18,[baseline[m] for m in metrics],width=.36,label='Original',color='#7f8c8d')
    axes[1].bar(positions+.18,[chosen_summary[m] for m in metrics],width=.36,label='Recommended',color='#117864'); axes[1].set_xticks(positions,['Median','10th percentile','Mean']); axes[1].set_ylabel('Pipes passed'); axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'paired_improvement.png',dpi=160); plt.close(fig)
    index=model_index(); fig,ax=plt.subplots(figsize=(11,6))
    grouped=defaultdict(list)
    for key,summary in stage_data('locked_test','test').items():
        arm='baseline' if key=='baseline' else key.rsplit('_s',1)[0]
        grouped[arm].append((key,summary))
    for arm,points in sorted(grouped.items()):
        ax.scatter([s['training_seconds']/60 for key,s in points],[s['median'] for key,s in points],color=COLORS.get(arm),s=50,label=arm,alpha=.8)
        for key,summary in points:
            if key in ('baseline',chosen):
                ax.scatter(summary['training_seconds']/60,summary['median'],s=140,facecolors='none',edgecolors='black')
                ax.annotate(key,(summary['training_seconds']/60,summary['median']),fontsize=9,xytext=(6,6),textcoords='offset points')
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),title='Configuration')
    ax.set_xlabel('Recorded training wall minutes per policy'); ax.set_ylabel('Held-out median pipes'); ax.set_title('Training cost versus reliable performance (episode budgets differ)'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'training_cost.png',dpi=160); plt.close(fig)
    curves=[(key,histories(index[key])) for key in selection['finalist_keys'] if key.startswith(selection['recommended_arm']+'_s')]
    fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
    for key,h in curves:
        x=[int(r['episode']) for r in h]
        axes[0].plot(x,rolling([r['reward'] for r in h]),alpha=.8,label=key)
        axes[1].plot(x,rolling([r['score'] for r in h]),alpha=.8)
        axes[2].plot(x,[float(r['epsilon']) for r in h],alpha=.8)
    axes[0].legend(ncol=3)
    axes[0].set_ylabel('100-episode mean return'); axes[1].set_ylabel('100-episode mean score'); axes[2].set_ylabel('Epsilon'); axes[2].set_xlabel('Training episode')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f"Recommended family: {selection['recommended_arm']} (three independent training seeds)")
    fig.tight_layout(); fig.savefig(out/'recommended_learning.png',dpi=160); plt.close(fig)
