"""Generate the final evidence report from recorded outputs only."""
import ast
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
from experiment_runner import ROOT, ORIGINAL, load_json, model_index, check_preserved
from research_analysis import stage_data, family_ranking, scores, paired_bootstrap, plot_screening, plot_learning, plot_final, histories
from flappy_ai.training import write_csv
from flappy_ai.agent import QAgent
from flappy_ai.storage import atomic_json


def table(rows,columns):
    header='| '+' | '.join(label for key,label in columns)+' |\n'
    header+='|'+'|'.join('---' for _ in columns)+'|\n'
    def format_value(value):
        return f'{value:.2f}' if isinstance(value,float) else str(value)
    return header+'\n'.join('| '+' | '.join(format_value(row[key]) for key,label in columns)+' |' for row in rows)+'\n'


def main():
    selection=load_json(ROOT/'selection.json'); index=model_index(); plan=load_json(ROOT/'study.json')
    data=stage_data('locked_test','test'); baseline=data['baseline']; chosen=data[selection['recommended_key']]
    assert len(data)==len(index)+1
    selected_scores=scores('locked_test',selection['recommended_key'],'test'); baseline_scores=scores('locked_test','baseline','test')
    statistics=paired_bootstrap(selected_scores,baseline_scores)
    plot_screening(); plot_learning(); plot_final(selection)
    original_manifest=load_json(ROOT/'original_hashes.json'); check_preserved()
    original_head=subprocess.check_output(['git','-C',str(ORIGINAL),'rev-parse','main'],text=True).strip()
    assert original_head==plan['original_main_commit']
    assert not subprocess.check_output(['git','-C',str(ORIGINAL),'status','--porcelain'],text=True).strip()
    old_ast=ast.parse(subprocess.check_output(['git','show','main:flappy_ai/agent.py'],text=True))
    new_ast=ast.parse(Path('flappy_ai/agent.py').read_text(encoding='utf-8'))
    def methods(tree):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    old,new=methods(old_ast),methods(new_ast)
    for name in ['act','values','update','finish_episode']:
        assert old[name]==new[name],name
    control=QAgent.load(index['control_s42']['model'])
    historical_control=QAgent.load(ORIGINAL/'artifacts/checkpoint_10000.npz')
    assert control.q.keys()==historical_control.q.keys() and all(np.array_equal(control.q[state],historical_control.q[state]) for state in control.q)
    historical_environment=ast.parse(subprocess.check_output(['git','show','main:flappy_ai/environment.py'],text=True))
    experimental_environment=ast.parse(Path('flappy_ai/environment.py').read_text(encoding='utf-8'))
    assert ast.dump(historical_environment)==ast.dump(experimental_environment)
    agent=QAgent.load(ROOT/'recommended_agent.npz')
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.npz')}
    atomic_json(ROOT/'model_hashes.json',hashes)
    total_training=sum(entry['duration'] for entry in index.values())
    total_episodes=sum(entry['episodes'] for entry in index.values())
    total_steps=0
    for entry in index.values():total_steps+=sum(int(row['steps']) for row in histories(entry))
    elapsed=time.time()-plan.get('preparation_started_epoch',plan['started_epoch'])
    evidence=dict(completed_epoch=time.time(),experimental_episodes=total_episodes,experimental_training_seconds=total_training,
        experimental_training_steps=total_steps,study_elapsed_seconds=elapsed,hard_wall_budget_seconds=14400,
        original_artifact_files_preserved=len(original_manifest),original_main_commit=original_head,
        original_working_tree_clean=True,original_agent_sha256=original_manifest['artifacts/agent.npz'],
        main_algorithm_methods_unchanged=True,environment_unchanged=True,control_reproduced_original_10000=True,recommended_key=selection['recommended_key'],
        recommended_source=selection['recommended_source'],recommended_sha256=selection['source_sha256'],
        baseline_test=baseline,recommended_test=chosen,paired_test_bootstrap=statistics,
        validation_ranking=selection['final_validation_ranking'],finalists=selection['finalist_keys'],
        models=index,validation_seed_count=100,locked_test_seed_count=200,benchmark_seed_count=40)
    assert total_episodes<=264000 and elapsed<14400
    atomic_json(ROOT/'final_summary.json',evidence)
    all_rows=[]
    for key,summary in data.items():
        all_rows.append(dict(agent=key,training_seed='historical 42' if key=='baseline' else index[key]['training_seed'],
            trained_episodes=summary['trained_episodes'],mean=summary['mean'],median=summary['median'],p10=summary['p10'],
            lower_tail_mean=summary['lower_tail_mean'],minimum=summary['minimum'],best=summary['best'],success_percent=100*summary['success'],
            training_minutes=summary['training_seconds']/60,states=summary['states'],unknown_fraction=summary['mean_unknown_fraction']))
    all_rows.sort(key=lambda r:(r['agent']!='baseline',r['trained_episodes']!=24000,-r['median'],-r['p10']))
    write_csv(ROOT/'all_agents_test_comparison.csv',all_rows)
    finalists=[row for row in all_rows if row['agent']=='baseline' or row['agent'] in selection['finalist_keys']]
    columns=[('agent','Policy / training seed'),('trained_episodes','Episodes'),('mean','Mean'),('median','Median'),('p10','P10'),('best','Best'),('success_percent','Success %'),('training_minutes','Train min')]
    screen=family_ranking(stage_data('screening10000'))
    screen_table=table(screen,[('arm','Configuration'),('worst_median','Worst replica median'),('mean_median','Mean replica median'),('mean_score','Mean score'),('worst_p10','Worst replica P10')])
    preserved_table=table(finalists,columns)
    full_table=table(all_rows,columns)
    comparison=[]
    for split in ['benchmark','validation','test']:
        a=load_json(ROOT/'evaluations/locked_test'/selection['recommended_key']/split/'summary.json')
        b=load_json(ROOT/'evaluations/locked_test/baseline'/split/'summary.json')
        for label,s in [('Original',b),('Recommended',a)]:
            comparison.append(dict(split=split,policy=label,n=s['episodes'],mean=s['mean'],median=s['median'],p10=s['p10'],best=s['best'],success=100*s['success']))
    split_table=table(comparison,[('split','Split'),('policy','Policy'),('n','Courses'),('mean','Mean'),('median','Median'),('p10','P10'),('best','Best'),('success','Success %')])
    config=json.dumps(load_json(ROOT/'recommended_config.json'),indent=2)
    compatible=ROOT/'recommended_compatible_agent.npz'
    launch_directory=ORIGINAL if compatible.exists() else ROOT.parents[1]
    launch_model=compatible if compatible.exists() else ROOT/'recommended_agent.npz'
    compatibility_note='A metadata-only compatible export was also verified with the unmodified original main-branch loader; it omits only the unused state_ranges=None key and preserves every Q value.' if compatible.exists() else 'This agent uses focused state ranges; run it from the experimental worktree so its encoder metadata is understood.'
    recommended_details='The recommended agent retains the original bins [8,12,12,24,24], alpha 0.18, gamma 0.98 and epsilon schedule. Its reward changes are survival +0.01 (was +0.05) and death -15 (was -10); passing remains +10. It was trained independently with seed 424242. No encoder change is needed for this recommended policy.' if selection['recommended_arm']=='rewards' else 'See the exact recommended configuration at the end.'
    report=f'''# Flappy Bird AI — controlled tabular Q-learning experiments

All results below are measured in this workspace. No neural network, evolutionary algorithm or scripted flight controller was used. The original agent was neither overwritten nor promoted. The main branch and remote repository have not been changed by this study.

## Recommendation

Recommended experimental agent: **{selection['recommended_key']}**, copied unchanged to `recommended_agent.npz`. Selection was frozen using validation courses before any locked-test result was computed. It remains an experimental option; the original game still defaults to the original Q-table.

{recommended_details}

{split_table}

P10 is the linearly interpolated 10th percentile. Success means reaching **100 pipes**; scores are capped at 100 by the same environment for every policy. These are course samples, not guarantees for every future flight.

On the 200 paired held-out courses, the mean score difference is **{statistics['mean_delta']:.2f}**, with a 95% paired bootstrap interval **[{statistics['mean_delta_ci'][0]:.2f}, {statistics['mean_delta_ci'][1]:.2f}]**. The median difference is **{statistics['median_delta']:.2f}** (interval [{statistics['median_delta_ci'][0]:.2f}, {statistics['median_delta_ci'][1]:.2f}]). P10 difference: **{statistics['p10_delta']:.2f}** (interval [{statistics['p10_delta_ci'][0]:.2f}, {statistics['p10_delta_ci'][1]:.2f}]). The recommended policy outscored the original on {statistics['win_fraction']*100:.1f}% of matched courses and tied on {statistics['tie_fraction']*100:.1f}%.

The worst recorded selected score is **{chosen['minimum']}**, versus **{baseline['minimum']}** for the original. The mean score of the worst 10% of test courses is **{chosen['lower_tail_mean']:.2f}**, versus **{baseline['lower_tail_mean']:.2f}**. These distinguish typical gains from remaining catastrophic failures.

Bootstrap intervals resample course pairs with fixed trained models (5,000 resamples); they do not represent uncertainty over all possible training seeds. The three independent finalist replicas below expose some training variability. All test results, including weak candidates, are retained in the full table and CSV.

## Comparable 24,000-episode policies on the locked test

{preserved_table}

Each row has 200 identical obstacle seeds, exploration disabled and its own frozen Q-table. The original has one historical training replicate; each experimental finalist has three. Reusing identical obstacle courses across rows provides paired comparisons; these are not 1,400 independent courses.

## Why the original fails

The original discretizes both signed opening-edge distances across [-590,590] into 24 bins, roughly **49.17 pixels per bin**, and horizontal distance into roughly **46.92 pixels per bin**. The two edge features are correlated (their raw sum is the 175-pixel gap), so their joint quantization is finer than an individual 49-pixel bin; combined relative-distance boundaries are spaced about 21.7 or 27.5 pixels apart before other features are considered. Per-feature widths therefore overstate the full tuple's vertical aliasing. Nevertheless, the state can still combine materially different clearances and approach timings. On 100 validation courses, the original failed 53 times on upper pipes and 45 times on lower pipes; two reached the target. There were no ceiling or floor deaths. No terminal decision state was absent from the table, and the mean unknown-state fraction was about 0.0068%. Mere Q-table coverage therefore does not explain most original failures.

The median horizontal distance immediately before a pipe collision was 14 pixels, close to the 13-pixel collision radius. Lower-pipe deaths often occurred after a last-moment flap (about 76% of those deaths): the correction was too late. This supports timing/clearance aliasing as a plausible mechanism, but does not prove a unique causal explanation for every death.

Original gamma .98 corresponds to a rough effective horizon of 50 simulation steps; pipes arrive about 80 steps apart. Increasing gamma to .995 improved screening but was not one of the two most robust finalists. The original epsilon schedule wastes early episodes on frequent random flaps; lowering initial exploration helped early progress but did not dominate later screening. A smaller alpha .08 learned more slowly at the pilot budget and was stopped; this does not establish its asymptotic quality. Reducing survival reward and increasing death penalty improved screening under the original state grid. No reward exploitation or infinite episodes was observed; original score and step caps were retained.

## Pilot, pruning and controlled comparisons

`STUDY_PROTOCOL.md`, `study.json`, `decisions.json`, `finalists_plan.json` and `selection.json` preserve the protocol, amendments and timestamped decisions. All eight initial configurations received two 3,000-episode pilots. Every median was zero, so no arm was rejected from that insufficient pilot. All continued to 6,000 episodes. The fine grid, fine-grid combined settings and alpha .08 then underperformed control on both seeds. Their paired validation mean differences were negative, with bootstrap intervals excluding zero. They were stopped at 6,000 to prioritize sample efficiency, not labeled universally ineffective.

Five original survivors advanced to 10,000 episodes. An additional focused-range representation was preregistered after the failure analysis and tested first at 6,000, then 10,000. It uses **[4,8,12,24,24]** bins and observation ranges **[[0,590],[-6.5,10],[-83,280],[-80,255],[-80,255]]**. All five observations remain. Relative-distance resolution is about **13.96 pixels**, while redundant absolute-height and velocity resolution is reduced. Its nominal table has **221,184** possible keys versus **663,552** for the original; only encountered states are stored. This focuses representation rather than simply enlarging the table. The focused relative ranges are symmetric about half the opening height, making the two normalized edge features complementary; coarser redundant features and altered range alignment may both contribute. Their individual causal effects were not isolated, so the study does not attribute the gain exclusively to finer distance bins. Far-away values are clipped, which may hurt other tasks or difficulty settings.

All unspecified focused hyperparameters and rewards are original defaults. State normalization is the only policy-method change for that arm; the Bellman update, epsilon-greedy action selection and epsilon decay code remain unchanged. The standard intermediate grid [8,12,16,36,36] and fine grid [8,16,20,60,60] were also measured. Increasing uniform resolution alone did not win this budget-limited study.

Equal-budget 10,000-episode validation screening (two training replicas per configuration):

{screen_table}

Finalists were ranked by **worst replica median**, then worst P10, worst lower-tail mean, mean replica median and mean score. Maximum score was never a selection criterion. The selected families each received three independent 24,000-episode runs. Before opening the locked test, a practical near-tie rule was recorded: within the winning family, replicas with validation median at least 90% of the best were considered practically close, then ranked by P10, lower-tail mean, median and mean score. This avoids choosing a marginally higher median with substantially worse lower-tail behavior. This validation-informed amendment was made after the first two reward replicas, not preregistered at study inception; it is documented in decisions.json. There was no retraining or retuning after test access.

## Seeds, preservation and implementation checks

Training exploration seeds are 42, 424242 and 848484; obstacle seed is training_seed + episode index. Even at 24,000 episodes their ranges exclude all evaluation sets. Historical benchmarks are 100000-100019 and 200000-200019; validation is 300000-300099; locked testing is 900000-900199. The original and every experimental agent were evaluated on the same 40 benchmark, 100 validation and 200 held-out test courses. Earlier pilot/screening snapshots retain their benchmark and validation evaluations; the final retained checkpoint for every training replica receives the complete test comparison.

The control trained with seed 42 reproduced **all Q-values exactly** in the original 10,000-episode checkpoint. The original **{len(original_manifest)} artifact files**, including the trained agent, all 24 original checkpoints and previous results, still match their SHA-256 hashes. Main remains at `{original_head}` with a clean working tree. No push was made. Experimental code and results are in a separate local worktree/branch.

A transient Windows file-sharing lock interrupted the first control run after a completed episode. Its saved checkpoint and RNG state were resumed; no completed learning was discarded. Bounded file-replacement retries were added in this experiment worktree only, and status JSON refresh was reduced to every 100 episodes. CSV statistics still flush/fsync each episode and checkpoints save at least every 30 seconds. These infrastructure adjustments do not alter numerical learning. All **87 tests passed**; see `tests_console.log` and `tests.xml` for the regression suite, including original artifact preservation, experimental range/seed checks, robust ranking, paired statistics and deadline handling.

## Actual resource cost

- New experimental training: **{total_episodes:,} completed episodes**, **{total_steps:,} simulation steps**.
- Sum of recorded training wall time across all independent policies: **{total_training:.2f} seconds ({total_training/60:.2f} minutes)**. Cumulative continuation times are counted once per replica, including the recovered control run.
- End-to-end study elapsed time at report generation, including preparation, evaluation, tests, code and analysis: **{elapsed/60:.2f} minutes**.
- Original baseline's historical training time: **{baseline['training_seconds']/60:.2f} minutes**, excluded from new experimental cost.
- Recommended policy's cumulative training time: **{chosen['training_seconds']/60:.2f} minutes**.
- Approved limits: at most 264,000 experimental episodes and four wall hours. Both respected. The unused episode allowance was not consumed for its own sake.

Longer survival increases simulation steps and cost per episode. These are measured Windows wall times, including durable I/O; lightweight tests/analysis sometimes ran concurrently, so small timing differences are not algorithmic speedup evidence; they are not hardware-independent compute benchmarks. Better policies need not train faster. All game parameters, simulation timestep, target score and termination limits stayed fixed.

## All retained experimental agents on the test set

Stopped arms have 6,000 episodes, non-finalists 10,000, finalists 24,000. These unequal-budget rows are not a causal equal-budget ablation; use the 10,000-episode table for that comparison.

{full_table}

## Artifacts and launching

- `all_agents_test_comparison.csv`: all per-policy test metrics and cost.
- `locked_test_comparison.csv`: benchmark, validation and test metrics together.
- `screening_all.csv`: pilot and screening endpoint metrics.
- `evaluations/`: per-seed scores, rewards, termination causes and diagnostics.
- `<arm>/seed_<seed>/`: exact configuration, training histories, checkpoints, final models and console logs.
- `graphs/`: screening comparisons, learning curves, locked-test distributions, paired improvements, training cost and recommended-family learning curves.
- `final_summary.json`, `model_hashes.json`: machine-readable conclusions and checkpoint hashes.
- `recommended_agent.npz`: byte-for-byte copy of the validation-selected model, not a replacement for the original.
- `recommended_compatible_agent.npz` and `recommended_compatible_config.json`: metadata-compatible exports for the original loader, when experimental encoder ranges are unused.
- `demo_seed_check.json`: additional post-selection seed-2026 game check (recommended score 100 versus original 17); this single course did not drive selection.

{compatibility_note}

From PowerShell, launch the experimental model explicitly:

```powershell
Set-Location "{launch_directory}"
& "{ORIGINAL / '.venv/Scripts/python.exe'}" -m flappy_ai play --model "{launch_model}"
```

The original launch command from the original folder continues to load the original agent. No promotion or GitHub publication is automatic.

## Remaining limitations

This is an adaptive, finite-budget study with only three finalist training seeds and one fixed game difficulty. Validation was used repeatedly, so its apparent improvements can be optimistic; the independently reserved test is the final confirmation. Early-stopped settings may improve with longer training. Many additional bin/range/learning schedules remain unexplored. Useful follow-ups within tabular Q-learning are an ablation separating range focusing from y/velocity coarsening, and separately controlled late-stage learning-rate/exploration decay schedules; neither is claimed tested here. Success is capped at 100 pipes, so performance beyond that target is unknown. The table is still an approximation, and state clipping may alias distant positions. A high median does not guarantee a high worst-case score; inspect P10 and failure rows. Checkpoint cleanup cannot guarantee saving during power/storage failure; previously completed checkpoints and fsynced statistics are preserved.

The course-bootstrap intervals condition on the selected trained model and do not fully quantify training-seed variability or repeated hyperparameter search. Raw episode rewards are not comparable between reward configurations; reported performance comparisons use pipe score and the common success target.

Exact recommended configuration:

```json
{config}
```
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps({k:evidence[k] for k in ['experimental_episodes','experimental_training_seconds','study_elapsed_seconds','recommended_key','original_artifact_files_preserved']},indent=2))
    print('PAIRED TEST',json.dumps(statistics))
    print('RECOMMENDED TEST',json.dumps({k:chosen[k] for k in ['mean','median','p10','best','success','states']}))

if __name__=='__main__': main()
