"""Natural-source threshold selection with leave-one-forest-out stress checks."""

from tree_learn.mask_refinement.evaluation import macro_metrics


def rank(record):
    return record['recovered'], record['macro']['f1'], record['macro']['coverage'], -record['accepted_pairs']


def fold_summary(trial, excluded):
    forests = [r for r in trial['forests'] if r['plot'] != excluded]
    baseline = macro_metrics([r['baseline'] for r in forests])
    macro = macro_metrics([r['refined'] for r in forests])
    lost = sum(r['lost'] for r in forests)
    return dict(macro=macro, lost=lost, recovered=sum(r['recovered'] for r in forests),
        accepted_pairs=sum(r['accepted_pairs'] for r in forests),
        eligible=lost == 0 and macro['f1'] >= baseline['f1'] - 1e-12)


def choose(trials):
    """A zero-edit fallback is explicit, not a successful recovery result.

    Fold checks stress source policy selection; they are not independent test data.
    External labels/scores never enter this function.
    """
    if not trials or trials[0]['accepted_pairs'] or not trials[0]['eligible']:
        raise ValueError('First trial must be the valid keep-all fallback.')
    plots = [r['plot'] for r in trials[0]['forests']]
    if len(set(plots)) != len(plots) or any([r['plot'] for r in t['forests']] != plots for t in trials):
        raise ValueError('Every threshold trial must contain the same ordered source forests.')
    folds = []
    # Tiny server fixtures with one/two forests cannot support this stress check.
    if len(plots) >= 3:
        for plot in plots:
            summaries = [fold_summary(trial, plot) for trial in trials]
            eligible = [i for i, record in enumerate(summaries) if record['eligible']]
            index = max(eligible, key=lambda i: rank(summaries[i]))
            held = next(r for r in trials[index]['forests'] if r['plot'] == plot)
            folds.append(dict(held_out_plot=plot, threshold=trials[index]['threshold'],
                lost=held['lost'], recovered=held['recovered'], accepted_pairs=held['accepted_pairs'],
                baseline_f1=held['baseline']['f1'], refined_f1=held['refined']['f1'],
                safe=held['lost'] == 0 and held['refined']['f1'] >= held['baseline']['f1'] - 1e-12))
    robust = all(fold['safe'] for fold in folds)
    eligible = [trial for trial in trials if trial['eligible'] and (robust or not trial['accepted_pairs'])]
    selected = max(eligible, key=rank)
    return selected, dict(performed=len(plots) >= 3, folds=folds, all_source_folds_safe=robust,
                          selected_keep_all=selected['accepted_pairs'] == 0)
