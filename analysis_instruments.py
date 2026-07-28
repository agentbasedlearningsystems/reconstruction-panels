"""The standing equity instruments, computed on every society whose log
carries both incomes and qualities.

Definitions are the designer's, from Kolonin, the designer, Goertzel, Pennachin, Ikle,
Znidar and Argentieri, "A Reputation System for Market Security and
Equity" (IJCAI-19 AI4SG workshop). Cite that paper in any caption.

  wealth            Va = (Vxa + Vax) / 2, market volume RECEIVED by agent
                    a and market volume SPENT by agent a, averaged. Here:
                    received = what the human buyer paid it for accepted
                    deliveries plus what other agents paid it for
                    components; spent = what it paid other agents.
  goodness          the paper's "talent": the quality of the agent's own
                    goods. Here the mean verified score over every
                    delivery of that agent that a tester scored, sold or
                    not (best-score variant reported alongside).
  talent-adjusted   the Gini coefficient of the equitable share
  Gini              Wa = Va / goodness_a; wealth that strays from the
                    agent's contribution in quality. High = trade is
                    unrelated to talent.
  security-Pearson  the correlation between income and goodness weighted
                    toward LOW earners: are the market's low ranks the
                    ones that deserve to be low?
  equity-Pearson    the same correlation weighted toward HIGH earners: is
                    the top of the market the part that earned it?
  utility           the mean accepted score; how satisfied the buyer is
                    with what it actually bought.

One definitional choice is mine and is flagged for the designer: the paper
says "weighted towards the lowness / highness of the reputation scores"
without fixing the weight shape. This uses w = 1 - x for security and
w = x for equity, x being income min-max normalized to [0, 1] across the
society, and reports the rank-normalized variant alongside so the number
does not hang on that choice.

Usage:
  python analysis_instruments.py [run ...]      (default: the LLM cohort)


Paper mapping: the declared tests of Sections 3 and 4, fixed before the data existed.
"""
import collections
import glob
import json
import statistics
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
CITATION = ('Kolonin, the designer, Goertzel, Pennachin, Ikle, Znidar and '
            'Argentieri, "A Reputation System for Market Security and '
            'Equity", IJCAI-19 AI4SG workshop')


def _r(v, n=4):
    return round(v, n) if isinstance(v, float) else v


def gini(xs):
    """Gini over a non-negative series, i the sorted index, N the count."""
    xs = sorted(x for x in xs if x is not None)
    n = len(xs)
    tot = sum(xs)
    if n < 2 or tot <= 0:
        return None
    return sum((2 * (i + 1) - n - 1) * x for i, x in enumerate(xs)) / (n * tot)


def wpearson(xs, ys, ws):
    """Pearson correlation with per-observation weights."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 3:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    cov = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    vx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    vy = sum(w * (y - my) ** 2 for w, y in zip(ws, ys))
    if vx <= 0 or vy <= 0:
        return None
    return cov / (vx * vy) ** 0.5


def norm01(vals):
    lo, hi = min(vals), max(vals)
    return [0.5] * len(vals) if hi == lo else [(v - lo) / (hi - lo)
                                               for v in vals]


def rank01(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    for r, i in enumerate(order):
        out[i] = r / (len(vals) - 1) if len(vals) > 1 else 0.5
    return out


def society(run):
    path = os.path.join(RUNS, run, 'log.jsonl')
    if not os.path.exists(path):
        return None
    received = collections.Counter()   # Vxa
    spent = collections.Counter()      # Vax
    scores = collections.defaultdict(list)   # agent -> every scored delivery
    accepted = []                      # scores of deliveries actually bought
    agents = set()
    periods = 0
    for line in open(path):
        r = json.loads(line)
        k = r.get('kind')
        periods = max(periods, r.get('period', 0) or 0)
        if k == 'turn':
            agents.add(r['agent'])
        elif k == 'sale':
            received[r['seller']] += r.get('paid') or 0.0
            if isinstance(r.get('score'), (int, float)):
                accepted.append(r['score'])
                scores[r['seller']].append(r['score'])
        elif k == 'agent_trade':
            received[r['seller']] += r.get('paid') or 0.0
            spent[r['buyer']] += r.get('paid') or 0.0
        elif k == 'delivery_test' and isinstance(r.get('score'), (int, float)):
            scores[r['seller']].append(r['score'])
    if not agents or not accepted:
        return None
    rows = []
    for a in sorted(agents):
        sc = scores.get(a, [])
        rows.append(dict(
            agent=a, received=round(received[a], 2), spent=round(spent[a], 2),
            wealth=round((received[a] + spent[a]) / 2, 2),
            net_income=round(received[a] - spent[a], 2),
            goodness=(round(sum(sc) / len(sc), 4) if sc else None),
            best_score=(round(max(sc), 4) if sc else None),
            deliveries_scored=len(sc)))

    talented = [r for r in rows if r['goodness']]
    share = [r['wealth'] / r['goodness'] for r in talented]
    share_best = [r['wealth'] / r['best_score'] for r in talented]

    # the correlation set: agents the market could rank at all
    corr = [r for r in rows if r['goodness'] is not None]
    out = dict(run=run, periods=periods, agents=len(agents),
               agents_with_scored_deliveries=len(corr),
               utility=round(sum(accepted) / len(accepted), 4),
               accepted_deliveries=len(accepted),
               talent_adjusted_gini=_r(gini(share)),
               talent_adjusted_gini_on_best_score=_r(gini(share_best)),
               wealth_gini=_r(gini([r['wealth'] for r in rows])),
               agents_excluded_from_gini=len(rows) - len(talented),
               per_agent=rows)
    if len(corr) >= 3:
        inc = [r['net_income'] for r in corr]
        good = [r['goodness'] for r in corr]
        for tag, weights in (('minmax', norm01(inc)), ('rank', rank01(inc))):
            sec = wpearson(inc, good, [1 - w for w in weights])
            eq = wpearson(inc, good, list(weights))
            out[f'security_pearson_{tag}'] = (round(sec, 4) if sec is not None
                                              else None)
            out[f'equity_pearson_{tag}'] = (round(eq, 4) if eq is not None
                                            else None)
        # the same activity control the evolutionary side needs: an agent
        # that delivered more may earn more AND be scored more often
        n = [float(r['deliveries_scored']) for r in corr]
        r_ig = wpearson(inc, good, [1.0] * len(inc))
        r_in = wpearson(inc, n, [1.0] * len(inc))
        r_gn = wpearson(good, n, [1.0] * len(inc))
        if None not in (r_ig, r_in, r_gn):
            den = ((1 - r_in ** 2) * (1 - r_gn ** 2)) ** 0.5
            out['pearson_partialling_out_activity'] = (
                _r((r_ig - r_in * r_gn) / den) if den > 1e-12 else None)
            out['income_vs_activity'] = _r(r_in)
            out['quality_vs_activity'] = _r(r_gn)
        plain = wpearson(inc, good, [1.0] * len(inc))
        out['pearson_unweighted'] = round(plain, 4) if plain is not None \
            else None
        gross = wpearson([r['received'] for r in corr], good,
                         [1.0] * len(corr))
        out['pearson_unweighted_gross_received'] = (
            round(gross, 4) if gross is not None else None)
    return out


DEFAULT = ['cw_school3_a', 'cw_school3_b', 'cw_school3_c', 'cw_school3_c2',
           'cw_school2_a', 'cw_school2_b', 'cw_school2_c', 'cw_school2_d',
           'cw_school2_e', 'cw_bazaar01', 'cw_bazaar02',
           'cw_cr2_A1', 'cw_cr2_A2', 'cw_cr2_A3', 'cw_cr2_A4',
           'cw_cr2_B1', 'cw_cr2_B2', 'cw_cr2_B3', 'cw_cr2_B4',
           'cw_cr2_N1', 'cw_cr2_N2']


def main(runs):
    runs = runs or (DEFAULT + sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(os.path.join(RUNS, 'loc_*', 'log.jsonl'))))
    got, live, skipped = {}, {}, []
    for r in runs:
        s = society(r)
        if not s:
            skipped.append(r)
        elif r.startswith('loc_'):
            # the locality cohort is still running; these are snapshots of
            # societies a few periods old, not results
            s['status'] = 'IN FLIGHT - partial, do not read yet'
            live[r] = s
        else:
            got[r] = s
    out = dict(
        what='the standing equity instruments on every language-model '
             'society whose log carries both incomes and qualities',
        citation=CITATION,
        definitions=dict(
            wealth='(market volume received + market volume spent) / 2',
            goodness='mean verified score over every scored delivery of '
                     'that agent, sold or not',
            talent_adjusted_gini='Gini over wealth / goodness',
            security_pearson='Pearson(net income, goodness) weighted '
                             'toward low earners (w = 1 - normalized '
                             'income)',
            equity_pearson='the same weighted toward high earners '
                           '(w = normalized income)',
            utility='mean score over deliveries the buyer accepted',
            weight_shape_flag='the source paper does not fix the weight '
                              'shape; minmax is reported as primary and '
                              'rank as the sensitivity - the designer\'s call'),
        societies=got, in_flight=live, skipped=skipped,
        cohort_medians={
            k: _r(statistics.median([v[k] for v in got.values()
                                     if v.get(k) is not None]))
            for k in ('utility', 'talent_adjusted_gini', 'wealth_gini',
                      'security_pearson_minmax', 'equity_pearson_minmax',
                      'security_pearson_rank', 'equity_pearson_rank',
                      'pearson_unweighted',
                      'pearson_partialling_out_activity',
                      'income_vs_activity', 'quality_vs_activity')},
        cohort_n=len(got))
    p = os.path.join(HERE, 'gcon_instruments_llm.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('wrote', p, f'({len(got)} societies, {len(skipped)} skipped)')
    return out


if __name__ == '__main__':
    main(sys.argv[1:])


def table(path=None):
    """The mechanical table, generated from the emitted JSON."""
    d = json.load(open(os.path.join(HERE, 'gcon_instruments_llm.json')))
    path = path or os.path.join(HERE, 'gcon_instruments_table.md')
    L = ['# Standing equity instruments, language-model societies\n',
         f'Instrument definitions from {d["citation"]}. Generated by '
         '`analysis_instruments.py` from `gcon_instruments_llm.json`; no '
         'number is hand-entered.\n',
         '| society | periods | agents | utility | talent-adjusted Gini | '
         'wealth Gini | security-Pearson | equity-Pearson | '
         'unweighted Pearson |',
         '|---|---|---|---|---|---|---|---|---|']
    for r, v in d['societies'].items():
        L.append(f"| {r} | {v['periods']} | {v['agents']} | "
                 f"{v['utility']} | {v['talent_adjusted_gini']} | "
                 f"{v['wealth_gini']} | {v.get('security_pearson_minmax')} | "
                 f"{v.get('equity_pearson_minmax')} | "
                 f"{v.get('pearson_unweighted')} |")
    m = d['cohort_medians']
    L.append(f"| **median of {d['cohort_n']}** | | | {m['utility']} | "
             f"{m['talent_adjusted_gini']} | {m['wealth_gini']} | "
             f"{m['security_pearson_minmax']} | "
             f"{m['equity_pearson_minmax']} | {m['pearson_unweighted']} |")
    L.append('\nWeights: security = 1 - min-max normalized net income, '
             'equity = min-max normalized net income. Rank-normalized '
             'weights give medians '
             f"{m['security_pearson_rank']} and {m['equity_pearson_rank']}; "
             'the source paper does not fix the weight shape.\n')
    if d['in_flight']:
        L.append('\nThe locality cohort is still running and is held out: '
                 + ', '.join(d['in_flight']) + '. Rerun this script when '
                 'COHORT_STATUS.md shows them finished.\n')
    open(path, 'w').write('\n'.join(L))
    print('wrote', path)
