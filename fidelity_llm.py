"""Reconstruction fidelity for LLM societies: seeded (e7) and unseeded
control (e7c) against the source (e6), at a matched horizon.

Measures, all computed from logs through --horizon periods (default:
the largest period both comparison runs have completed):
  capture structure   per want: which positions sell, and the
                      concentration (top seller share); does the
                      specialist structure regrow?
  positional          for each want, cosine between the seeded run's
  correspondence      top seller position and the source run's top
                      seller position for that want (the role geography
                      regrown), vs the control's.
  practice            per-want approach-signature distributions (where
                      logged), total-variation distance to source.
  floor               the source society's own split-half distance on
                      capture structure.

Usage:
  python fidelity_llm.py [--horizon N]


Paper mapping: the pilot panel table of Section 3 (source vs seeded vs unseeded control).
"""
import argparse
import collections
import json
import math


def load(name, horizon):
    sales = []
    signs = collections.defaultdict(dict)
    sigs = collections.defaultdict(collections.Counter)
    maxp = 0
    for line in open(f'runs/{name}/log.jsonl'):
        r = json.loads(line)
        p = r.get('period', 0)
        maxp = max(maxp, p)
        if horizon and p > horizon:
            continue
        if r['kind'] == 'sale':
            sales.append(r)
        elif r['kind'] == 'turn':
            signs[r['agent']][p] = r.get('sign')
        elif r['kind'] == 'delivery_test' and r.get('sig') and r.get('score') is not None:
            sigs[r['want']][r['sig']] += 1
    return dict(sales=sales, signs=signs, sigs=sigs, maxp=maxp)


def cos(a, b):
    if not a or not b:
        return 0.0
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb) if na and nb else 0.0


def capture(d):
    """want -> (top_seller, top_share, n_sales, top_seller_last_sign)"""
    out = {}
    by = collections.defaultdict(collections.Counter)
    for s in d['sales']:
        by[s['want']][s['seller']] += 1
    for w, ctr in by.items():
        top, n = ctr.most_common(1)[0]
        tot = sum(ctr.values())
        sign_hist = d['signs'].get(top, {})
        last = sign_hist[max(sign_hist)] if sign_hist else None
        out[w] = dict(top=top, share=round(n / tot, 3), n=tot,
                      distinct=len(ctr), top_sign=last)
    return out


def tv(c1, c2):
    n1, n2 = sum(c1.values()), sum(c2.values())
    if not n1 or not n2:
        return None
    ks = set(c1) | set(c2)
    return 0.5 * sum(abs(c1[k] / n1 - c2[k] / n2) for k in ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--horizon', type=int, default=0)
    a = ap.parse_args()

    src_full = load('e6_github_market', 0)
    e7 = load('e7_reconstruction', a.horizon)
    e7c = load('e7c_unseeded_null', a.horizon)
    hz = a.horizon or min(e7['maxp'], e7c['maxp'])
    if not a.horizon:
        e7 = load('e7_reconstruction', hz)
        e7c = load('e7c_unseeded_null', hz)
    src = load('e6_github_market', hz)

    print(f'=== matched horizon: {hz} periods '
          f'(e7 at {e7["maxp"]}, control at {e7c["maxp"]}, source full=25)')
    cs, c7, cc = capture(src), capture(e7), capture(e7c)
    print(f'{"want":22s} {"src top(share)":>16s} {"seeded top(share)":>18s} '
          f'{"null top(share)":>16s} {"pos-cos seeded":>14s} {"null":>6s}')
    for w in sorted(cs):
        s, s7, sc = cs.get(w), c7.get(w), cc.get(w)
        pc7 = cos(s['top_sign'], s7['top_sign']) if s and s7 else None
        pcc = cos(s['top_sign'], sc['top_sign']) if s and sc else None
        f7 = '-' if pc7 is None else f'{pc7:.2f}'
        fc = '-' if pcc is None else f'{pcc:.2f}'
        print(f'{w:22s} '
              f'{(s["top"]+" ("+str(s["share"])+")") if s else "-":>16s} '
              f'{(s7["top"]+" ("+str(s7["share"])+")") if s7 else "-":>18s} '
              f'{(sc["top"]+" ("+str(sc["share"])+")") if sc else "-":>16s} '
              f'{f7:>14} {fc:>6}')
    # split-half floor on the source: capture concentration first vs second half
    first = load('e6_github_market', 12)
    cf = capture(first)
    agree = sum(1 for w in cs if w in cf and cs[w]['top'] == cf[w]['top'])
    print(f'\nsource split-half floor: top-seller agreement '
          f'{agree}/{len(cs)} wants (first 12 periods vs full run)')
    # positional seeding check: how far each e7 agent started from an e6 position
    # practice TV where signatures exist (e7/e7c logs carry sig; source does not)
    for w in sorted(set(e7['sigs']) & set(e7c['sigs'])):
        print(f'practice sig TV seeded-vs-null [{w}]: '
              f'{round(tv(e7["sigs"][w], e7c["sigs"][w]), 3)}')
    print('\nNOTE: source (e6) predates signature logging; practice fidelity '
          'to source uses capture structure and positions; signature TV is '
          'reported seeded-vs-null.')


if __name__ == '__main__':
    main()
