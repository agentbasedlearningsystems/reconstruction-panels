"""Newborn adoption test for the birth societies (WORKSHEET_locality.md, P3).

Implements the test declared before any birth-society data existed:
six newborns per birth society enter at periods 10,12,14,16,18,20 by
replacing an incumbent (fresh agent, fresh sign, no instruction). Per
newborn, the read is one binary: is the newborn's post-birth practiced
behavior nearer its HOST society's incumbent practice than the OTHER
birth society's incumbent practice? Under the permuted-source null each
newborn is a fair coin (factor ~1/2): floor 1/64 per society; the
declared combination is the POOLED EXACT BINOMIAL over all twelve
newborns (worksheet arithmetic: 1/4096 at perfect adoption; 79/4096
~ 0.02 at five-of-six per society), powered against one defector per
society. (An earlier draft of this script used Fisher's chi-square
method; corrected to the declared pooled binomial BEFORE any adoption
binary was read.)

Practice profile (the panel instrument, applied per agent): counts over
paid human deliveries by want id (kind=sale, seller) plus component
sales by item type (kind=agent_trade, seller). Sensitivity variant adds
audition attempts (kind=delivery_test). Similarity is total-variation
similarity, 1 - TV/  ... = 1 - 0.5*sum|p-q|. A newborn with no paid
events falls back to its attempt profile and the fallback is disclosed
in the output; a newborn with no events at all counts as non-adopting
(conservative) and is disclosed.

Segment handling: run logs concatenate resumed attempts (replayed
periods re-logged). Only the FINAL contiguous segment is read: any
drop in the period sequence resets collection (the same deepest-segment
rule as instruments_battery.py).

The combined statistic only computes when BOTH societies carry their
full six births at target depth; run with --host-only to validate the
pipeline on one society without reading any adoption binary.

Output: papers/transmission_adoption.json and a printed table.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BIRTH_PERIODS = [10, 12, 14, 16, 18, 20]


def final_segment(path):
    """Return the log's DEEPEST contiguous attempt as a list of events.

    Logs concatenate resumed attempts (each replays from period 1 via the
    response cache, byte-identically, then extends). The deepest attempt
    is therefore the authoritative record; a later, shallower attempt is
    a prefix replay still in flight (instruments_battery.py rule)."""
    segs = [[]]
    last_p = -1
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        p = e.get('period')
        if p is None:
            continue
        if p < last_p:          # period dropped: a resumed attempt begins
            segs.append([])
        last_p = p
        segs[-1].append(e)
    return max(segs, key=lambda s: max((e.get('period', 0) for e in s), default=0))


def births_in(seg):
    """agent id -> birth period, final segment only."""
    out = {}
    for e in seg:
        if e.get('kind') == 'birth':
            out[e['agent']] = e['period']
    return out


def practice(seg, aid, after_period, paid_only=True):
    """Practice profile of one agent strictly after a period."""
    c = Counter()
    for e in seg:
        if e.get('period', 0) <= after_period:
            continue
        k = e.get('kind')
        if k == 'sale' and e.get('seller') == aid:
            c['want:' + e['want']] += 1
        elif k == 'agent_trade' and e.get('seller') == aid:
            c['item:' + e['item']] += 1
        elif not paid_only and k == 'delivery_test' and e.get('seller') == aid:
            c['try:' + e['want']] += 1
    return c


def aggregate(seg, aids, after_period, paid_only=True):
    c = Counter()
    for a in aids:
        c.update(practice(seg, a, after_period, paid_only))
    return c


def tv_similarity(c1, c2):
    if not c1 or not c2:
        return None
    n1, n2 = sum(c1.values()), sum(c2.values())
    keys = set(c1) | set(c2)
    tv = 0.5 * sum(abs(c1[k] / n1 - c2[k] / n2) for k in keys)
    return 1.0 - tv


def binom_tail(k, n, p=0.5):
    """One-sided P(X >= k), exact."""
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i)
               for i in range(k, n + 1))


def pooled_binom(k_total, n_total):
    """The declared combination: exact binomial tail over all newborns."""
    return binom_tail(k_total, n_total)


def society(run):
    path = os.path.join(ROOT, 'runs', run, 'log.jsonl')
    seg = final_segment(path)
    born = births_in(seg)
    max_p = max(e.get('period', 0) for e in seg)
    return {'run': run, 'seg': seg, 'born': born, 'depth': max_p}


def newborn_reads(host, alt):
    """Per-newborn adoption binaries for the host society's newborns."""
    newborn_ids = set(host['born'])
    incumbents_h = sorted({e.get('agent') or e.get('seller')
                           for e in host['seg']
                           if e.get('kind') == 'turn'} - newborn_ids - {None})
    newborn_ids_alt = set(alt['born'])
    incumbents_a = sorted({e.get('agent') or e.get('seller')
                           for e in alt['seg']
                           if e.get('kind') == 'turn'} - newborn_ids_alt - {None})
    window = min(BIRTH_PERIODS) - 1        # incumbent era shared by all newborns
    ref_h = aggregate(host['seg'], incumbents_h, window)
    ref_a = aggregate(alt['seg'], incumbents_a, window)
    rows = []
    for aid, bp in sorted(host['born'].items(), key=lambda t: t[1]):
        prof = practice(host['seg'], aid, bp)
        fallback = False
        if not prof:
            prof = practice(host['seg'], aid, bp, paid_only=False)
            fallback = True
        sim_h = tv_similarity(prof, ref_h)
        sim_a = tv_similarity(prof, ref_a)
        if sim_h is None or sim_a is None:
            adopted = False
            note = 'no events at all: counted non-adopting (conservative)'
        else:
            adopted = sim_h > sim_a
            note = 'attempt-profile fallback' if fallback else ''
        rows.append({'agent': aid, 'birth_period': bp,
                     'events': sum(prof.values()),
                     'sim_host': None if sim_h is None else round(sim_h, 4),
                     'sim_alt': None if sim_a is None else round(sim_a, 4),
                     'adopted': adopted, 'note': note})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host-only', metavar='RUN',
                    help='pipeline validation on one society: profiles and '
                         'host similarity only; no adoption binary is read')
    ap.add_argument('--pair', nargs=2, metavar=('RUN_A', 'RUN_B'),
                    default=['loc_birth_a_v3', 'loc_birth_b_v3'])
    args = ap.parse_args()

    if args.host_only:
        h = society(args.host_only)
        print('%s depth=%d births=%s' % (h['run'], h['depth'],
                                         sorted(h['born'].items(), key=lambda t: t[1])))
        newborn_ids = set(h['born'])
        incumbents = sorted({e.get('agent') for e in h['seg']
                             if e.get('kind') == 'turn'} - newborn_ids - {None})
        ref = aggregate(h['seg'], incumbents, min(BIRTH_PERIODS) - 1)
        print('incumbent reference classes=%d mass=%d' % (len(ref), sum(ref.values())))
        for aid, bp in sorted(h['born'].items(), key=lambda t: t[1]):
            prof = practice(h['seg'], aid, bp)
            kind = 'paid'
            if not prof:
                prof = practice(h['seg'], aid, bp, paid_only=False)
                kind = 'attempts'
            sim = tv_similarity(prof, ref)
            print('  %s born p%-2d  %s events=%-3d sim_host=%s'
                  % (aid, bp, kind, sum(prof.values()),
                     'NA' if sim is None else '%.3f' % sim))
        print('\npipeline valid; adoption binaries not read (declared test '
              'computes only with both societies at depth).')
        return

    a = society(args.pair[0])
    b = society(args.pair[1])
    for s in (a, b):
        if len(s['born']) < 6 or s['depth'] < 30:
            sys.exit('%s not ready: births=%d depth=%d (need 6 and 30)'
                     % (s['run'], len(s['born']), s['depth']))
    rows_a = newborn_reads(a, b)
    rows_b = newborn_reads(b, a)
    k_a = sum(r['adopted'] for r in rows_a)
    k_b = sum(r['adopted'] for r in rows_b)
    p_a = binom_tail(k_a, len(rows_a))
    p_b = binom_tail(k_b, len(rows_b))
    p_comb = pooled_binom(k_a + k_b, len(rows_a) + len(rows_b))
    out = {'declared': 'WORKSHEET_locality.md P3; script written before any '
                       'adoption binary was read; society A component read '
                       'first at its declared depth, disclosed',
           'society_a': {'run': a['run'], 'depth': a['depth'], 'rows': rows_a,
                         'adopted': '%d/6' % k_a, 'p_one_sided': p_a},
           'society_b': {'run': b['run'], 'depth': b['depth'], 'rows': rows_b,
                         'adopted': '%d/6' % k_b, 'p_one_sided': p_b},
           'pooled_binomial_combined_p': p_comb}
    dest = os.path.join(ROOT, 'papers', 'transmission_adoption.json')
    json.dump(out, open(dest, 'w'), indent=1)
    for name, rows, k, p in (('A', rows_a, k_a, p_a), ('B', rows_b, k_b, p_b)):
        print('society %s: adopted %d/6, one-sided p=%.4g' % (name, k, p))
        for r in rows:
            print('   %(agent)s p%(birth_period)d events=%(events)d '
                  'host=%(sim_host)s alt=%(sim_alt)s adopted=%(adopted)s '
                  '%(note)s' % r)
    print('pooled exact binomial combined p = %.4g' % p_comb)
    print('written:', dest)


if __name__ == '__main__':
    main()
