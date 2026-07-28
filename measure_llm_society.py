"""Measure an LLM society's conventions -> a reconstruction seed.

The LLM-substrate half of coevolutionary reconstruction (the CMA-ES
half is scripts/measure_proportions.py in the simulation repo). What is
measured and seeded is the society's CONVENTIONS — the role geography
and revealed practice — never its capabilities or private knowledge:

  positions       each agent's settled sign/seeking vectors, weighted
                  toward the periods in which it actually earned (who
                  stands where in meaning space = the role structure)
  want_practice   per want: settled-delivery approach-signature shares
                  (the society's revealed practice per demand)
  hiring          agent-to-agent settles per period (carried as data,
                  reported, and NOT injected into the seed prompt —
                  mirroring the CMA-ES protocol where combinations are
                  carried as data only and never bias generation)

Usage:
  python measure_llm_society.py runs/<name>/log.jsonl out_seed.json [burn_in]
"""
import collections
import json
import sys


def main(logpath, out, burn_in=0):
    sign_by_agent = collections.defaultdict(list)   # aid -> [(period, sign, seeking)]
    earned = collections.Counter()                  # aid -> settles won
    want_sigs = collections.defaultdict(collections.Counter)
    hires = 0
    periods = 0
    for line in open(logpath):
        r = json.loads(line)
        periods = max(periods, r.get('period', 0))
        if r['kind'] == 'turn' and r['period'] >= burn_in:
            sign_by_agent[r['agent']].append(
                (r['period'], r.get('sign'), None))
        elif r['kind'] == 'sale' and r['period'] >= burn_in:
            earned[r['seller']] += 1
        elif r['kind'] == 'delivery_test' and r['period'] >= burn_in \
                and r.get('score') is not None and r.get('sig'):
            want_sigs[r['want']][r['sig']] += 1
        elif r['kind'] == 'agent_trade' and r['period'] >= burn_in:
            hires += 1

    positions = []
    for aid, rows in sorted(sign_by_agent.items()):
        # the agent's settled position: its latest sign (roles at the
        # society's end state), tagged with how much it earned there
        rows.sort()
        last = rows[-1]
        positions.append(dict(agent_hint=aid, sign=last[1],
                              settles=earned.get(aid, 0)))

    practice = {}
    for want, ctr in want_sigs.items():
        tot = sum(ctr.values())
        practice[want] = {k: round(v / tot, 3) for k, v in ctr.most_common(6)}

    seed = dict(
        provenance=dict(
            description=f'Conventions measured from {logpath} '
                        f'(burn-in {burn_in}); coevolutionary '
                        f'reconstruction seed: role geography + revealed '
                        f'practice; capabilities and private knowledge '
                        f'NOT carried.',
            periods=periods,
            hires=hires),
        positions=positions,
        want_practice=practice,
    )
    json.dump(seed, open(out, 'w'), indent=1)
    print(f'{len(positions)} positions, {len(practice)} wants with practice, '
          f'{hires} hires over {periods} periods -> {out}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 0)
