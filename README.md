# Coevolutionary Reconstruction: Panels, Seeding, and the Decay Diagnostic

Anonymous reproduction package for the paper. This file explains what
the method is, what every file does, and how to regenerate each
result, assuming no prior knowledge of the project.

## What this is, in plain terms

The experiments simulate small societies of language-model agents who
earn a living by building and selling machine-learning software to a
simulated buyer. Nobody assigns roles; agents choose what to make,
display an eight-number sign, and trade under simple market rules.

Reconstruction is a method for making such a society behave in the
measured proportions of a SOURCE population, with a built-in warning
when it fails:

1. SEEDING. We measure how often each behavior occurs in the source
   society and write those frequencies into a fresh society's
   starting dispositions. Nothing else is copied: no personalities,
   no scripts, no vocabulary.
2. GROWTH UNDER LAW. The seeded society then lives under the market
   rules. A seeded disposition survives only if the agent can earn a
   living practicing it.
3. THE DECAY DIAGNOSTIC. After growth, we compare practiced behavior
   against the source. Where they match, the incentive model
   explains the behavior. Where the seeded behavior decayed back
   toward what an unseeded society does, the simulation itself has
   flagged that behavior as unexplained. This gap is the method's
   self-audit.

## Files

| file | what it is |
|---|---|
| llm_backend.py | The connection to the language model, with response caching so reruns are cheap and deterministic. Requires the ANTHROPIC_API_KEY environment variable; no key ships in this repository. |
| sandbox_run.py | Runs agent-written code in a separate process with limits, and reports scores from held-out tests. |
| run_experiments.py | Launches a society from a configuration file: the source society, a seeded daughter, or an unseeded control. |
| e6_seed.json | The seed written from the source society used in the paper's pilot table: behavior frequencies only. |
| seedA_school2a.json, seedB_school2c.json | Seeds from two different source societies, used for the two-source panel (each daughter should track its OWN source). |
| fidelity_llm.py | The measurement program for the paper's pilot table. It aligns a source, a seeded daughter, and an unseeded control at the same age and reports, per product: who the top seller is, that seller's share, and the position-similarity between daughter and source. |
| measure_llm_society.py | General society measurements: who sold what, when, at what price; used by the declared tests. |
| analysis_instruments.py | The declared statistical tests, fixed before the data existed. |
| requirements.txt | Python dependencies. |

## Reproduce, step by step

1. `python3 -m venv venv && venv/bin/pip install -r requirements.txt`
2. `export ANTHROPIC_API_KEY=...` (your own key; each full society
   costs several dollars in model calls; cached reruns cost cents).
3. Run an unseeded control society:
   `venv/bin/python run_experiments.py <control_config>`
4. Run a seeded daughter of a source:
   `venv/bin/python run_experiments.py <seeded_config>`
5. Produce the paper's pilot table (source vs seeded vs control):
   `venv/bin/python fidelity_llm.py`
   The columns mean: which agent position leads each product, how
   concentrated that product's sales are, and whether the daughter's
   leadership PATTERN matches the source (position-cosine near 1)
   while the control drifts (near 0 or negative).

A per-figure file, FIGURES.md, maps every number in the paper to one
command; it ships with the supplementary deadline.

## What to expect

A 25-period society takes several hours on an ordinary machine, most
of it waiting on model calls; the response cache makes any repeat of
the same society near-free and byte-identical. Corpus data carries
its own license notes beside each corpus directory.

## Free confirmation, priced replication

Agent turns are billed model calls, so we state checking economics
plainly.

1. CONFIRMATION (free, no API key). `GCON_OFFLINE=1` runs any society
   replay entirely from the shipped transcript packs (with the
   supplementary material): builds retrain locally, deliveries
   re-score on held-out data, the market re-clears, and the run
   matches the published logs. A prompt outside the pack raises a
   clear error instead of placing a model call, so you cannot be
   billed. This confirms that every reported number follows from the
   recorded interactions; it re-samples nothing.
2. ANALYSIS (free). Every figure and table regenerates from shipped
   logs by a listed command.
3. REPLICATION (priced): fresh societies with your own key cost, at
   list prices, about $3 (single mind), $5 (first-world market
   society), up to about $15 (thirty-day locality society). The
   evolutionary substrate has no model bill at all.

Pack building for authors: `python3 instruments/make_cache_pack.py
--out packs/NAME.tgz -- <replay command>`; reviewers unpack into
`transcripts/` and run the same command with `GCON_OFFLINE=1`.
