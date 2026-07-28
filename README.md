# Coevolutionary Reconstruction: Panels, Seeding, and the Decay Diagnostic
Anonymous reproduction package. Code and configurations for growing
seeded societies, running unseeded null controls, and computing the
declared panel and transmission tests.

## Reproduce
1. python -m venv venv && venv/bin/pip install -r requirements.txt
2. Set ANTHROPIC_API_KEY in the environment (never in files).
3. Seeded panel society: venv/bin/python run_experiments.py <panel_config>
4. Fidelity and proportion reads: venv/bin/python fidelity_llm.py <run_dir>
5. Declared tests: venv/bin/python measure_llm_society.py <run_dir>
Each figure in the paper maps to one command above; the per-figure
list is in FIGURES.md (added with the July 31 supplement).
