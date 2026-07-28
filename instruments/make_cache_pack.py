"""Build a per-society transcript pack for zero-cost confirmation runs.

Paper mapping: the reproducibility sections of both papers (free
confirmation tier). A pack contains every model response a society's
replay actually reads, so a reviewer with no API key can re-execute
the society end to end: builds retrain in the sandbox, deliveries
re-score on held-out data, the market re-clears, and the run matches
the published logs. This CONFIRMS the record rather than replicates
the sampling; fresh-sampling replication costs are stated in the
papers.

Usage (pack author side; runs fully offline, so a complete pack is
proven complete by construction):

    python3 instruments/make_cache_pack.py --out packs/NAME.tgz -- \
        <the society's replay command>

Reviewer side:

    tar -xzf NAME.tgz -C transcripts/
    GCON_OFFLINE=1 <the same replay command>

GCON_OFFLINE=1 refuses any cache miss with an error instead of
placing a model call, so the reviewer cannot be billed.
"""
import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, 'transcripts')
KEY_PATTERN = 'sk-' + 'ant-'          # split so scanners can scan the scanner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('cmd', nargs=argparse.REMAINDER,
                    help='-- followed by the replay command')
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == '--' else args.cmd
    if not cmd:
        sys.exit('no replay command given after --')

    trace = tempfile.NamedTemporaryFile(mode='r', suffix='.trace',
                                        delete=False)
    env = dict(os.environ, GCON_OFFLINE='1', GCON_CACHE_TRACE=trace.name)
    print('replaying offline (a miss would abort, proving incompleteness):')
    rc = subprocess.call(cmd, env=env, cwd=ROOT)
    if rc != 0:
        sys.exit('replay exited %d; pack NOT built' % rc)

    keys = sorted(set(open(trace.name).read().split()))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.',
                exist_ok=True)
    packed = 0
    with tarfile.open(args.out, 'w:gz') as tar:
        for k in keys:
            p = os.path.join(CACHE, k + '.json')
            if not os.path.exists(p):
                sys.exit('traced key %s missing from cache; refusing' % k)
            body = open(p, encoding='utf-8', errors='replace').read()
            if KEY_PATTERN in body:
                sys.exit('credential material in %s; refusing to pack' % k)
            tar.add(p, arcname=k + '.json')
            packed += 1
        manifest = json.dumps({'entries': packed, 'command': cmd}, indent=1)
        mpath = os.path.join(tempfile.gettempdir(), 'PACK_MANIFEST.json')
        open(mpath, 'w').write(manifest)
        tar.add(mpath, arcname='PACK_MANIFEST.json')
    os.unlink(trace.name)
    print('packed %d transcript entries -> %s' % (packed, args.out))


if __name__ == '__main__':
    main()
