"""Run model-written code in a subprocess with time and memory caps.

The caps are the platform's hard rails (the lesson of the July compute
bombs): no product evaluation may exceed its time budget or memory cap,
whatever the code inside does.
"""
import os
import resource
import subprocess
import sys
import tempfile


def run_code(code, timeout=180, mem_gb=6, workdir=None, pythonpath=None):
    with tempfile.NamedTemporaryFile('w', suffix='.py', dir=workdir,
                                     delete=False) as f:
        f.write(code)
        path = f.name

    def limits():
        resource.setrlimit(resource.RLIMIT_DATA,
                           (mem_gb << 30, mem_gb << 30))

    env = dict(os.environ)
    if pythonpath:
        env['PYTHONPATH'] = pythonpath

    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout, preexec_fn=limits,
                           cwd=workdir, env=env)
        return p.returncode, p.stdout[-4000:], p.stderr[-4000:]
    except subprocess.TimeoutExpired:
        return -9, '', f'TIMEOUT after {timeout}s'
    finally:
        os.unlink(path)
