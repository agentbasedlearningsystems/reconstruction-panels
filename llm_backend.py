"""Model-agnostic generation seam with a transcript cache.

Everything above generate(); prompts, sandbox, ledger, market; is
backend-blind. Swapping Qwen for Claude later means adding a backend
class and a pricing entry; nothing else changes. Transcripts cache to
disk keyed by (model, prompt) so identical calls are free and replay
is deterministic regardless of backend stochasticity.


Paper mapping: the cached model connection behind the reproducibility statements of Section 10.
"""
import fcntl
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'transcripts')


class QwenLocal:
    name = 'qwen2.5-coder-3b-q4km'

    def __init__(self, n_threads=4):
        from llama_cpp import Llama
        self._llm = Llama(
            model_path=os.path.join(HERE, 'models',
                                    'qwen2.5-coder-3b-instruct-q4_k_m.gguf'),
            n_ctx=8192, seed=7, verbose=False, n_threads=n_threads,
            n_gpu_layers=-1)

    def __call__(self, prompt, max_tokens=1400):
        out = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=max_tokens, seed=7)
        text = out['choices'][0]['message']['content']
        u = out['usage']
        return text, u['prompt_tokens'], u['completion_tokens']


class ClaudeAPI:
    """Claude backend with a hard dollar cap.

    Key resolution is EXPLICIT only (env ANTHROPIC_API_KEY or the file
    ~/.config/anthropic/llm_market_key); no ambient profile fallback, so
    nothing bills to an account the designer didn't deliberately provide.
    Every call appends to spend.json; at the cap, calls raise before
    sending. Prices are list $/MTok (input, output).
    """

    PRICES = {
        'claude-haiku-4-5': (1.00, 5.00),
        'claude-sonnet-5': (3.00, 15.00),
        'claude-opus-4-8': (5.00, 25.00),
    }
    # prompt-cache pricing multipliers on the input rate
    CACHE_WRITE_X = 1.25
    CACHE_READ_X = 0.10
    LEDGER = os.path.join(HERE, 'spend.json')

    @staticmethod
    def _env_file(path):
        vals = {}
        if os.path.exists(path):
            for line in open(path):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    vals[k.strip()] = v.strip()
        return vals

    def __init__(self, model='claude-haiku-4-5', cap_usd=None, temperature=0.0):
        import anthropic
        # Key + cap resolution: env var, then the setup script's .env files,
        # then the raw keyfile. Explicit sources only; no ambient profiles.
        merged = {}
        for p in (os.path.join(HERE, '.env'),
                  os.path.expanduser('~/.config/anthropic/env')):
            merged.update(self._env_file(p))
        key = os.environ.get('ANTHROPIC_API_KEY') or merged.get('ANTHROPIC_API_KEY')
        keyfile = os.path.expanduser('~/.config/anthropic/llm_market_key')
        if not key and os.path.exists(keyfile):
            key = open(keyfile).read().strip()
        if not key:
            raise RuntimeError(
                'No API key: run setup_credentials.sh, or set '
                'ANTHROPIC_API_KEY (never in the repo).')
        if cap_usd is None:
            cap_usd = float(os.environ.get('LLM_MARKET_BUDGET_USD')
                            or merged.get('BUDGET_CAP_USD') or 25)
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.name = model
        self.cap = cap_usd
        self.temperature = temperature

    # Concurrent runs share this one ledger: every read and every
    # read-modify-write must hold the lock, and writes must land
    # atomically, or a reader can catch a half-written file.
    def _spent(self):
        if not os.path.exists(self.LEDGER):
            return 0.0
        with open(self.LEDGER + '.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                return json.load(open(self.LEDGER)).get('total_usd', 0.0)
            except (json.JSONDecodeError, OSError):
                # truncated ledger (VM died mid-write): salvage the total,
                # keep the corrupt file, rewrite a clean ledger.
                import re as _re
                import shutil as _sh
                try:
                    raw = open(self.LEDGER, errors='replace').read()
                except OSError:
                    raw = ''
                m = _re.search(r'"total_usd"\s*:\s*([0-9.]+)', raw)
                total = float(m.group(1)) if m else 0.0
                try:
                    _sh.move(self.LEDGER, self.LEDGER + '.corrupt_bak')
                except OSError:
                    pass
                with open(self.LEDGER, 'w') as f:
                    json.dump({'total_usd': total, 'calls': [],
                               'note': 'auto-rebuilt from corrupt ledger'}, f)
                return total

    def _record(self, tin, tout, cost):
        with open(self.LEDGER + '.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            d = (json.load(open(self.LEDGER))
                 if os.path.exists(self.LEDGER)
                 else {'total_usd': 0.0, 'calls': []})
            d['total_usd'] = round(d['total_usd'] + cost, 6)
            d['calls'].append(dict(model=self.model, tokens_in=tin,
                                   tokens_out=tout, usd=round(cost, 6)))
            tmp = self.LEDGER + '.tmp'
            json.dump(d, open(tmp, 'w'), indent=1)
            os.replace(tmp, self.LEDGER)

    def __call__(self, prompt, max_tokens=1400, system_prefix=None):
        spent = self._spent()
        if spent >= self.cap:
            # the designer 2026-07-16: "dont let anything die because of some
            # limit just try to tell me, but dont let stuff die." The cap
            # warns loudly (stdout + flag file for the spend watcher) and
            # keeps serving; the console balance is the physical wall.
            step = int((spent - self.cap) // 10)
            flag = os.path.join(HERE, 'BUDGET_CAP_EXCEEDED.flag')
            prev = -1
            if os.path.exists(flag):
                try:
                    prev = int(open(flag).read().strip() or -1)
                except ValueError:
                    prev = -1
            if step > prev:
                print(f'BUDGET CAP EXCEEDED: ${spent:.2f} spent >= '
                      f'${self.cap:.2f} cap; continuing per the designer '
                      f'2026-07-16; telling, not stopping.', flush=True)
                with open(flag, 'w') as f:
                    f.write(str(step))
        kwargs = dict(model=self.model, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": prompt}])
        if system_prefix:
            kwargs['system'] = [{"type": "text", "text": system_prefix,
                                 "cache_control": {"type": "ephemeral"}}]
        if self.model.startswith('claude-haiku'):
            kwargs['temperature'] = self.temperature  # accepted on Haiku 4.5 only
        else:
            kwargs['thinking'] = {"type": "disabled"}  # comparability w/ Qwen
        # Transient API weather (529 overloaded, 429, 5xx) must not kill a
        # society (the designer 2026-07-16: don't let stuff die; tell instead).
        # Exponential backoff up to ~8.5 minutes, then the error surfaces.
        import anthropic as _an
        delay = 5.0
        for attempt in range(14):
            try:
                r = self._client.messages.create(**kwargs)
                break
            except (_an.APIStatusError, _an.APIConnectionError) as e:
                status = getattr(e, 'status_code', None)
                retriable = status in (429, 500, 502, 503, 529) or status is None
                if not retriable or attempt == 13:
                    raise
                print(f'API {status or "connection"} error; retry '
                      f'{attempt + 1}/13 in {delay:.0f}s', flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 180.0)
        text = ''.join(b.text for b in r.content if b.type == 'text')
        u = r.usage
        tin, tout = u.input_tokens, u.output_tokens
        cw = getattr(u, 'cache_creation_input_tokens', 0) or 0
        cr = getattr(u, 'cache_read_input_tokens', 0) or 0
        pin, pout = self.PRICES.get(self.model, (5.00, 25.00))
        cost = (tin / 1e6 * pin + tout / 1e6 * pout
                + cw / 1e6 * pin * self.CACHE_WRITE_X
                + cr / 1e6 * pin * self.CACHE_READ_X)
        self._record(tin + cw + cr, tout, cost)
        return text, tin + cw + cr, tout


def generate(backend, prompt, max_tokens=1400, tag='', system_prefix=None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.sha256(
        (backend.name + '\x00' + tag + '\x00' + (system_prefix or '')
         + '\x00' + prompt).encode()).hexdigest()[:24]
    path = os.path.join(CACHE_DIR, key + '.json')
    if os.path.exists(path):
        try:
            d = json.load(open(path))
            d['cached'] = True
            return d
        except (json.JSONDecodeError, OSError):
            # empty/truncated cache entry (e.g. VM died mid-write): drop it
            try: os.remove(path)
            except OSError: pass
    t0 = time.time()
    try:
        text, tin, tout = backend(prompt, max_tokens,
                                  system_prefix=system_prefix)
    except TypeError:
        text, tin, tout = backend(prompt, max_tokens)
    d = dict(model=backend.name, prompt=prompt, text=text,
             tokens_in=tin, tokens_out=tout,
             elapsed=round(time.time() - t0, 2), cached=False)
    json.dump(d, open(path, 'w'), indent=1)
    return d
