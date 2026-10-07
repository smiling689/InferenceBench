import os
import sys


if os.environ.get('R1_PREFILL_CLOCK_PATCH') == '1':
    try:
        from r1_prefill_clock import install

        install()
        print('R1 prefill queue timeout uses a coherent rank-leader clock.', file=sys.stderr, flush=True)
    except Exception as error:
        print(f'R1 prefill clock patch failed: {error}', file=sys.stderr, flush=True)
        os._exit(1)
