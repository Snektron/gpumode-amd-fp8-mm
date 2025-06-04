#!/usr/bin/env python
import sys
import re

with open(sys.argv[1], 'r') as f:
	lines = f.readlines()

next_is_total = False
dim = None
block_tile = None
warp_tile = None
pipeline = None
fused_transpose = None
streamk = None
block_size = None

perf = {}

for line in lines:
	if line.startswith('mma:'):
		next_is_total = True
		# Parse configuration
		m = re.match(r'^mma: \((\d+) (\d+) (\d+)\) \((\d+) (\d+) (\d+)\) \((\d+) (\d+) (\d+)\) (\d) (\d) (\d+) (\d+):.*$', line)
		assert m is not None
		dim = (int(m[1]), int(m[2]), int(m[3]))
		block_tile = (int(m[4]), int(m[5]), int(m[6]))
		warp_tile = (int(m[7]), int(m[8]), int(m[9]))
		pipeline = int(m[10])
		fused_transpose = m[11] == "1"
		streamk = m[12] == "1"
		block_size = int(m[13])
	elif next_is_total:
		next_is_total = False
		m = re.match(r'^total: \d+\.\d+ ms (\d+\.\d+) us$', line)
		assert m is not None
		time = float(m[1])
		if block_tile[0] < 256 and block_tile[1] < 256:
			perf.setdefault(dim, {})[(block_tile, warp_tile, pipeline, fused_transpose, streamk, block_size)] = time

est_score = 1
unique_configs = set()
best = {}
for dim, configs in perf.items():
	configs = list(configs.items())
	configs.sort(key=lambda k: k[1])
	best_config, best_time = configs[0]
	print(dim, best_config, best_time)
	best[dim] = best_config
	est_score *= best_time
	unique_configs.add(best_config)

print('estimated score:', est_score ** (1/len(perf)))
print('unique configs:', len(unique_configs))
print('unique sizes:', len(perf))

first = True
for dim, config in best.items():
	m, n, k = dim
	block_tile, warp_tile, pipeline, fused_transpose, streamk, block_size = config
	bm, bn, bk = block_tile
	wm, wn, wk = warp_tile
	fused_transpose = 'true' if fused_transpose else 'false'
	streamk = 'true' if streamk else 'false'
	ifelse = '    ' if first else ' else '
	first = False
	print(f'{ifelse}if (m == {m} && n == {n} && k == {k}) {{')
	print(f'        run_kernel<{block_size}, cube{{{bm}, {bn}, {bk}}}, cube{{{wm}, {wn}, {wk}}}, {pipeline}, {fused_transpose}, {streamk}>(a, b, as, bs, c, m, n, k, measure);')
	print('    }', end='')
print('')
