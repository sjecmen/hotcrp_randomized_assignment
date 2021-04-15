from assignment import *
from interface import *
import numpy as np
import matplotlib.pyplot as plt

input_fname = 'acmcompass2021-allprefs.csv'
revs_fname = 'acmcompass2021-pcinfo.csv'
paps_fname = 'acmcompass2021-data.csv'
papjson_fname = 'acmcompass2021-data.json'
output_fname = 'acmcompass2021-assignments'

paper_load = 3
pc_load = 3
chair_load = 0 # can go down to 0
pc_lower = 2
chair_lower = 0

pc_ids, chair_ids = parse_reviewers(revs_fname)
rev_ids = pc_ids + chair_ids
pap_ids = parse_papers(paps_fname)

rev_loads = np.zeros(len(rev_ids))
rev_loads[:len(pc_ids)] = pc_load
rev_loads[len(pc_ids):] = chair_load
rev_lowers = np.zeros(len(rev_ids))
rev_lowers[:len(pc_ids)] = pc_lower
rev_lowers[len(pc_ids):] = chair_lower


# track reading
track_bonus = 5
T = parse_track_bonuses(rev_ids, pap_ids, revs_fname, papjson_fname, track_bonus)

bid_scale = 4
norm = True
S, M = sims_from_csv(input_fname, rev_ids, pap_ids, bid_scale, norm, T)

print('papload, pcload, chairload, rev_lb, chair_lb:', paper_load, pc_load, chair_load, pc_lower, chair_lower)
print('track bonus', track_bonus)
print('shape of S', S.shape)

qs = []
vs = []
for q in np.linspace(0.1, 1, 10):
    Q = np.full_like(S, q)
    try:
        v, F = find_fractional_assignment(S, M, Q, rev_loads, paper_load, rev_lowers)
    except RuntimeError:
        continue
    qs.append(q)
    vs.append(v)

percent_vs = [v / max(vs) for v in vs]
print('\n'.join([str(x) for x in zip(qs, percent_vs)]))

q = 0.7
print('final q value is', q)
Q = np.full_like(S, q)
v, F = find_fractional_assignment(S, M, Q, rev_loads, paper_load, rev_lowers)
A = sample_assignment(F)
assignment_to_csv(output_fname+'-q70.csv', A, rev_ids, pap_ids)

q = 1
print('also saving q', q)
Q = np.full_like(S, q)
v, F = find_fractional_assignment(S, M, Q, rev_loads, paper_load, rev_lowers)
A = sample_assignment(F)
assignment_to_csv(output_fname+'-q100.csv', A, rev_ids, pap_ids)

