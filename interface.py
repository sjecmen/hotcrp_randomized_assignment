import csv
import numpy as np

# Get list of reviewer IDs
#   - fname: filename of the input file from /users ==> download "PC info"
def parse_reviewers(fname):
    with open(fname, newline='') as f:
        r = csv.reader(f)
        header = next(r)
        assert all([a == b for a, b in zip(header, ['first', 'last', 'email', 'affiliation', 'country', 'roles'])])
        reviewer_ids = []
        chair_ids = []
        for row in r:
            if 'pc' in row[5] and 'chair' not in row[5]: # can change
                reviewer_ids.append(row[2])
            if 'pc' in row[5] and 'chair' in row[5]:
                chair_ids.append(row[2])
    assert len(reviewer_ids) == len(set(reviewer_ids))
    assert len(chair_ids) == len(set(chair_ids))
    return reviewer_ids, chair_ids


# Get list of paper IDs
#   - fname: filename of the input file from /reviewprefs => download "CSV"
def parse_papers(fname):
    with open(fname, newline='') as f:
        r = csv.reader(f)
        header = next(r)
        assert all([a == b or b == None for a, b in zip(header, ['ID', 'Title', 'Authors', None, '# Reviews', 'Status', 'OveMer'])])
        paper_ids = []
        for row in r:
            paper_ids.append(row[0])
    assert len(paper_ids) == len(set(paper_ids))
    return paper_ids 


# Get similarity and conflict matrices of size (#revs, #paps)
#   - fname: filename of the input file from /reviewprefs => download "PC review preferences"
#   - reviewer_ids, paper_ids: lists of IDs
def sims_from_csv(fname, reviewer_ids, paper_ids, bid_scale=4, norm=True):
    print('constructing similarities with scale', bid_scale, 'and norm', norm)
    reviewer_indices = {r: i for i, r in enumerate(reviewer_ids)}
    paper_indices = {p: i for i, p in enumerate(paper_ids)}

    shape = (len(reviewer_ids), len(paper_ids))
    S, M = np.zeros(shape), np.zeros(shape)
    B = np.zeros(shape) # unscaled bids

    with open(fname, newline='') as f:
        r = csv.reader(f)
        header = next(r) # header
        assert all([a == b for a, b in zip(header, ['paper', 'title', 'first', 'last', 'email', 'preference', 'topic_score', 'conflict'])])
        for row in r:
            paper_id = row[0]
            reviewer_id = row[4]
            preference_score = int(row[5]) if row[5] != '' else 0
            topic_score = int(row[6]) if row[6] != '' else 0
            conflict = 1 if row[7] == 'conflict' or preference_score == -100 else 0
            r = reviewer_indices[reviewer_id]
            p = paper_indices[paper_id]

            M[r, p] = conflict
            if conflict == 1:
                continue # don't use topic or preference if conflict

            B[r, p] = preference_score
            S[r, p] = topic_score

    # S contains unscaled topic_scores, so scale to [0, 1]
    S = (S - np.min(S)) / (np.max(S) - np.min(S))
    assert(np.all(S >= 0) and np.all(S <= 1))
    S[M == 1] = 0 # reset to 0 for normalization
    #print('scaled topics', S)

    for r in range(S.shape[0]): # could remove loop
        maxbid = np.max(np.abs(B[r, :]))
        if maxbid > 100:
            print('WARNING: very large bid of', maxbid, 'detected')
        if maxbid != 0:
            B[r, :] /= maxbid
    assert(np.all(B >= -1) and np.all(B <= 1))
    #print('scaled bids', B)

    # S and B now contain scaled topic scores and bids, so compute
    S = np.power(bid_scale, B) * S
    #print('unnormed sims', S)

    # normalization
    if norm:
        for i in range(S.shape[0]): # could remove loop
            total = np.sum(S[i, :])
            if total != 0:
                S[i, :] /= total
    return S, M

# Write assignment to output file
#   - fname: output file name
#   - A: {0, 1} assignment matrix of size (#revs, #paps)
#   - reviewer_ids, paper_ids: lists of IDs
def assignment_to_csv(fname, A, reviewer_ids, paper_ids):
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['paper', 'action', 'email', 'reviewtype'])
        w.writerow(['all', 'clearreview', 'all', 'any'])
        #for pid in paper_ids: # for testing
        #    w.writerow([pid, 'clearreview', 'all', 'any'])
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] == 1:
                    w.writerow([paper_ids[j], 'review', reviewer_ids[i], 'primary'])

