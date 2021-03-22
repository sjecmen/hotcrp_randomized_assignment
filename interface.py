import csv
import numpy as np

# Get list of reviewer IDs
# input file from: /users ==> download "PC info"
def parse_reviewers(fname):
    with open(fname, newline='') as f:
        r = csv.reader(f)
        header = next(r)
        assert all([a == b for a, b in zip(header, ['first', 'last', 'email', 'affiliation', 'country', 'roles'])])
        reviewer_ids = []
        for row in r:
            if 'pc' in row[5] and 'chair' not in row[5]: # can change
                reviewer_ids.append(row[2])
    assert len(reviewer_ids) == len(set(reviewer_ids))
    return reviewer_ids


# Get list of paper IDs
# input file from: /reviewprefs => download "CSV"
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
# input file from: /reviewprefs => download "PC review preferences"
# sim_func is a function from preference (bid) and topic score to overall similarity
def sims_from_csv(fname, reviewer_ids, paper_ids, sim_func=lambda p, t: p + t):
    testing = True

    reviewer_indices = {r: i for i, r in enumerate(reviewer_ids)}
    paper_indices = {p: i for i, p in enumerate(paper_ids)}

    shape = (len(reviewer_ids), len(paper_ids))
    S, M = np.zeros(shape), np.zeros(shape)

    with open(fname, newline='') as f:
        r = csv.reader(f)
        header = next(r) # header
        assert all([a == b for a, b in zip(header, ['paper', 'title', 'first', 'last', 'email', 'preference', 'topic_score', 'conflict'])])
        for row in r:
            paper_id = row[0]
            reviewer_id = row[4]
            preference_score = int(row[5]) if row[5] != '' else 0
            topic_score = int(row[6]) if row[6] != '' else 0
            similarity = sim_func(preference_score, topic_score)
            conflict = 1 if row[7] == 'conflict' else 0

            if not testing or (reviewer_id in reviewer_indices and paper_id in paper_indices):
                r = reviewer_indices[reviewer_id]
                p = paper_indices[paper_id]
                S[r, p] = similarity
                M[r, p] = conflict
    return S, M

# Write assignment to output file
# A: {0, 1} assignment matrix of size (#revs, #paps)
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

