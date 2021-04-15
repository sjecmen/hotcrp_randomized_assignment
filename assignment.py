import numpy as np
import gurobipy as gp
from gurobipy import GRB
from cffi import FFI 
from _bvn_extension.lib import run_bvn

rng = np.random.default_rng()

# Inputs: similarity matrix, conflict matrix, probability limit matrix, reviewer loads, and paper loads
# Output: fractional assignment value, fractional assignment matrix
# All matrices of size (#revs, #paps)
def find_fractional_assignment(S, M, Q, revloads, paploads, revload_lower=0):
    if type(revloads) == int:
        revloads = np.full(S.shape[0], revloads)
    if type(paploads) == int:
        if paploads != 3:
            print('paper load of', paploads)
        paploads = np.full(S.shape[1], paploads)

    model = gp.Model("my_model") 
    model.setParam('OutputFlag', 0)
    obj = 0
    n, d = S.shape
    assert len(revloads) == n and len(paploads) == d and S.shape == M.shape and S.shape == Q.shape
    review_demand = np.sum(paploads)
    review_supply = np.sum(revloads)

    if review_demand > review_supply:
        print('infeasible assignment')
        raise RuntimeError('infeasible')

    A = [[None for j in range(d)] for i in range(n)]

    for i in range(n):
        for j in range(d):
            if (M[i, j] == 1):
                v = model.addVar(lb = 0, ub = 0, name = f"{i} {j}")
            else:
                v = model.addVar(lb = 0, ub = Q[i, j], name = f"{i} {j}") 
                
            A[i][j] = v
            obj += v * S[i, j]

    model.setObjective(obj, GRB.MAXIMIZE)
    
    for i in range(n):
        papers = 0
        for j in range(d):
            papers += A[i][j]
        model.addConstr(papers <= revloads[i]) 
        model.addConstr(papers >= revload_lower[i])
    
    for j in range(d):
        reviewers = 0
        for i in range(n):
            reviewers += A[i][j]
        model.addConstr(reviewers == paploads[j])
    
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print("WARNING: model not solved")
        raise RuntimeError('unsolved')

    F = np.zeros_like(S)
    for i in range(n):
        for j in range(d):
            F[i, j] = A[i][j].x

    return model.objVal, F 


# Input: fractional assignment matrix
# Output: {0, 1} deterministic assignment matrix sampled from the fractional assignment
# All matrices of size (#revs, #paps)
def sample_assignment(F):
    ffi = FFI()
    nrev, npap = F.shape
    F = F.T.flatten().astype(np.double) # bvn written for size (#paps, #revs)
    Fbuf = ffi.new("double[]", nrev*npap)
    for i in range(F.size):
        Fbuf[i] = F[i]
    Sbuf = ffi.new("int[]", nrev)
    for i in range(nrev):
        Sbuf[i] = 1
    
    run_bvn(Fbuf, Sbuf, npap, nrev)

    flow_matrix = np.zeros((npap, nrev))
    for i in range(F.size):
        coords = np.unravel_index(i, (npap, nrev))
        flow_matrix[coords] = Fbuf[i]
    return flow_matrix.T
