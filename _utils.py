import numpy as np

def _check_solver(solver, penalty, dual):
    
    if solver not in ["liblinear"] and penalty not in ("l2", "none", None):
        raise ValueError(
            "Solver %s supports only 'l2' or 'none' penalties, got %s penalty."
            % (solver, penalty)
        )
    if solver != "liblinear" and dual:
        raise ValueError(
            "Solver %s supports only dual=False, got dual=%s" % (solver, dual)
        )
    
    if solver == "proximal_grad" and penalty not in ("l1"):
        raise ValueError(
            "Solver %s supports only 'l1' penalty, got %s penalty."
            % (solver, penalty)
        )
        
    return solver

def sigmoid(x):
    return 1 / (1 + np.exp(-x))