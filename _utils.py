import numpy as np

def _check_solver(solver, penalty):
    if solver == 'lbfgs' and penalty not in ['l2', None]:
        raise ValueError(
            "Solver lbfgs supports only 'l2' or None penalties, got %s penalty."
            % penalty   )
    if solver == 'liblinear' and penalty not in ['l1', 'l2']:
        raise ValueError(
            "Solver liblinear supports only 'l1' or 'l2' penalties, got %s penalty."
            % penalty   )
    if solver == 'proximal_grad' and penalty not in ['l1']:
        raise ValueError(
            "Solver proximal_grad supports only 'l1' penalty, got %s penalty."
            % penalty   )
    return solver

def sigmoid(x):
    return 1 / (1 + np.exp(-x))