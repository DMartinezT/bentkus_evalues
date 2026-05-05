import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from tqdm import tqdm
import pandas as pd

def phi(x):
    return norm.pdf(x)

def G(x):
    return 1 - norm.cdf(x)

def M(x):
    # Mills ratio
    return phi(x) / G(x)

def J_k(k, x):
    if k == 0:
        return np.ones_like(x)
    if k == 1:
        return M(x) - x
    if k == 2:
        return 1 + x**2 - x*M(x)
    if k == 3:
        return (x**2 + 2)*M(x) - (x**3 + 3*x)
    # Recursion for higher k if needed: J_k = (k-1)J_{k-2} - x J_{k-1}
    j_prev2 = np.ones_like(x)
    j_prev1 = M(x) - x
    for i in range(2, k + 1):
        j_curr = (i - 1) * j_prev2 - x * j_prev1
        j_prev2 = j_prev1
        j_prev1 = j_curr
    return j_prev1

def I_k(k, x):
    # E[(Z-x)_+^k] = J_k(x) * G(x)
    return J_k(k, x) * G(x)


def bentkus_e_value(z, lambd, alpha=2):
    # Indicator e-value for alpha = 0 (B0)
    if alpha == 0:
        return np.where(z >= lambd, 1.0 / G(lambd), 0.0)
    
    # E_n^B = (z - lambd)_+^alpha / I_alpha(alpha, lambd)
    denom = I_k(alpha, lambd)
    return np.maximum(0, z - lambd)**alpha / denom

def exponential_e_value(z, lambd):
    # E_n^IWR = exp(lambd * z - lambd**2 / 2)
    return np.exp(lambd * z - lambd**2 / 2)

def U_delta_alpha(lambd, delta, alpha=2):
    # Threshold function for alpha = 0 (B0)
    if alpha == 0:
        return np.where(G(lambd) <= delta, lambd, np.inf)
    
    # Threshold function for alpha > 0
    denom = I_k(alpha, lambd)
    return lambd + (denom / delta)**(1/alpha)

def A_delta(lambd, delta):
    return (np.log(1/delta) + lambd**2 / 2) / lambd

def find_optimal_lambd_bentkus(delta, alpha=2):
    if alpha == 0:
        # Analytical minimum avoiding optimizer failures on step functions
        return norm.ppf(1 - delta) 
    
    res = minimize_scalar(lambda l: U_delta_alpha(l, delta, alpha), bounds=(0, 5), method='bounded')
    return res.x

def find_optimal_lambd_exponential(delta):
    return np.sqrt(2 * np.log(1/delta))

# --- Experiment 1: Post-hoc Inference ---
def experiment_posthoc():

    np.random.seed(42)

    alpha_B0 = 0
    alpha_B1 = 1
    alpha_B2 = 2
    deltas_opt = [0.1, 0.05, 0.01]
    
    lambdas_b0 = [find_optimal_lambd_bentkus(d, alpha_B0) for d in deltas_opt]
    lambdas_b1 = [find_optimal_lambd_bentkus(d, alpha_B1) for d in deltas_opt]
    lambdas_b2 = [find_optimal_lambd_bentkus(d, alpha_B2) for d in deltas_opt]
    lambdas_exp = [find_optimal_lambd_exponential(d) for d in deltas_opt]
    
    print(f"Optimal lambdas B0 (alpha=0): {lambdas_b0}")
    print(f"Optimal lambdas B1 (alpha=1): {lambdas_b1}")
    print(f"Optimal lambdas B2 (alpha=2): {lambdas_b2}")
    print(f"Optimal lambdas Exponential: {lambdas_exp}")
    
    # Mixture e-values
    def mixture_b0(z):
        return np.mean([bentkus_e_value(z, l, alpha_B0) for l in lambdas_b0])

    def mixture_b1(z):
        return np.mean([bentkus_e_value(z, l, alpha_B1) for l in lambdas_b1])

    def mixture_b2(z):
        return np.mean([bentkus_e_value(z, l, alpha_B2) for l in lambdas_b2])
    
    def mixture_exp(z):
        return np.mean([exponential_e_value(z, l) for l in lambdas_exp])
    
    # Test values for Z
    z_values = [2.1, 2.4, 2.7, 3.0]
    results = []
    for z in z_values:
        e0 = mixture_b0(z)
        e1 = mixture_b1(z)
        e2 = mixture_b2(z)
        ee = mixture_exp(z)
        
        # Post-hoc delta is 1/E
        delta_0 = 1.0 / e0 if e0 > 0 else np.inf
        delta_1 = 1.0 / e1 if e1 > 0 else np.inf
        delta_2 = 1.0 / e2 if e2 > 0 else np.inf
        delta_e = 1.0 / ee if ee > 0 else np.inf
        
        results.append({
            "$Z$": z,
            "$E_n^{\\text{B0}}(Z; \\pi)$": e0,
            "$E_n^{\\text{B1}}(Z; \\pi)$": e1,
            "$E_n^{\\text{B2}}(Z; \\pi)$": e2,
            "$E_n^{\\text{IWR}}(Z; \\pi)$": ee,
            "B0 post-hoc $\\delta$": delta_0,
            "B1 post-hoc $\\delta$": delta_1,
            "B2 post-hoc $\\delta$": delta_2,
            "IWR post-hoc $\\delta$": delta_e
        })
    
    df = pd.DataFrame(results)
    print("\nPost-hoc Inference Results:")
    print(df.to_string(index=False))
    return df


# --- Experiment 2: Multiple Testing ---
def experiment_multiple_testing():

    np.random.seed(42)

    K = 100
    delta_star = 0.1
    alpha_B0 = 0
    alpha_B1 = 1
    alpha_B2 = 2
    
    # delta range [0.001, 0.02]
    d_min = 0.1 / K # 0.001
    d_max = 0.1 / 5 # 0.02
    
    # Grid for B0 (alpha=0)
    l_min_b0 = find_optimal_lambd_bentkus(d_max, alpha_B0)
    l_max_b0 = find_optimal_lambd_bentkus(d_min, alpha_B0)
    lambdas_grid_b0 = np.linspace(l_min_b0, l_max_b0, 10)

    # Grid for B1 (alpha=1)
    l_min_b1 = find_optimal_lambd_bentkus(d_max, alpha_B1)
    l_max_b1 = find_optimal_lambd_bentkus(d_min, alpha_B1)
    lambdas_grid_b1 = np.linspace(l_min_b1, l_max_b1, 10)

    # Grid for B2 (alpha=2)
    l_min_b2 = find_optimal_lambd_bentkus(d_max, alpha_B2) 
    l_max_b2 = find_optimal_lambd_bentkus(d_min, alpha_B2)
    lambdas_grid_b2 = np.linspace(l_min_b2, l_max_b2, 10)
    
    # Grid for Exponential
    le_min = find_optimal_lambd_exponential(d_max)
    le_max = find_optimal_lambd_exponential(d_min)
    lambdas_grid_exp = np.linspace(le_min, le_max, 10)

    def mixture_b0(z):
        return np.mean([bentkus_e_value(z, l, alpha_B0) for l in lambdas_grid_b0])

    def mixture_b1(z):
        return np.mean([bentkus_e_value(z, l, alpha_B1) for l in lambdas_grid_b1])

    def mixture_b2(z):
        return np.mean([bentkus_e_value(z, l, alpha_B2) for l in lambdas_grid_b2])
    
    def mixture_exp(z):
        return np.mean([exponential_e_value(z, l) for l in lambdas_grid_exp])

    def e_bh(evals, alpha_fdr):
        K = len(evals)
        sorted_evals = np.sort(evals)[::-1]
        rejections = 0
        for k in range(1, K + 1):
            if sorted_evals[k-1] >= K / (k * alpha_fdr):
                rejections = k
            else:
                pass
        return rejections

    proportions = [0.01, 0.025, 0.05, 0.075, 0.1]
    num_sims = 100
    mu = 3.5 # Signal strength
    
    results = []
    for p in proportions:
        num_non_null = int(p * K)
        num_null = K - num_non_null
        
        rejs_b0 = []
        rejs_b1 = []
        rejs_b2 = []
        rejs_e = []
        
        for _ in tqdm(range(num_sims)):
            z_null = np.random.normal(0, 1, num_null)
            z_non_null = np.random.normal(mu, 1, num_non_null)
            z = np.concatenate([z_null, z_non_null])
            
            e0s = np.array([mixture_b0(zi) for zi in z])
            e1s = np.array([mixture_b1(zi) for zi in z])
            e2s = np.array([mixture_b2(zi) for zi in z])
            ees = np.array([mixture_exp(zi) for zi in z])
            
            rejs_b0.append(e_bh(e0s, delta_star))
            rejs_b1.append(e_bh(e1s, delta_star))
            rejs_b2.append(e_bh(e2s, delta_star))
            rejs_e.append(e_bh(ees, delta_star))
            
        results.append({
            "Prop. Non-Null": p,
            "B0 Avg Reject.": np.mean(rejs_b0),
            "B1 Avg Reject.": np.mean(rejs_b1),
            "B2 Avg Reject.": np.mean(rejs_b2),
            "IWR Avg Reject.": np.mean(rejs_e)
        })
        
    df = pd.DataFrame(results)
    print("\nMultiple Testing Results:")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    df_posthoc = experiment_posthoc()
    df_posthoc.to_latex("tables/experiment_posthoc.tex", index=False, float_format="%.3f")
    
    df_multiple = experiment_multiple_testing()
    df_multiple.set_index("Prop. Non-Null").T.to_latex("tables/experiment_multiple_testing.tex", float_format="%.2f")