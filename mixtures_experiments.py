import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
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
    # But for alpha=2, we only need J1 and J2.
    # For alpha=3, we need J2 and J3.
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

def V_alpha(alpha, x):
    # V_alpha(x) = (alpha-1) * J_{alpha-2}(x) / J_{alpha-1}(x)
    return (alpha - 1) * J_k(alpha - 2, x) / J_k(alpha - 1, x)

def bentkus_e_value(z, lambd, alpha=2):
    # E_n^B = (z - lambd)_+^alpha / I_alpha(alpha, lambd)
    denom = I_k(alpha, lambd)
    return np.maximum(0, z - lambd)**alpha / denom

def exponential_e_value(z, lambd):
    # E_n^IWR = exp(lambd * z - lambd**2 / 2)
    return np.exp(lambd * z - lambd**2 / 2)

def U_delta_alpha(lambd, delta, alpha=2):
    # Threshold function
    v = V_alpha(alpha, lambd)
    # Denominator in formula: (v-lambd) * J_{alpha-1}(lambd) * G(lambd) = (v-lambd) * I_{alpha-1}(lambd)
    # Actually, easier to use I_alpha(lambd) directly as derived
    denom = I_k(alpha, lambd)
    return lambd + (denom / delta)**(1/alpha)

def A_delta(lambd, delta):
    return (np.log(1/delta) + lambd**2 / 2) / lambd

def find_optimal_lambd_bentkus(delta, alpha=2):
    res = minimize_scalar(lambda l: U_delta_alpha(l, delta, alpha), bounds=(0, 5), method='bounded')
    return res.x

def find_optimal_lambd_exponential(delta):
    return np.sqrt(2 * np.log(1/delta))

# --- Experiment 1: Post-hoc Inference ---
def experiment_posthoc():
    alpha = 2
    deltas_opt = [0.1, 0.05, 0.01]
    
    lambdas_bentkus = [find_optimal_lambd_bentkus(d, alpha) for d in deltas_opt]
    lambdas_exp = [find_optimal_lambd_exponential(d) for d in deltas_opt]
    
    print(f"Optimal lambdas Bentkus: {lambdas_bentkus}")
    print(f"Optimal lambdas Exponential: {lambdas_exp}")
    
    # Mixture e-values
    def mixture_bentkus(z):
        return np.mean([bentkus_e_value(z, l, alpha) for l in lambdas_bentkus])
    
    def mixture_exp(z):
        return np.mean([exponential_e_value(z, l) for l in lambdas_exp])
    
    # Test values for Z
    z_values = [2.1, 2.4, 2.7, 3.0]
    results = []
    for z in z_values:
        eb = mixture_bentkus(z)
        ee = mixture_exp(z)
        # Post-hoc delta is 1/E
        delta_b = 1.0 / eb if eb > 0 else np.inf
        delta_e = 1.0 / ee if ee > 0 else np.inf
        results.append({
            "Z": z,
            "Bentkus E-value": eb,
            "Exp E-value": ee,
            "Bentkus post-hoc delta": delta_b,
            "Exp post-hoc delta": delta_e
        })
    
    df = pd.DataFrame(results)
    print("\nPost-hoc Inference Results:")
    print(df.to_string(index=False))
    return df


# --- Experiment 2: Multiple Testing ---
def experiment_multiple_testing():
    K = 100
    delta_star = 0.1
    alpha = 2
    
    # delta range [0.001, 0.02]
    d_min = 0.1 / K # 0.001
    d_max = 0.1 / 5 # 0.02
    
    l_min = find_optimal_lambd_bentkus(d_max, alpha) # Note: higher delta -> smaller lambda
    l_max = find_optimal_lambd_bentkus(d_min, alpha)
    
    lambdas_grid = np.linspace(l_min, l_max, 10)
    
    # Exponential equivalent grid
    le_min = find_optimal_lambd_exponential(d_max)
    le_max = find_optimal_lambd_exponential(d_min)
    lambdas_grid_exp = np.linspace(le_min, le_max, 10)

    def mixture_bentkus(z):
        return np.mean([bentkus_e_value(z, l, alpha) for l in lambdas_grid])
    
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
                # e-BH is not necessarily monotone in k for the threshold, 
                # but the standard procedure is to find the largest k.
                pass
        return rejections

    proportions = [0, 0.01, 0.05, 0.10, 0.20]
    num_sims = 500
    mu = 3.5 # Signal strength
    
    results = []
    for p in proportions:
        num_non_null = int(p * K)
        num_null = K - num_non_null
        
        rejs_b = []
        rejs_e = []
        
        for _ in range(num_sims):
            z_null = np.random.normal(0, 1, num_null)
            z_non_null = np.random.normal(mu, 1, num_non_null)
            z = np.concatenate([z_null, z_non_null])
            
            ebs = np.array([mixture_bentkus(zi) for zi in z])
            ees = np.array([mixture_exp(zi) for zi in z])
            
            rejs_b.append(e_bh(ebs, delta_star))
            rejs_e.append(e_bh(ees, delta_star))
            
        results.append({
            "Prop. Non-Null": p,
            "Bentkus Avg Rejections": np.mean(rejs_b),
            "Exp Avg Rejections": np.mean(rejs_e)
        })
        
    df = pd.DataFrame(results)
    print("\nMultiple Testing Results:")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    df_posthoc = experiment_posthoc()
    df_posthoc.round(4).to_latex("tables/experiment_posthoc.tex", index=False)
    
    df_multiple = experiment_multiple_testing()
    df_multiple.round(4).to_latex("tables/experiment_multiple_testing.tex", index=False)
