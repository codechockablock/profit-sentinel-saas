import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(42)
random.seed(42)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm > 1e-8:
        return v / norm
    return np.zeros_like(v)

def raw_similarity(a, b):
    return np.real(np.vdot(a, b))

def gen_unit_vector(d):
    phases = np.random.uniform(0, 2*np.pi, d)
    v = np.exp(1j * phases)
    return normalize(v)  # Unit norm now

d = 4000
n_codebook = 100
codebook = [gen_unit_vector(d) for _ in range(n_codebook)]
F_mat = np.array(codebook).T

def convergence_lock_resonator(unbound, F_mat, iters=200, alpha=0.95, power=1.2):
    x = normalize(unbound)
    for _ in range(iters):
        sims = np.real(F_mat.conj().T @ x)
        sims = np.maximum(sims, 0)
        sims = sims ** power
        projection = F_mat @ sims
        x = alpha * x + (1 - alpha) * projection
        x = normalize(x)
    return x

# Hierarchy clean (should be ~1.0 now)
depth = 20
chain = [gen_unit_vector(d) for _ in range(depth)]
phases = np.pi / 4 * np.arange(depth)

h = chain[0]
for i in range(1, depth):
    h = h * chain[i] * np.exp(1j * phases[i])
h = normalize(h)  # Keep unit

target = gen_unit_vector(d)
h = h * target
h = normalize(h)

unbound = h.copy()
for i in range(depth - 1, -1, -1):
    unbound = unbound * np.conj(chain[i]) * np.exp(-1j * phases[i])
unbound = normalize(unbound)

raw_clean = raw_similarity(unbound, target)
print(f"Hierarchy clean raw: {raw_clean:.3f}")

# Full sim
def run_full_sim(alpha, power, noise_coeff=0.3):
    loads = [0, 5000, 10000, 20000, 30000]
    n_true = 50
    true_pairs = [(i, (i + 17) % n_codebook) for i in range(n_true)]
    s_clean = sum(codebook[r] * codebook[f] for r, f in true_pairs)
    s_clean = normalize(s_clean)  # Optional, but keeps scaling sane
    success_rates = []
    mean_raw_trues = []
    for load in loads:
        if load == 0:
            overload = np.zeros(d, dtype=complex)
        else:
            overload = (np.random.randn(d) + 1j * np.random.randn(d))
            overload = overload / np.linalg.norm(overload) * np.sqrt(load) if np.linalg.norm(overload) > 0 else np.zeros(d, dtype=complex)
        s = s_clean + overload + noise_coeff * (np.random.randn(d) + 1j * np.random.randn(d))
        s = normalize(s)  # Optional
        correct = 0
        raw_trues_list = []
        for r, true_f in true_pairs:
            unbound = np.conj(codebook[r]) * s
            cleaned = convergence_lock_resonator(unbound, F_mat, alpha=alpha, power=power)
            raw_sims = np.real(F_mat.conj().T @ cleaned)
            pred = np.argmax(raw_sims)
            raw_trues_list.append(raw_sims[true_f])
            if pred == true_f:
                correct += 1
        success_rates.append(correct / n_true)
        mean_raw_trues.append(np.mean(raw_trues_list))
    return loads, success_rates, mean_raw_trues

# Baseline
print("\nBaseline (alpha=0.95, power=1.2):")
loads, base_succ, base_raw = run_full_sim(0.95, 1.2)
for load, succ, raw in zip(loads, base_succ, base_raw):
    print(f"Load {load}: Success {succ:.3f}, Mean raw true {raw:.2f}")

# GA (sped up)
def evaluate(alpha, power):
    num_trials = 4
    n_true_ga = 10
    loads_ga = [20000, 30000]
    true_pairs_ga = [(i, (i + 17) % n_codebook) for i in range(n_true_ga)]
    s_clean_ga = sum(codebook[r] * codebook[f] for r, f in true_pairs_ga)
    s_clean_ga = normalize(s_clean_ga)
    trial_fitnesses = []
    for _ in range(num_trials):
        noise_coeff = np.random.uniform(0.2, 0.8)
        load_successes = []
        for load in loads_ga:
            overload = (np.random.randn(d) + 1j * np.random.randn(d))
            overload = overload / np.linalg.norm(overload) * np.sqrt(load) if np.linalg.norm(overload) > 0 else np.zeros(d, dtype=complex)
            s = s_clean_ga + overload + noise_coeff * (np.random.randn(d) + 1j * np.random.randn(d))
            correct = 0
            for r, true_f in true_pairs_ga:
                unbound = np.conj(codebook[r]) * s
                cleaned = convergence_lock_resonator(unbound, F_mat, alpha=alpha, power=power)
                raw_sims = np.real(F_mat.conj().T @ cleaned)
                pred = np.argmax(raw_sims)
                if pred == true_f:
                    correct += 1
            load_successes.append(correct / n_true_ga)
        trial_fitnesses.append(np.mean(load_successes))
    return np.mean(trial_fitnesses)

pop_size = 15
gens = 20
pop = [[np.random.uniform(0.9, 0.99), np.random.uniform(1.0, 3.0)] for _ in range(pop_size)]  # Wider power search
best_fitness = -np.inf
best_params = [0.95, 1.2]

print("\nRunning GA (faster now)...")
for gen in range(gens):
    fitnesses = [evaluate(ind[0], ind[1]) for ind in pop]
    current_best_idx = np.argmax(fitnesses)
    if fitnesses[current_best_idx] > best_fitness:
        best_fitness = fitnesses[current_best_idx]
        best_params = pop[current_best_idx][:]

    print(f"Gen {gen+1}/{gens}: Best {best_fitness:.3f} (alpha={best_params[0]:.3f}, power={best_params[1]:.3f})")

    # Elitism + breed
    sorted_idx = np.argsort(fitnesses)[::-1]
    elites = [pop[i][:] for i in sorted_idx[:4]]
    new_pop = elites[:]
    while len(new_pop) < pop_size:
        p1, p2 = random.choices(elites, k=2)
        child = [p1[0] if random.random() < 0.5 else p2[0], p1[1] if random.random() < 0.5 else p2[1]]
        if random.random() < 0.4: child[0] = np.clip(child[0] + np.random.normal(0, 0.01), 0.9, 0.99)
        if random.random() < 0.4: child[1] = np.clip(child[1] + np.random.normal(0, 0.2), 1.0, 3.0)
        new_pop.append(child)
    pop = new_pop

# Optimized
print("\nOptimized full sim:")
loads, opt_succ, opt_raw = run_full_sim(best_params[0], best_params[1])
for load, succ, raw in zip(loads, opt_succ, opt_raw):
    print(f"Load {load}: Success {succ:.3f}, Mean raw true {raw:.2f}")

# Plot comparison
plt.figure()
plt.plot(loads, base_succ, 'o-', label='Baseline Success')
plt.plot(loads, opt_succ, 's-', label='Optimized Success')
plt.plot(loads, base_raw, '--', label='Baseline Raw True')
plt.plot(loads, opt_raw, '--', label='Optimized Raw True')
plt.xlabel('Overload Vectors')
plt.ylabel('Success / Raw Similarity')
plt.title('GA-Tuned vs Baseline')
plt.legend()
plt.grid(True)
plt.show()

# Variable noise test on optimized
# (Add the var test from before if you want)