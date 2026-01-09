import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm > 1e-8:
        return v / norm
    return v  # Avoid divide by zero

def raw_similarity(a, b):
    return np.real(np.vdot(a, b))

def gen_vector(d):
    phases = np.random.uniform(0, 2*np.pi, d)
    return np.exp(1j * phases)

# Parameters
d = 8192
n_codebook = 100
codebook = [gen_vector(d) for _ in range(n_codebook)]

# Filler matrix d x n_codebook
F_mat = np.array(codebook).T

# Stable resonator with ReLU-like positive feedback and safe norm
def stable_resonator(unbound, F_mat, iters=100):
    x = unbound.copy()
    for _ in range(iters):
        sims = np.real(F_mat.conj().T @ x)
        sims = np.maximum(sims, 0)  # Positive feedback only (standard stabilization)
        x = F_mat @ sims
        x = normalize(x)
    return x

# Hierarchy (perfect raw d = exact)
depth = 20
chain = [gen_vector(d) for _ in range(depth)]
phases = np.pi / 4 * np.arange(depth)

h = chain[0]
for i in range(1, depth):
    h = h * chain[i] * np.exp(1j * phases[i])

target = gen_vector(d)
h = h * target

unbound = h.copy()
for i in range(depth - 1, -1, -1):
    unbound = unbound * np.conj(chain[i]) * np.exp(-1j * phases[i])

raw_clean = raw_similarity(unbound, target)
print(f"Hierarchy clean raw: {raw_clean:.1f} (~{d} = perfect)")

# Counterfactual
h_counter = h.copy()
flip_phase = np.pi * np.sum(phases[depth//3:])
h_counter = h_counter * np.exp(1j * flip_phase)

unbound_counter = h_counter.copy()
for i in range(depth - 1, -1, -1):
    unbound_counter = unbound_counter * np.conj(chain[i]) * np.exp(-1j * phases[i])

raw_counter = raw_similarity(unbound_counter, target)
print(f"Counterfactual raw: {raw_counter:.1f} (sharp drop)\n")

# Overload
n_true = 50
true_pairs = [(i, (i + 17) % n_codebook) for i in range(n_true)]
s_clean = sum(codebook[r] * codebook[f] for r, f in true_pairs)

loads = [0, 5000, 10000, 20000, 30000]
success_rates = []
mean_raw_trues = []

for load in loads:
    overload = sum(gen_vector(d) for _ in range(load))
    s = s_clean + overload + 0.7 * (np.random.randn(d) + 1j * np.random.randn(d))
    
    correct = 0
    raw_trues_list = []
    for r, true_f in true_pairs:
        unbound = np.conj(codebook[r]) * s
        cleaned = stable_resonator(unbound, F_mat)
        raw_sims = np.real(F_mat.conj().T @ cleaned)
        pred = np.argmax(raw_sims)
        raw_trues_list.append(raw_sims[true_f])
        if pred == true_f:
            correct += 1
    
    success_rates.append(correct / n_true)
    mean_raw_trues.append(np.mean(raw_trues_list))
    print(f"Load {load}: Success {success_rates[-1]:.3f}, Mean raw true after cleanup {mean_raw_trues[-1]:.4f}")

plt.plot(loads, mean_raw_trues, marker='o', label='Mean Raw True After Cleanup')
plt.plot(loads, success_rates, marker='s', label='Success Rate')
plt.xlabel('Overload Vectors')
plt.ylabel('Metric')
plt.title('Stable Resonator IE-HRR (No Warnings, Amplified Bias)')
plt.legend()
plt.grid(True)
plt.show()

print("\nFixed: safe normalize, positive sims feedback, list for mean.")
print("No warnings, bias amplified cleanly—expect success hold high, raw true strong.")