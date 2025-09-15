# Save as run_round12.py and execute: python run_round12.py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

def round12_qgd_5d(N=20):
    # 5D Proxy Grid (t,x,y,z,w) - Flattened for toy evo
    coords = np.linspace(-1, 1, N)
    # Mock 5D Gaussian (broadcast for multi-dim sum)
    grid_shape = (N,) * 5
    indices = np.meshgrid(*[coords]*5, indexing='ij')
    r2 = sum(i**2 for i in indices)
    psi = np.exp(-r2 / 0.1)
    rho = np.mean(psi**2)
    
    # Evo Loop: Rolls + potential (flattened for speed)
    psi_flat = psi.flatten()
    total_size = len(psi_flat)
    for _ in range(50):  # Extra steps for 5D chaos
        psi_flat = np.roll(psi_flat, total_size // 5)  # Toy multi-dim shift
        V = 0.02 * np.arange(total_size) / total_size  # Scaled potential
        psi_flat -= 0.002 * V * psi_flat
    rho = np.mean(psi_flat**2)
    
    # FRW w/ Bounce (giggle-scaled rho_crit)
    def friedmann(t, y, rho_crit=0.15):
        a, da = y
        H2 = (8*np.pi/3 * rho) * (1 - rho / rho_crit**2)
        dda = - (4*np.pi/3) * rho * a
        return [da, dda]
    
    t_span = (-0.1, 0.1)
    t_eval = np.linspace(*t_span, 50)
    sol = solve_ivp(friedmann, t_span, [1.0, -0.08], t_eval=t_eval, rtol=1e-12)
    
    # Strain h ~ da/a + sinusoidal "giggle" mod
    h_qgd = np.gradient(sol.y[1]) / sol.y[0] + 0.1 * np.sin(10 * t_eval)
    
    # Mock GW150914 Snippet (bandpassed proxy, Sept 11 vibes)
    gw_t = t_eval
    gw_h = 1.5e-21 * np.sin(2*np.pi*(35*gw_t + 0.5*700*gw_t**2)) * np.exp(-np.abs(gw_t)/1.0) + 1e-23 * np.random.randn(50)
    
    # Fit: Chirp + bounce + giggle absurdity
    def qgd_chirp_giggle(t, A, f0, df, tau, bounce_amp, giggle_freq):
        chirp = A * np.sin(2 * np.pi * (f0 * t + 0.5 * df * t**2)) * np.exp(-np.abs(t)/tau)
        bounce = bounce_amp * np.sign(t) * np.exp(-t**2 / 0.01)
        giggle = 0.05 * np.sin(giggle_freq * t)
        return chirp + bounce + giggle
    popt, _ = curve_fit(qgd_chirp_giggle, gw_t, gw_h, p0=[1e-21, 35, 700, 1.0, -1e-21, 10])
    chi2 = np.sum((gw_h - qgd_chirp_giggle(gw_t, *popt))**2 / 50)
    
    energy_cons = np.std(gw_h) / np.mean(np.abs(gw_h))
    min_a = np.min(sol.y[0])
    bounce = np.any(np.diff(np.sign(sol.y[1])) != 0)
    
    return {'energy_cons': energy_cons, 'min_a': min_a, 'bounce': bounce, 'chi2_dof': chi2/50, 'popt': popt}

# Run it!
results = round12_qgd_5d()
print(results)
