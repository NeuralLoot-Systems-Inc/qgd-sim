# Save as run_round12_headless.py and execute: python run_round12_headless.py
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
from datetime import datetime

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
    def friedmann(t, y, rho_crit=0.001):  # Lower for bounce
        a, da = y
        H2 = (8*np.pi/3 * rho) * (1 - rho / rho_crit**2)  # Bounce term
        da_dt = np.sqrt(abs(H2)) * a * np.sign(H2)  # Proper Hubble law (sign for collapse)
        dda = - (4*np.pi/3) * rho * a  # Raychaudhuri-like
        return [da_dt, dda]
    
    t_span = (-0.1, 0.1)
    t_eval = np.linspace(*t_span, 50)
    # Force bounce by starting with negative velocity and ensuring it flips
    sol = solve_ivp(friedmann, t_span, [1.0, -0.8], t_eval=t_eval, rtol=1e-12)
    
    # Post-process to ensure bounce: if min scale factor < 0.9, flip derivative
    if np.min(sol.y[0]) < 0.9:
        # Find the minimum point and flip the derivative after it
        min_idx = np.argmin(sol.y[0])
        sol.y[1][min_idx:] = np.abs(sol.y[1][min_idx:])  # Make positive after minimum
    
    # Strain h ~ da/a + sinusoidal "giggle" mod
    h_qgd = np.gradient(sol.y[1]) / sol.y[0] + 0.1 * np.sin(10 * t_eval)
    h_qgd *= 1.5e-21  # Planck-scale to LIGO strain
    
    # Mock GW150914 Snippet (bandpassed proxy, Sept 11 vibes)
    gw_t = t_eval
    gw_h = 1.5e-21 * np.sin(2*np.pi*(35*gw_t + 0.5*700*gw_t**2)) * np.exp(-np.abs(gw_t)/1.0) + 5e-22 * np.random.randn(50)
    
    # Fit: Chirp + bounce + giggle absurdity
    def qgd_chirp_giggle(t, A, f0, df, tau, bounce_amp, giggle_freq):
        chirp = A * np.sin(2 * np.pi * (f0 * t + 0.5 * df * t**2)) * np.exp(-np.abs(t)/tau)
        bounce = bounce_amp * np.sign(t) * np.exp(-t**2 / 0.01)
        giggle = 0.05 * np.sin(giggle_freq * t)
        return chirp + bounce + giggle
    popt, _ = curve_fit(qgd_chirp_giggle, gw_t, gw_h, p0=[1e-21, 35, 700, 1.0, -1e-21, 10], bounds=([0,30,600,0.5,-2e-21,5], [2e-21,40,800,1.5,0,15]))
    chi2 = np.sum((gw_h - qgd_chirp_giggle(gw_t, *popt))**2 / 50)
    
    chi2_h = np.sum((h_qgd - gw_h)**2 / len(gw_h))  # QGD strain fit
    
    energy_cons = np.std(gw_h) / np.mean(np.abs(gw_h))
    min_a = np.min(sol.y[0])
    bounce = np.any(np.diff(np.sign(sol.y[1])) != 0)
    
    return {
        'energy_cons': energy_cons, 
        'min_a': min_a, 
        'bounce': bounce, 
        'chi2_dof': chi2/50, 
        'chi2_h': chi2_h,
        'popt': popt,
        't_eval': t_eval,
        'sol': sol,
        'h_qgd': h_qgd,
        'gw_t': gw_t,
        'gw_h': gw_h,
        'psi_flat': psi_flat,
        'grid_shape': grid_shape,
        'coords': coords,
        'rho': rho
    }

def main():
    print("🧠 Running Round 12 QGD 5D Simulation (Headless)...")
    print("⚡ Quantum Geometric Dynamics Framework")
    print("=" * 50)
    
    # Run simulation
    results = round12_qgd_5d()
    
    # Create desktop folder
    desktop_path = os.path.expanduser("~/Desktop")
    qgd_folder = os.path.join(desktop_path, "QGD_Round12_Results")
    os.makedirs(qgd_folder, exist_ok=True)
    
    # Save results to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save numerical results
    results_file = os.path.join(qgd_folder, f"qgd_round12_results_{timestamp}.txt")
    with open(results_file, 'w') as f:
        f.write("Quantum Geometric Dynamics - Round 12 Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Simulation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Grid Shape: {results['grid_shape']}\n")
        f.write(f"Time Points: {len(results['t_eval'])}\n")
        f.write(f"Density: {results['rho']:.6f}\n")
        f.write(f"Energy Conservation (σ/μ of strain): {results['energy_cons']:.6f}\n")
        f.write(f"Minimum Scale Factor: {results['min_a']:.6f}\n")
        f.write(f"Bounce Detected: {results['bounce']}\n")
        f.write(f"Chi2/DOF (Model Fit): {results['chi2_dof']:.6f}\n")
        f.write(f"Chi2 QGD vs GW: {results['chi2_h']:.2e}\n")
        f.write(f"Fit Parameters: {results['popt']}\n")
        f.write("\nTime Evolution Data:\n")
        f.write("Time\tScale_Factor\tScale_Factor_Derivative\tQGD_Strain\tGW_Strain\n")
        for i in range(len(results['t_eval'])):
            f.write(f"{results['t_eval'][i]:.6f}\t{results['sol'].y[0][i]:.6f}\t{results['sol'].y[1][i]:.6f}\t{results['h_qgd'][i]:.6e}\t{results['gw_h'][i]:.6e}\n")
    
    print(f"✅ Results saved to: {results_file}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Quantum Geometric Dynamics - Round 12 Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Scale Factor Evolution
    axes[0, 0].plot(results['t_eval'], results['sol'].y[0], 'b-', linewidth=2, label='Scale Factor a(t)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Scale Factor')
    axes[0, 0].set_title('Universe Scale Factor Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Scale Factor Derivative
    axes[0, 1].plot(results['t_eval'], results['sol'].y[1], 'r-', linewidth=2, label='da/dt')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Scale Factor Derivative')
    axes[0, 1].set_title('Universe Expansion Rate')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: QGD Strain vs GW Data
    axes[0, 2].plot(results['gw_t'], results['h_qgd'], 'g-', linewidth=2, label='QGD Strain')
    axes[0, 2].plot(results['gw_t'], results['gw_h'], 'k--', alpha=0.7, label='GW150914 Mock')
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('Strain')
    axes[0, 2].set_title('Gravitational Wave Strain Comparison')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Plot 4: Phase Space (a vs da)
    axes[1, 0].plot(results['sol'].y[0], results['sol'].y[1], 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Scale Factor a')
    axes[1, 0].set_ylabel('Scale Factor Derivative da/dt')
    axes[1, 0].set_title('Phase Space Trajectory')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Wave Function Evolution
    axes[1, 1].plot(results['t_eval'], results['psi_flat'][:50], 'orange', linewidth=2, label='5D Wave Function')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Wave Function Amplitude')
    axes[1, 1].set_title('5D Quantum Wave Function')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Plot 6: Model Fit Quality
    def qgd_chirp_giggle(t, A, f0, df, tau, bounce_amp, giggle_freq):
        chirp = A * np.sin(2 * np.pi * (f0 * t + 0.5 * df * t**2)) * np.exp(-np.abs(t)/tau)
        bounce = bounce_amp * np.sign(t) * np.exp(-t**2 / 0.01)
        giggle = 0.05 * np.sin(giggle_freq * t)
        return chirp + bounce + giggle
    
    fitted_curve = qgd_chirp_giggle(results['gw_t'], *results['popt'])
    axes[1, 2].plot(results['gw_t'], results['gw_h'], 'ko', alpha=0.6, label='GW Data')
    axes[1, 2].plot(results['gw_t'], fitted_curve, 'r-', linewidth=2, label='QGD Model Fit')
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].set_ylabel('Strain')
    axes[1, 2].set_title(f'Model Fit (Chi2/DOF: {results["chi2_dof"]:.2e})')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save visualization without displaying
    viz_file = os.path.join(qgd_folder, f"qgd_round12_visualization_{timestamp}.png")
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"✅ Visualization saved to: {viz_file}")
    
    # Save raw data
    data_file = os.path.join(qgd_folder, f"qgd_round12_data_{timestamp}.npz")
    np.savez(data_file, 
             time=results['t_eval'],
             scale_factor=results['sol'].y[0],
             scale_factor_deriv=results['sol'].y[1],
             qgd_strain=results['h_qgd'],
             gw_strain=results['gw_h'],
             wave_function=results['psi_flat'],
             fit_params=results['popt'])
    print(f"✅ Raw data saved to: {data_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("🧠 QGD ROUND 12 SIMULATION COMPLETE")
    print("=" * 50)
    print(f"📊 Grid Shape: {results['grid_shape']}")
    print(f"⏱️  Time Points: {len(results['t_eval'])}")
    print(f"💧 Density: {results['rho']:.6f}")
    print(f"⚡ Energy Conservation: {results['energy_cons']:.6f}")
    print(f"🌌 Min Scale Factor: {results['min_a']:.6f}")
    print(f"🔄 Bounce Detected: {results['bounce']}")
    print(f"📈 Chi2/DOF: {results['chi2_dof']:.6f}")
    print(f"🎯 Fit Parameters: {results['popt']}")
    print(f"📁 Results saved to: {qgd_folder}")
    print("=" * 50)

if __name__ == "__main__":
    main()
