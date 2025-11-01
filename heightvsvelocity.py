import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
heights = np.array([3.3, 4.3, 5.3, 6.3])  # cm
velocity = np.array([1.525481772, 1.712275458, 1.780766476, 2.073409918])  # mm/s
uncertainty = np.array([0.1000043266]*4)  # mm/s

# --- Weighted Linear Fit ---
# weights = 1/σ²
weights = 1 / uncertainty**2

# Perform weighted linear regression
p, cov = np.polyfit(heights, velocity, 1, w=weights, cov=True)
m, b = p  # slope and intercept
m_err, b_err = np.sqrt(np.diag(cov))

# Best-fit line
x_fit = np.linspace(min(heights)-0.2, max(heights)+0.2, 200)
y_fit = m*x_fit + b

# --- Chi-squared calculation ---
residuals = velocity - (m*heights + b)
chi_squared = np.sum((residuals / uncertainty)**2)
dof = len(heights) - 2  # degrees of freedom (N - number of fit parameters)
reduced_chi_squared = chi_squared / dof

# --- Output results ---
print(f"Slope (m) = {m:.4f} ± {m_err:.4f} mm/s/cm")
print(f"Intercept (b) = {b:.4f} ± {b_err:.4f} mm/s")
print(f"Chi-squared = {chi_squared:.3f}")
print(f"Reduced Chi-squared = {reduced_chi_squared:.3f}")

# --- Plot ---
plt.figure(figsize=(7,5))
plt.errorbar(heights, velocity, yerr=uncertainty, fmt='o', capsize=5, label='Data ± uncertainty')
plt.plot(x_fit, y_fit, 'r-', label=f'Best fit: v = {m:.3f}h + {b:.3f}')
plt.title("Velocity Increases With Respect to Initial Height", fontsize=18)
plt.xlabel("Height of Nozzle Tip Above Chip Plate (cm)", fontsize=16)
plt.ylabel("Velocity (mm/s)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()