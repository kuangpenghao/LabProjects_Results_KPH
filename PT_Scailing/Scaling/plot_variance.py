import os
import matplotlib.pyplot as plt
import numpy as np

# Data extracted from user prompt
data = [
    {
        "dim_z": 256,
        "model.unary_factors.weight": 3.6093e+00,
        "model.iterator.head_selection.ternary_factor_u": 8.5127e-02,
        "model.iterator.head_selection.ternary_factor_v": 8.8690e-02,
        "model.iterator.topic_modeling.binary_factor": 8.9284e-02,
        "cls.predictions.bias": 1.1138e+00,
        "cls.predictions.transform.dense.weight": 8.3463e-02,
        "cls.predictions.transform.dense.bias": 1.2450e+00,
        "cls.predictions.transform.LayerNorm.weight": 7.4628e+00,
        "cls.predictions.transform.LayerNorm.bias": 3.8037e+00,
        "cls.predictions.decoder.weight": 4.7282e-02,
    },
    {
        "dim_z": 512,
        "model.unary_factors.weight": 3.0113e+00,
        "model.iterator.head_selection.ternary_factor_u": 5.4982e-02,
        "model.iterator.head_selection.ternary_factor_v": 5.5261e-02,
        "model.iterator.topic_modeling.binary_factor": 5.1364e-02,
        "cls.predictions.bias": 8.6258e-01,
        "cls.predictions.transform.dense.weight": 5.1486e-02,
        "cls.predictions.transform.dense.bias": 8.8711e-01,
        "cls.predictions.transform.LayerNorm.weight": 6.4628e+00,
        "cls.predictions.transform.LayerNorm.bias": 3.0659e+00,
        "cls.predictions.decoder.weight": 2.3585e-02,
    },
    {
        "dim_z": 1024,
        "model.unary_factors.weight": 2.5641e+00,
        "model.iterator.head_selection.ternary_factor_u": 3.6140e-02,
        "model.iterator.head_selection.ternary_factor_v": 3.6073e-02,
        "model.iterator.topic_modeling.binary_factor": 3.2546e-02,
        "cls.predictions.bias": 7.4276e-01,
        "cls.predictions.transform.dense.weight": 3.3861e-02,
        "cls.predictions.transform.dense.bias": 5.9532e-01,
        "cls.predictions.transform.LayerNorm.weight": 6.0595e+00,
        "cls.predictions.transform.LayerNorm.bias": 2.7104e+00,
        "cls.predictions.decoder.weight": 1.1889e-02,
    },
    {
        "dim_z": 1536,
        "model.unary_factors.weight": 2.3218e+00,
        "model.iterator.head_selection.ternary_factor_u": 2.8445e-02,
        "model.iterator.head_selection.ternary_factor_v": 2.8292e-02,
        "model.iterator.topic_modeling.binary_factor": 2.5876e-02,
        "cls.predictions.bias": 6.8076e-01,
        "cls.predictions.transform.dense.weight": 2.6988e-02,
        "cls.predictions.transform.dense.bias": 6.1663e-01,
        "cls.predictions.transform.LayerNorm.weight": 6.7196e+00,
        "cls.predictions.transform.LayerNorm.bias": 2.4974e+00,
        "cls.predictions.decoder.weight": 7.8319e-03,
    },
    {
        "dim_z": 2560,
        "model.unary_factors.weight": 2.0352e+00,
        "model.iterator.head_selection.ternary_factor_u": 2.1158e-02,
        "model.iterator.head_selection.ternary_factor_v": 2.1137e-02,
        "model.iterator.topic_modeling.binary_factor": 1.9828e-02,
        "cls.predictions.bias": 6.4517e-01,
        "cls.predictions.transform.dense.weight": 2.0475e-02,
        "cls.predictions.transform.dense.bias": 4.0720e-01,
        "cls.predictions.transform.LayerNorm.weight": 6.4510e+00,
        "cls.predictions.transform.LayerNorm.bias": 2.0357e+00,
        "cls.predictions.decoder.weight": 4.8195e-03,
    },
    {
        "dim_z": 3840,
        "model.unary_factors.weight": 1.7958e+00,
        "model.iterator.head_selection.ternary_factor_u": 1.6880e-02,
        "model.iterator.head_selection.ternary_factor_v": 1.6838e-02,
        "model.iterator.topic_modeling.binary_factor": 1.6162e-02,
        "cls.predictions.bias": 6.0669e-01,
        "cls.predictions.transform.dense.weight": 1.6512e-02,
        "cls.predictions.transform.dense.bias": 4.3202e-01,
        "cls.predictions.transform.LayerNorm.weight": 7.3879e+00,
        "cls.predictions.transform.LayerNorm.bias": 1.8971e+00,
        "cls.predictions.decoder.weight": 3.0659e-03,
    }
]

params = [
    "model.unary_factors.weight",
    "model.iterator.head_selection.ternary_factor_u",
    "model.iterator.head_selection.ternary_factor_v",
    "model.iterator.topic_modeling.binary_factor",
    "cls.predictions.bias",
    "cls.predictions.transform.dense.weight",
    "cls.predictions.transform.dense.bias",
    "cls.predictions.transform.LayerNorm.weight",
    "cls.predictions.transform.LayerNorm.bias",
    "cls.predictions.decoder.weight"
]

dim_zs = [d["dim_z"] for d in data]

# Use standard deviation directly
stds = {param: [d[param] for d in data] for param in params}

base_dir = "parameter_stds"
os.makedirs(base_dir, exist_ok=True)

folder_name = os.path.join(base_dir, "logX_logY")
os.makedirs(folder_name, exist_ok=True)

log_dim_zs = np.log10(dim_zs)

for param in params:
    plt.figure(figsize=(8, 6))
    
    # Original Data Points
    plt.plot(dim_zs, stds[param], marker='o', linestyle='', label='Data')
    
    # Linear Fit in log-log space
    log_stds = np.log10(stds[param])
    slope, intercept = np.polyfit(log_dim_zs, log_stds, 1)
    
    # R-squared calculation
    predictions = slope * log_dim_zs + intercept
    ss_res = np.sum((log_stds - predictions) ** 2)
    ss_tot = np.sum((log_stds - np.mean(log_stds)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Generate line points
    fit_stds = 10 ** (slope * log_dim_zs + intercept)
    plt.plot(dim_zs, fit_stds, color='red', linestyle='--', label=f'Fit: slope={slope:.4f}, $R^2$={r_squared:.4f}')
    
    plt.xscale('log')
    plt.yscale('log')
    
    # Set explicit tick labels for x-axis
    plt.xticks(dim_zs, [str(d) for d in dim_zs])
    
    plt.xlabel("dim_z")
    plt.ylabel(f"Std of {param}")
    plt.title(f"Std vs dim_z for {param}\n(log-log with linear fit)")
    '''
    # Add text annotation for R^2 and slope on the plot
    plt.text(0.05, 0.90, f'$R^2 = {r_squared:.4f}$\nSlope $= {slope:.4f}$', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    '''         
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    filename = f"{param}.png"
    plt.savefig(os.path.join(folder_name, filename))
    plt.close()

print(f"Files saved in {folder_name}")
