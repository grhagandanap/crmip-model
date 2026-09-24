import warnings
import pandas as pd
import numpy as np
from crmip_model import CRMIPModel
warnings.filterwarnings('ignore', category=RuntimeWarning)

np.random.seed(42) # seeding random numbers

# start
file_path = r"~/CRM - AOI Beta Demo.xlsx" # specific path of the project

# Load data
df = pd.read_excel(file_path, sheet_name="Data Day") # membaca data produksi
df = df[df['Days'].notna()].reset_index(drop=True)
df = df
df = df.fillna(0)
wc = pd.read_excel(file_path, sheet_name="WC")
coords = pd.read_excel(file_path, sheet_name="Location")
well_coords_dict = dict(zip(coords["WELL"], zip(coords["X"], coords["Y"])))

# total number of producer
number_of_producer = 31

# Initialize model
model = CRMIPModel(
    days=df["Days"],
    date=df["Date"],
    inj_rates=df[df.columns[number_of_producer+2:]],
    prod_rates=df[df.columns[2:number_of_producer+2]],
    well_coords=well_coords_dict,
    wc_data = wc[wc.columns[2:number_of_producer+2]]
)

# Apply distance constraints
model.add_distance_constraints(
    constraint_type='auto',
    threshold=300,
    well_coords=well_coords_dict)

# Define tuning parameter
# There 2 available methods: SLSQP and trust-constr
stepwise_kwargs = {"optimizer_method": "SLSQP", # SLSQP dan trust-constr
                   "max_time_per_step": 500,
                   "n_jobs": -1,
                   "verbose": True}

# Run comparison models (includes Gentil WC fitting using preliminary static_noaq connectivities)
model.run_comparison_models(verbose=True,
                            stepwise_kwargs=stepwise_kwargs)

# Plot liquid production comparison: True for showing the oil production, False for only showing liquid production
model.plot_comparison_models(plot_oil=False)

# plot the connectivity map
model.plot_connectivity_over_time(
    method="dynamic_noaq",
    well_coords=well_coords_dict,
    timesteps=[0, 12, 24, 36, 48, 60, 72, 84],
    arrow_width=2.5,
    cmap='Reds',
    show_all_arrows=False
)

model.plot_connectivity_over_time(
    method="dynamic_withaq",
    well_coords=well_coords_dict,
    timesteps=[0, 12, 24, 36, 48, 60, 72, 84],
    arrow_width=2.5,
    cmap='Reds',
    show_all_arrows=False
)

# Export dynamic connectivity plots without background
model.export_comparison_params(r"~/CRM - AOI Beta Demo Result.xlsx")