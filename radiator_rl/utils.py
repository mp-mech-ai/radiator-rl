import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd

def compare_models(
    model1_data: dict,
    model2_data: dict,
    dt: int,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    owner_schedule: list[tuple] = None,
    steps_per_day: int = 144,
    show: bool = True
):
    """Compare two models with professional styling."""
    
    # Professional color scheme
    colors = {
        'model1': '#2E86AB',        # Professional blue
        'model2': '#F18F01',        # Warm orange
        'T_out': '#6C757D',         # Gray
        'target': '#06A77D',        # Teal
        'owner': '#A23B72',         # Muted purple
        'grid': '#E9ECEF',          # Light grid
        'text': '#2C3E50'           # Dark text
    }
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Compute cumulative values
    energy1_cumsum = np.cumsum(model1_data['energy_consumed'])
    energy2_cumsum = np.cumsum(model2_data['energy_consumed'])
    cost1_cumsum = np.cumsum(model1_data['energy_cost'])
    cost2_cumsum = np.cumsum(model2_data['energy_cost'])
    reward1_cumsum = np.cumsum(model1_data['reward'])
    reward2_cumsum = np.cumsum(model2_data['reward'])
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    ax = [[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
          [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]]
    
    # Set background color
    fig.patch.set_facecolor('white')
    for row in ax:
        for axis in row:
            axis.set_facecolor('#FAFAFA')
            axis.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color=colors['grid'])
    
    # --- Temperature Plot ---
    # Owner presence periods first (so they're in background)
    if owner_schedule:
        min_step = min(model1_data['time'])
        max_step = max(model1_data['time'])
        
        presence_periods = []
        for step in range(min_step, max_step + 1):
            step_in_day = step % steps_per_day
            is_present = any(start <= step_in_day < end for start, end in owner_schedule)
            
            if is_present:
                if not presence_periods or presence_periods[-1][1] != step - 1:
                    presence_periods.append([step, step])
                else:
                    presence_periods[-1][1] = step
        
        for i, (start, end) in enumerate(presence_periods):
            ax[0][0].axvspan(start, end + 1, color=colors['owner'], alpha=0.08, 
                           label='Owner Present' if i == 0 else '', zorder=0)
    
    # Temperature lines
    ax[0][0].plot(model1_data['time'], model1_data['T_out'], 
                  label='T_out', color=colors['T_out'], linestyle='--', 
                  linewidth=2, alpha=0.7, zorder=1)
    ax[0][0].plot(model1_data['time'], model1_data['T_in'], 
                  label=f'T_in {model1_name}', color=colors['model1'], 
                  linewidth=2.5, zorder=2)
    ax[0][0].plot(model2_data['time'], model2_data['T_in'], 
                  label=f'T_in {model2_name}', color=colors['model2'], 
                  linewidth=2.5, zorder=2)
    
    # Target range
    ax[0][0].axhspan(20, 22, color=colors['target'], alpha=0.15, 
                     label='Target Range', zorder=0)
    
    ax[0][0].set_xlabel(f'Time (every {dt/60:.0f} min)', fontsize=11, color=colors['text'])
    ax[0][0].set_ylabel('Temperature (°C)', fontsize=11, color=colors['text'])
    ax[0][0].legend(loc='best', framealpha=0.95, fontsize=10)
    ax[0][0].set_xlim([min(model1_data['time']), max(model1_data['time'])])
    ax[0][0].set_title('Temperature Comparison', fontsize=13, fontweight='bold', 
                       color=colors['text'], pad=15)
    ax[0][0].tick_params(colors=colors['text'])
    
    # --- Energy Consumption Plot ---
    ax[0][1].plot(model1_data['time'], energy1_cumsum, 
                  label=f'{model1_name}', color=colors['model1'], linewidth=2.5)
    ax[0][1].plot(model2_data['time'], energy2_cumsum, 
                  label=f'{model2_name}', color=colors['model2'], linewidth=2.5)
    ax[0][1].set_xlabel(f'Time (every {dt/60:.0f} min)', fontsize=11, color=colors['text'])
    ax[0][1].set_ylabel('Cumulative Energy (kWh)', fontsize=11, color=colors['text'])
    ax[0][1].legend(loc='lower right', framealpha=0.95, fontsize=10)
    ax[0][1].set_title('Energy Consumption Comparison', fontsize=13, fontweight='bold', 
                       color=colors['text'], pad=15)
    ax[0][1].tick_params(colors=colors['text'])
    
    # Statistics box
    final_energy1 = energy1_cumsum[-1]
    final_energy2 = energy2_cumsum[-1]
    savings_pct = ((final_energy1 - final_energy2) / final_energy1 * 100) if final_energy1 > 0 else 0
    
    box_props = dict(boxstyle='round,pad=0.6', facecolor='white', 
                     edgecolor=colors['grid'], alpha=0.95, linewidth=1.5)
    text_str = f'{model1_name}: {final_energy1:.2f} kWh\n{model2_name}: {final_energy2:.2f} kWh\nSavings: {savings_pct:.1f}%'
    ax[0][1].text(0.03, 0.97, text_str, transform=ax[0][1].transAxes, 
                  fontsize=10, verticalalignment='top', bbox=box_props, 
                  color=colors['text'])
    
    # --- Reward Plot ---
    ax[1][0].plot(model1_data['time'], reward1_cumsum, 
                  label=f'{model1_name}', color=colors['model1'], linewidth=2.5)
    ax[1][0].plot(model2_data['time'], reward2_cumsum, 
                  label=f'{model2_name}', color=colors['model2'], linewidth=2.5)
    ax[1][0].set_xlabel(f'Time (every {dt/60:.0f} min)', fontsize=11, color=colors['text'])
    ax[1][0].set_ylabel('Cumulative Reward', fontsize=11, color=colors['text'])
    ax[1][0].legend(loc='best', framealpha=0.95, fontsize=10)
    ax[1][0].set_title('Reward Comparison', fontsize=13, fontweight='bold', 
                       color=colors['text'], pad=15)
    ax[1][0].tick_params(colors=colors['text'])
    
    # Statistics box
    final_reward1 = reward1_cumsum[-1]
    final_reward2 = reward2_cumsum[-1]
    text_str = f'{model1_name}: {final_reward1:.2f}\n{model2_name}: {final_reward2:.2f}'
    ax[1][0].text(0.03, 0.05, text_str, transform=ax[1][0].transAxes, 
                  fontsize=10, verticalalignment='bottom', bbox=box_props, 
                  color=colors['text'])
    
    # --- Energy Cost Plot ---
    ax[1][1].plot(model1_data['time'], cost1_cumsum, 
                  label=f'{model1_name}', color=colors['model1'], linewidth=2.5)
    ax[1][1].plot(model2_data['time'], cost2_cumsum, 
                  label=f'{model2_name}', color=colors['model2'], linewidth=2.5)
    ax[1][1].set_xlabel(f'Time (every {dt/60:.0f} min)', fontsize=11, color=colors['text'])
    ax[1][1].set_ylabel('Cumulative Cost (€)', fontsize=11, color=colors['text'])
    ax[1][1].legend(loc='lower right', framealpha=0.95, fontsize=10)
    ax[1][1].set_title('Energy Cost Comparison', fontsize=13, fontweight='bold', 
                       color=colors['text'], pad=15)
    ax[1][1].tick_params(colors=colors['text'])
    
    # Statistics box
    final_cost1 = cost1_cumsum[-1]
    final_cost2 = cost2_cumsum[-1]
    cost_savings_pct = ((final_cost1 - final_cost2) / final_cost1 * 100) if final_cost1 > 0 else 0
    text_str = f'{model1_name}: €{final_cost1:.2f}\n{model2_name}: €{final_cost2:.2f}\nSavings: {cost_savings_pct:.1f}%'
    ax[1][1].text(0.03, 0.97, text_str, transform=ax[1][1].transAxes, 
                  fontsize=10, verticalalignment='top', bbox=box_props, 
                  color=colors['text'])
    
    if show:
        plt.show()
    
    return fig, ax


def get_T_measurement(path, num_workers, data_index=None):
    """Load temperature measurements from CSV file."""
    df = pd.read_csv(path, index_col=False)
    dt = int((datetime.strptime(df["Date"][1], "%Y-%m-%d %H:%M:%S") - 
                datetime.strptime(df["Date"][0], "%Y-%m-%d %H:%M:%S")).total_seconds())
    
    step_per_day = 24*3600 // dt

    start_time = df.iloc[0, 0]
    if data_index is None:
        rand_ind = np.sort(np.random.choice(np.arange(0, len(df) - step_per_day), 
                                        size=num_workers, replace=False))
        
        T_out_measurement = []
        for i, ind in enumerate(rand_ind):
            T_out_measurement.append(list(df.iloc[ind:ind+step_per_day, 1]))
    else:
        T_out_measurement = [list(df.iloc[step_per_day*data_index:step_per_day*data_index+step_per_day, 1])]
    
    return T_out_measurement, dt, start_time