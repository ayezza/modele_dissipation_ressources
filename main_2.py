import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# --- Simulation Parameters ---
# Feel free to tweak these to see different outcomes

# Environment
GRID_SIZE = 50
INITIAL_BAY_HEALTH = 0.9  # 0.0 (dead) to 1.0 (pristine)
HEALTH_RECOVERY_RATE = 0.005 # How fast the bay cleans itself

# Fish Population
INITIAL_FISH_COUNT = 500 # Initial number of fish in the bay
# Fish behavior
FISH_REPRODUCTION_RATE = 0.05 # Likelihood of a fish reproducing each step
MIN_HEALTH_FOR_REPRO = 0.4 # Fish won't reproduce in a very polluted bay

# Corporation / Exploitation
INITIAL_EXPLOITATION = 0.05 # Initial fishing/pollution effort (0 to 1)
PROFIT_PER_FISH = 10
COST_PER_EFFORT = 200
REINVESTMENT_RATE = 0.05 # How much of the profit is reinvested to increase exploitation

# --- The Simulation Model Class ---

class BayEcosystem:
    def __init__(self):
        """Initialize the ecosystem."""
        # The bay's health is a grid. Pollution is concentrated.
        self.bay_health = np.full((GRID_SIZE, GRID_SIZE), INITIAL_BAY_HEALTH)
        self.pollution_source = (GRID_SIZE - 1, GRID_SIZE // 2) # Factory outlet at the bottom edge

        # Place initial fish randomly
        self.fish = [{'x': random.randint(0, GRID_SIZE-1),
                      'y': random.randint(0, GRID_SIZE-1)}
                     for _ in range(INITIAL_FISH_COUNT)]

        # Corporate parameters
        self.exploitation_effort = INITIAL_EXPLOITATION
        self.profit_history = []
        self.fish_pop_history = []
        self.effort_history = []

    def _get_health_at(self, x, y):
        """Get the health at a specific location."""
        return self.bay_health[y, x]

    def step(self):
        """Advance the simulation by one time step."""
        if not self.fish: # Stop if all fish are gone
            return

        # --- 1. Corporate Actions and Environmental Impact ---
        # Corporation fishes based on its effort level
        fish_caught = int(len(self.fish) * self.exploitation_effort)
        
        # Remove caught fish (from random individuals)
        if fish_caught > 0 and len(self.fish) > fish_caught:
            random.shuffle(self.fish)
            self.fish = self.fish[fish_caught:]

        # Calculate profit for this step
        revenue = fish_caught * PROFIT_PER_FISH
        cost = self.exploitation_effort * COST_PER_EFFORT
        profit = revenue - cost
        
        # Profit Motive: Reinvest profits to increase exploitation
        if profit > 0:
            self.exploitation_effort += (profit / cost) * REINVESTMENT_RATE
            # Cap the effort to a maximum of 1.0 (100%)
            self.exploitation_effort = min(self.exploitation_effort, 1.0)
        
        # Pollution from exploitation degrades bay health
        pollution_impact = self.exploitation_effort * 0.2 # Higher effort = more pollution
        
        # Add pollution at the source and let it diffuse slightly
        y, x = self.pollution_source
        self.bay_health[max(0, y-2):y+1, max(0, x-2):min(GRID_SIZE, x+3)] -= pollution_impact
        
        # Simple diffusion model: health spreads from healthier to less healthy cells
        # This is a simplified convolution
        kernel = np.array([[0.01, 0.02, 0.01],
                           [0.02, 0.88, 0.02],
                           [0.01, 0.02, 0.01]])
        for _ in range(2): # Apply diffusion twice for a smoother effect
            padded = np.pad(self.bay_health, pad_width=1, mode='edge')
            convolved = np.zeros_like(self.bay_health)
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    convolved[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
            self.bay_health = convolved


        # --- 2. Natural Bay Processes ---
        # The bay slowly recovers its health naturally
        self.bay_health += HEALTH_RECOVERY_RATE
        # Ensure health stays within bounds [0, 1]
        self.bay_health = np.clip(self.bay_health, 0, 1)

        # --- 3. Fish Behavior ---
        new_fish = []
        surviving_fish = []
        
        for f in self.fish:
            # Fish move randomly to an adjacent cell
            f['x'] += random.randint(-1, 1)
            f['y'] += random.randint(-1, 1)
            f['x'] = np.clip(f['x'], 0, GRID_SIZE - 1)
            f['y'] = np.clip(f['y'], 0, GRID_SIZE - 1)
            
            # Check for survival: fish have a chance to die in polluted water
            local_health = self._get_health_at(f['x'], f['y'])
            if random.random() > local_health:
                continue # The fish dies

            # Reproduction: Fish reproduce based on rate and local health
            if random.random() < FISH_REPRODUCTION_RATE and local_health > MIN_HEALTH_FOR_REPRO:
                new_fish.append(f.copy())
            
            surviving_fish.append(f)
            
        self.fish = surviving_fish + new_fish

        # --- 4. Record History ---
        self.profit_history.append(profit)
        self.fish_pop_history.append(len(self.fish))
        self.effort_history.append(self.exploitation_effort)


# --- Visualization Setup ---
model = BayEcosystem()

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2)

# Main plot for the bay
ax1 = fig.add_subplot(gs[:, 0])
ax1.set_title("The Bay")
ax1.set_xticks([])
ax1.set_yticks([])
# 'viridis' is a good colormap: blue (low health/polluted) -> green -> yellow (high health/pristine)
bay_image = ax1.imshow(model.bay_health, cmap='viridis_r', vmin=0, vmax=1)
fish_scatter = ax1.scatter([], [], s=10, c='red', zorder=10)
pollution_marker = ax1.scatter(model.pollution_source[1], model.pollution_source[0], 
                               s=150, c='black', marker='X', label='Pollution Source')
ax1.legend()


# Subplot for fish population
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Fish Population")
ax2.set_xlabel("Time (steps)")
ax2.set_ylabel("Count")
ax2.set_ylim(0, INITIAL_FISH_COUNT * 2.5)
line_pop, = ax2.plot([], [], 'b-')

# Subplot for profit and exploitation
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title("Corporate Exploitation")
ax3.set_xlabel("Time (steps)")
ax3.set_ylabel("Metrics")
line_profit, = ax3.plot([], [], 'g-', label='Profit ($)')
line_effort, = ax3.plot([], [], 'r--', label='Exploitation Effort')
ax3.legend()
ax3.set_ylim(-500, 1500) # Pre-set Y-axis for profit
ax3_effort = ax3.twinx()
ax3_effort.set_ylabel("Effort Level", color='r')
ax3_effort.set_ylim(0, 1.1)
ax3_effort.tick_params(axis='y', labelcolor='r')


def update(frame):
    """The function called for each frame of the animation."""
    model.step()
    
    # Update the bay visualization
    bay_image.set_data(model.bay_health)
    if model.fish:
        fish_coords = np.array([[f['x'] for f in model.fish], [f['y'] for f in model.fish]])
        fish_scatter.set_offsets(fish_coords.T)
    else:
        fish_scatter.set_offsets([])

    # Update the graphs
    time_steps = range(len(model.fish_pop_history))
    line_pop.set_data(time_steps, model.fish_pop_history)
    line_profit.set_data(time_steps, model.profit_history)
    line_effort.set_data(time_steps, model.effort_history)

    for ax in [ax2, ax3]:
        ax.relim()
        ax.autoscale_view(scalex=True, scaley=False) # Autoscale x, but keep y fixed for better comparison

    fig.suptitle(f"Time Step: {frame}", fontsize=16)
    return bay_image, fish_scatter, line_pop, line_profit, line_effort


# Create and run the animation
ani = FuncAnimation(fig, update, frames=300, interval=100, blit=True, repeat=False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
print("Simulation complete. Check the plots for results.")
# Note: You can save the animation using ani.save('bay_simulation.mp4', writer='ffmpeg', fps=30)
# Uncomment the above line to save the animation as a video
# Make sure you have ffmpeg installed for saving animations
# You can install it via pip: pip install ffmpeg-python 