import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle as pkl
import tqdm

output_dir = 'output'
downsample = 1

# Load initial conditions
with open(os.path.join(output_dir, 'initial_conditions.pkl'), 'rb') as f:
    ic = pkl.load(f)

# Gather & downsample frame filenames
fnames = sorted(f for f in os.listdir(output_dir)
                if f.startswith('frame_') and f.endswith('.pkl'))
fnames = fnames[::downsample]
print(f'Found {len(fnames)} frames')

def load_frame(fname):
    with open(os.path.join(output_dir, fname), 'rb') as f:
        return pkl.load(f)

frames = [load_frame(fname) for fname in fnames]

# ----------------------------------------------------------------------------
# Helper to format a time (in seconds) as
#   - "Xd Xh Xm Xs.s" if ≥1 s (1 decimal place on seconds)
#   - "1.23e-02 s" style if <1 s (2 decimal places)
def format_time(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds:.2e} s"
    days = int(seconds // 86400)
    seconds %= 86400
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    return f"{days}d {hours}h {minutes}m {seconds:.1f}s"
# ----------------------------------------------------------------------------

# Make figure + axes
fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))

# Initial imshow objects
images = [
    axs[0, 0].imshow(ic['rho'], cmap='jet'),
    axs[0, 1].imshow(ic['Cp'],  cmap='jet'),
    axs[0, 2].imshow(ic['k'],   cmap='jet'),
    axs[1, 0].imshow(ic['T'],   cmap='jet'),
    axs[1, 1].imshow(ic['qx'],  cmap='jet'),
    axs[1, 2].imshow(ic['qy'],  cmap='jet')
]

titles = [
    'Density', 'Specific Heat Capacity', 'Thermal Conductivity',
    'Temperature', 'Heat-flow (x)', 'Heat-flow (y)'
]
for ax, title in zip(axs.flat, titles):
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

# Colorbars for each panel
for ax, im in zip(axs.flat, images):
    fig.colorbar(im, ax=ax)

# Fix temperature color limits to the IC range
Tmin, Tmax = np.nanmin(ic['T']), np.nanmax(ic['T'])
images[3].set_clim(vmin=Tmin, vmax=Tmax)

# Add a text object for the timestamp (top‐center)
time_text = fig.text(0.5, 0.98, '', ha='center', va='top', fontsize=12)

# Tight layout but leave space at top for the timestamp
fig.tight_layout(rect=[0, 0, 1, 0.95])

# Progress bar
pbar = tqdm.tqdm(total=len(frames), desc='Animating frames', unit='frame')

def animate(i):
    s = frames[i]
    # Update core fields
    images[3].set_data(s['T'])
    images[4].set_data(s['qx'])
    images[5].set_data(s['qy'])
    # Dynamically rescale heat‐flux colorbars
    images[4].set_clim(vmin=np.nanmin(s['qx']), vmax=np.nanmax(s['qx']))
    images[5].set_clim(vmin=np.nanmin(s['qy']), vmax=np.nanmax(s['qy']))
    # Update timestamp
    t_str  = format_time(s['timesum'])
    dt_str = format_time(s['dt'])
    time_text.set_text(f"Time: {t_str}   Δt: {dt_str}")
    # Advance progress bar
    pbar.update(1)
    pbar.set_postfix(frame=i)
    # Return artists (not strictly needed if blit=False)
    return (*images, time_text)

ani = FuncAnimation(
    fig, animate,
    frames=len(frames),
    interval=100,
    blit=False
)

# Save with bbox_inches='tight' so nothing is clipped
ani.save(
    'animation.mp4',
    writer='ffmpeg',
    fps=5,
    savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1}
)

pbar.close()
