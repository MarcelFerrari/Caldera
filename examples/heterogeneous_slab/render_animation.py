import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pickle as pkl
import tqdm

output_dir = 'output'
downsample = 1
Tmin = 30.0
Tmax = 90.0

# Load initial conditions
with open(os.path.join(output_dir, 'initial_conditions.pkl'), 'rb') as f:
    ic = pkl.load(f)

# Load all .pjl frames
fnames = sorted(f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.pkl'))
fnames = fnames[::downsample]  # Downsample frames

print(f'Found {len(fnames)} frames')

def load_frame(fname):
    with open(fname, 'rb') as f:
        return pkl.load(f)

# Load all frames into a list
fnames = [os.path.join(output_dir, fname) for fname in fnames]
frames = [load_frame(f) for f in fnames]

# Initialize figure and axes
fig, axs = plt.subplots(2, 3, figsize=(15, 7.5))
plt.tight_layout()

# Load initial conditions to get the shape
ny, nx = ic['T'].shape

# Create initial imshow objects
images = [
    axs[0, 0].imshow(ic['rho'], cmap='jet'),
    axs[0, 1].imshow(ic['Cp'], cmap='jet'),
    axs[0, 2].imshow(ic['k'], cmap='jet'),
    axs[1, 0].imshow(ic['T'], cmap='jet'),
    axs[1, 1].imshow(ic['qx'], cmap='jet'),
    axs[1, 2].imshow(ic['qy'], cmap='jet')
]

# Set titles and labels
titles = ['Density', 'Specific Heat Capacity', 'Thermal Conductivity', 'Temperature', 'Heat-flow (x-direction)', 'Heat-flow (y-direction)']
for ax, title in zip(axs.flat, titles):
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

# Add colorbars
for ax, im in zip(axs.flat, images):
    fig.colorbar(im, ax=ax)

# plt.savefig('plot.pdf')
# Progress bar
pbar = tqdm.tqdm(total=len(frames), desc='Animating frames', unit='frame')

# Update function
def animate(i):
    s = frames[i]

    # Update images
    images[3].set_data(s['T'])
    images[4].set_data(s['qx'])
    images[5].set_data(s['qy'])

    # Dynamically rescale color limits
    # images[3].set_clim(vmin=np.nanmin(s['T']), vmax=np.nanmax(s['T']))
    images[3].set_clim(vmin=Tmin, vmax=Tmax)
    images[4].set_clim(vmin=np.nanmin(s['qx']), vmax=np.nanmax(s['qx']))
    images[5].set_clim(vmin=np.nanmin(s['qy']), vmax=np.nanmax(s['qy']))

    # Update progress bar
    pbar.update(1)
    pbar.set_postfix(frame=i)

# Create and save animation
ani = anim.FuncAnimation(fig, animate, frames=len(frames), interval=100)
ani.save('animation.mp4', writer='ffmpeg', fps=5) 
pbar.close()