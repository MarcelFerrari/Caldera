## Getting Started

This is a quickstart guide to get the solver up and running.

### Cloning the Repo and Installing Dependencies

The first step is to clone the GitHub repo and install the necessary dependencies.

This snippet will clone the repo, create a virtualenvironment and install the necessary python dependencies.

```bash
git clone https://github.com/MarcelFerrari/Caldera.git
cd Caldera
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Setting up the environment

Now we can set up the environment.

```bash
export PATH=$PATH:$(realpath bin)
```

This should allow you to run the following command:

```bash
caldera --help
```

If this does not work, consider making the launcher in the `/bin` folder executable:

```bash
chmod +x bin/caldera
```

### Runing the example

Navigate to `examples/copper_pan`:

```bash
cd examples/copper_pan
```

Run the solver:

```bash
caldera
```

You should see an output that looks like this:

```bash
$ caldera

   (          (   (                     
   )\      )  )\  )\ )   (   (       )  
 (((_)  ( /( ((_)(()/(  ))\  )(   ( /(  
 )\___  )(_)) _   ((_))/((_)(()\  )(_)) 
((/ __|((_)_ | |  _| |(_))   ((_)((_)_  
 | (__ / _` || |/ _` |/ -_) | '_|/ _` | 
  \___|\__,_||_|\__,_|\___| |_|  \__,_|   (v0.1)

NUMBA: Threading layer set to workqueue with 4 threads.
Starting heat transfer simulation...
Loading geometry...
Geometry loaded successfully.
Dumping initial onditions...
Running simulation...
Residual after solving: 1.3445e-16
Rejecting time step: dt = 1.0000e-03 > dt_max = 1.7096e-04 and dT_max = 29.24592410974403. Reducing dt.
Residual after solving: 1.8487e-16
Rejecting time step: dt = 1.6242e-04 > dt_max = 6.9344e-05 and dT_max = 11.710830912758496. Reducing dt.
Residual after solving: 1.2593e-16
Rejecting time step: dt = 6.5877e-05 > dt_max = 5.5352e-05 and dT_max = 5.950770023913201. Reducing dt.
Residual after solving: 2.3270e-16
Step 1/2000, Time: 5.26e-05s, dt: 5.2584e-05s, Max dT: 4.9309e+00
Residual after solving: 1.1353e-16
Rejecting time step: dt = 6.5730e-05 > dt_max = 6.5147e-05 and dT_max = 5.0447432562289904. Reducing dt.
Residual after solving: 1.2333e-16
Step 2/2000, Time: 1.14e-04s, dt: 6.1890e-05s, Max dT: 4.7987e+00
Residual after solving: 1.7033e-16
Step 3/2000, Time: 1.92e-04s, dt: 7.7362e-05s, Max dT: 4.8155e+00
Residual after solving: 1.0536e-16
Step 4/2000, Time: 2.89e-04s, dt: 9.6703e-05s, Max dT: 4.6894e+00
Residual after solving: 2.1881e-16
Step 5/2000, Time: 4.09e-04s, dt: 1.2088e-04s, Max dT: 4.4367e+00
Residual after solving: 1.0671e-16
Step 6/2000, Time: 5.61e-04s, dt: 1.5110e-04s, Max dT: 4.0906e+00
Residual after solving: 2.1377e-16
Step 7/2000, Time: 7.49e-04s, dt: 1.8887e-04s, Max dT: 3.6928e+00
...
```

# Input File

Caldera needs two input files: a text-based toml input file named `input.toml` and a geometry file `geometry.png`.

This example input.toml file contains all the options:

```toml
# Solver options
[options]
output_dir = "output"                 # Directory where results will be written

# Threading options
nthreads = 4                          # Number of CPU threads to use
threading_layer = "workqueue"         # Threading backend: "workqueue", "omp" or "tbb"
geometry_input = "geometry.png"       # Input geometry file (PNG format)

# Timestepping control
dt_initial = 0.001                    # Initial time step (seconds)
max_T_change = 5.0                    # Maximum temperature change per step (Kelvin)
max_steps = 2000                      # Maximum number of simulation steps
max_time = 10000.0                    # Maximum simulation time (seconds)
max_dt_rejection_iterations = 30      # Max iterations for time step bisection
framedump_interval = 1                # Interval for writing output frames

# Boundary conditions
T_up = 30.0                          # Upper boundary temperature (Kelvin/C)
T_down = 90.0                        # Lower boundary temperature (Kelvin/C)
T_left = 30.0                        # Left boundary temperature (Kelvin/C)
T_right = 30.0                       # Right boundary temperature (Kelvin/C)

SIDE_ZERO_FLUX = false               # Enable zero-flux boundary conditions on the
                                     # side boundaries instead of fixed temperatures
TOP_ZERO_FLUX = false                # Enable zero-flux boundary conditions on the
                                     # top boundary instead of fixed temperature

# Geometry parameters
[geometry]
xsize = 0.1                           # Physical size in X (meters)
ysize = 0.06                          # Physical size in Y (meters)

# Material properties
[geometry.materials]
"copper" = {hex = "#1a1a1a", rho = 8960.0, Cp = 385.0, k = 400.0, T = 30.0}
"water" = {hex = "#0000ff", rho = 1000.0, Cp = 4200.0, k = 0.65, T = 30.0} 

# Artificially high Cp to simulate advective cooling
"air" = {hex = "#00ffff", rho = 1.0, Cp = 1e9, k = 0.026, T = 30.0}
```

The `[geometry.materials]` section specifies the material properties of the system. Each material needs to have a unique rgb hex code, as well as density, specific heat, thermal conductiviy and initial temperature value.

During generation of the geometry, the hex code will be matched to the input `geometry.png` file in order to generate the material properties of the problem.

In order to generate a `geometry.png` input file, it is recommended to draw an image using Inkscape. Then, it is possible to export the image as PNG with a given resolution. Make sure to match the document size with the physical size of the system in the input file. The export resolution will be the resolution at which the simulation is carried out.

NOTE: when exporting a PNG file from Inkscape, it is important to set the following options:
```bash
Interlacing:  unchecked
Compression:  0 - No Compression
pHYs DPI:     0.0
Antialias:    0
```
This can be done by clicking the settings cog right next to the export format in the export tab.