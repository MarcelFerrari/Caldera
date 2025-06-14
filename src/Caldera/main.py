"""
Caldera: Multimaterial heat transfer solver.
https://github.com/MarcelFerrari/Caldera

File: main.py
Description: Main module for the Caldera heat transfer solver.

Author: Marcel Ferrari
Copyright (c) 2025 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import toml
import os
import pickle
import time
import numpy as np
import numba as nb
import scipy.sparse as sp

from Caldera.geometry import GeometryHandler
from Caldera.context import ContextNamespace
from Caldera.operators import compute_qx, compute_qy

class Caldera:
    def __init__(self, args: dict):
        self.args = args
        
        # Load input parameters
        self.input_file = args.get("input", "input.toml")
        self.opt, self.geom = self.load_input(self.input_file)
        
        # Set up output directory
        self.output_dir = self.opt.get("output_dir", "output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.setup_threading()
                    
    def load_input(self, input_file) -> tuple[dict, dict]:
        # Load input parameters from TOML file
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' does not exist.")
        
        with open(input_file, 'r') as f:
            data = toml.load(f)
        opt = ContextNamespace(data.get('options', {}))
        geom = ContextNamespace(data.get('geometry', {}))
        return opt, geom
    
    def setup_threading(self):
        # Set up numba threading layer
        self.threading_layer = self.opt.get("threading_layer", "workqueue")
        nb.config.THREADING_LAYER = self.threading_layer
        self.nthreads = self.opt.get("nthreads", None)
        if self.nthreads is not None:
            nb.set_num_threads(self.nthreads)
        else:
            self.nthreads = nb.get_num_threads()

        print(f"NUMBA: Threading layer set to {self.threading_layer} with {self.nthreads} threads.")

    def scale_and_solve(self, Acsc, rhs):
        # Compute row norms (for D_r) using CSR
        Acsr = Acsc.tocsr()
        Dr = row_norms_csr(Acsr.data, Acsr.indptr, self.n_rows)
        
        # Compute column norms (for D_c) using CSC
        Dc = col_norms_csc(Acsc.data, Acsc.indptr, self.n_rows)

        # Avoid division by zero
        Dr[Dr == 0] = 1.0
        Dc[Dc == 0] = 1.0

        # Build diagonal scaling matrices
        D_r_inv = sp.diags(1.0 / Dr)  # shape (n_rows, n_rows)
        D_c_inv = sp.diags(1.0 / Dc)  # shape (n_cols, n_cols)

        # Scale the matrix: A_scaled = D_r^{-1} * A * D_c^{-1}
        A_scaled = D_r_inv @ Acsc @ D_c_inv

        # Scale the RHS: rhs_scaled = D_r^{-1} * rhs
        rhs_scaled = (1.0 / Dr) * rhs

        # Solve the scaled system
        x_scaled = sp.linalg.spsolve(A_scaled, rhs_scaled)
        # x_scaled = sp.linalg.bicgstab(A_scaled, rhs_scaled, rtol=1e-12, maxiter=1000)[0]

        # Rescale the solution: x = D_c^{-1} * x_scaled
        x = (1.0 / Dc) * x_scaled

        return x
    
    def solve(self):
        print("Starting heat transfer simulation...")
        
        # Read geometry and initialize model
        print("Loading geometry...")
        handler = GeometryHandler(self.opt, self.geom)
        
        # Validate input file and generate problem geometry
        input_file = self.opt.get("geometry_input", "geometry.png")
        if not input_file.endswith('.png'):
            raise ValueError("Input file must be an PNG file.")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' does not exist.")
        handler.load_geometry(self.opt.get("geometry_input", "geometry.png"))
        print("Geometry loaded successfully.")

        # Set up geometry options
        nx1, ny1 = handler.nx1, handler.ny1
        dx, dy = handler.dx, handler.dy

        # Set up material properties
        rho = handler.rho
        Cp = handler.Cp
        rhocp = rho * Cp  # Density times specific heat capacity
        k = handler.k

        # Boundary conditions
        T_up = self.opt.T_up
        T_down = self.opt.T_down
        T_left = self.opt.T_left
        T_right = self.opt.T_right
        SIDE_ZERO_FLUX = self.opt.SIDE_ZERO_FLUX
        TOP_ZERO_FLUX = self.opt.TOP_ZERO_FLUX

        # Set up time-stepping parameters
        dt = self.opt.dt_initial
        timesum = 0.0
        max_steps = self.opt.max_steps
        max_T_change = self.opt.max_T_change
        max_reject_iter = self.opt.get('max_dt_rejection_iterations', 30)
        
        # Set up solution
        Told = handler.T_init  # Set initial temperature
        
        # Apply boudary conditions to initial temperature
        Told[:, 0] = T_left
        Told[:, -2] = T_right
        Told[0, :] = T_up
        Told[-2, :] = T_down
        Told[-1, :] = np.nan  # Bottom ghost row
        Told[:, -1] = np.nan # Right ghost column

        T = np.zeros_like(Told)  # Allocate memory for temperature field
        qx = np.zeros_like(T) # Allocate memory for qx and qy
        qy = np.zeros_like(T)

        # Compute initial heat fluxes
        qx = compute_qx(nx1, ny1, dx, k, Told, qx)
        qy = compute_qy(nx1, ny1, dy, k, Told, qy)

        # Initialize the heat flow model
        print("Dumping initial onditions...")
        with open(os.path.join(self.output_dir, 'initial_conditions.pkl'), 'wb') as f:
            pickle.dump({
                'xsize': self.geom.xsize,
                'ysize': self.geom.ysize,
                'rho': rho,
                'Cp': Cp,
                'k': k,
                'T': Told,
                'qx': qx,
                'qy' : qy,
                'dt' : dt,
                'timesum' : timesum,
            }, f)
            
        # Run the simulation
        print("Running simulation...")
        framecounter = 1
        zpad = len(str(max_steps))  # Zero padding for step numbers
        self.n_rows = nx1 * ny1  # Number of rows in the matrix
        for step in range(max_steps):            
            # Timestep bisection logic
            for reject_iter in range(max_reject_iter):
                # Placeholder for simulation step logic
                i_idx, j_idx, vals, rhs = assemble_matrix(nx1, ny1,
                                                          dx, dy, dt,
                                                          rhocp, k, Told,
                                                          T_up, T_down, T_left, T_right,
                                                          SIDE_ZERO_FLUX, TOP_ZERO_FLUX)
                
                # Assemble the matrix in COO format
                A = sp.coo_matrix((vals, (i_idx, j_idx)), shape=(self.n_rows, self.n_rows))

                # Convert to CSC
                A = A.tocsc()

                # Solve using direct solver
                # T = sp.linalg.spsolve(A, rhs)
                T = self.scale_and_solve(A, rhs)

                # Check solution residual
                residual = np.linalg.norm(A.dot(T) - rhs)/np.linalg.norm(rhs)
                print(f"Residual after solving: {residual:.4e}")
                
                # Reshape to 2D array
                T = T.reshape((ny1, nx1))
                
                # Set ghost nodes to nan]
                T[-1, :] = np.nan  # Bottom ghost row
                T[:, -1] = np.nan  # Right ghost column


                # Compute temperature change
                dT = np.abs(T - Told)
                dT_max = np.nanmax(dT)
                
                dtmax = dt*(max_T_change/dT_max)
                if dt <= dtmax: # Accept the time step
                    break
                else:
                    # Reject the time step, reduce dt
                    if reject_iter < max_reject_iter - 1:
                        print(f"Rejecting time step: dt = {dt:.4e} > dt_max = {dtmax:.4e} and dT_max = {dT_max}. Reducing dt.")
                        dt = dtmax * 0.95
                    else:   
                        print(f"Warning: Maximum rejection iterations reached ({max_reject_iter}).")
                        break

            # Update the old temperature for the next iteration
            Told[...] = T[...]

            # Compute heat fluxes
            qx = compute_qx(nx1, ny1, dx, k, T, qx)
            qy = compute_qy(nx1, ny1, dy, k, T, qy)

            # Update time
            timesum += dt
            
            # Print progress
            print(f"Step {step + 1}/{max_steps}, Time: {timesum:.2e}s, dt: {dt:.4e}s, Max dT: {dT_max:.4e}")
            # Save the current state
            if (step + 1) % self.opt.framedump_interval == 0 or step == max_steps - 1:
                with open(os.path.join(self.output_dir, f'frame_{str(framecounter).zfill(zpad)}.pkl'), 'wb') as f:
                    pickle.dump({
                        'T': T,
                        'qx': qx,
                        'qy' : qy,
                        'dt' : dt,
                        'timesum' : timesum,
                    }, f)
                framecounter += 1
            
        
            # Attempt to increase the time step
            dt = max(0.95 * dtmax, 1.25 * dt)

            if timesum >= self.opt.max_time:
                print(f"Maximum simulation time reached: {self.opt.max_time:.2f}s.")
                break

# Numba compiled functions - Not part of the class
@nb.njit(cache = True)
def idx(nx1, i, j):
    # Helper function to map 2D indices to 1D index
    # i: matrix row index (y-index)
    # j: matrix column index (x-index)
    return i*nx1 + j
    
@nb.njit(cache = True)
def insert(mat, i, j, v):
    # Mat is a tuple (i_idx, j_idx, vals)
    cur = mat[3][0]
    mat[0][cur] = i
    mat[1][cur] = j
    mat[2][cur] = v

    # Increment current index
    mat[3][0] += 1

@nb.njit(cache=True)
def assemble_matrix(nx1, ny1, dx, dy, dt, rhocp, k, T0, T_up, T_down, T_left, T_right, SIDE_ZERO_FLUX, TOP_ZERO_FLUX):
    # Assemble matrix in COO format
    n_eqs = 1               # Number of equations to solve
    n_rows = nx1*ny1*n_eqs  # Number of rows in the matrix        
    max_nnz = 8            # Maximum number of non-zero elements (~8 per row)

    # Preallocate memory for COO format
    i_idx = np.zeros((max_nnz*n_rows,), dtype=np.int32)
    j_idx = np.zeros((max_nnz*n_rows,), dtype=np.int32)
    vals = np.zeros((max_nnz*n_rows,), dtype=np.float64)
    mat = (i_idx, j_idx, vals, np.array([0], dtype=np.int32))
    b = np.zeros((n_rows,), dtype=np.float64)

    # Domain
    for i in range(ny1):
        for j in range(nx1):
            kij = idx(nx1, i, j)
            # Top boundary
            if i == 0:
                insert(mat, kij, kij, 1.0)
                if TOP_ZERO_FLUX:
                    # Zero-flux Neumann boundary condition
                    insert(mat, kij, idx(nx1, i + 1, j), -1.0)
                    b[kij] = 0.0
                else:
                    # Fixed temperature Dirichlet boundary condition
                    b[kij] = T_up
            elif i >= ny1 - 2:  # Bottom boundary or ghost row
                # Fixed temperature Dirichlet boundary condition
                insert(mat, kij, kij, 1.0)
                if i == ny1 - 2:  # Bottom boundary
                    b[kij] = T_down
                else:
                    # Bottom ghost row, no contribution to the matrix
                    b[kij] = 0.0
            elif j == 0:  # Left boundary
                # Zero-flux boundary condition
                insert(mat, kij, kij, 1.0)
                if SIDE_ZERO_FLUX:
                    # Zero-flux Neumann boundary condition
                    insert(mat, kij, idx(nx1, i, j + 1), -1.0)
                    b[kij] = 0.0 
                else:
                    # Fixed temperature Dirichlet boundary condition
                    b[kij] = T_left
            elif j == nx1 - 2:  # Right boundary
                insert(mat, kij, kij, 1.0)
                if SIDE_ZERO_FLUX:
                    # Zero-flux Neumann boundary condition
                    insert(mat, kij, idx(nx1, i, j-1), -1.0)
                    b[kij] = 0.0
                else:
                    # Fixed temperature Dirichlet boundary condition
                    b[kij] = T_right
            elif j == nx1 - 1: # Right ghost-column
                # Right ghost column, no contribution to the matrix
                insert(mat, kij, kij, 1.0)
                b[kij] = 0.0
            else: # Internal nodes
                # Harmonic mean for thermal conductivity
                kA = 2.0 * k[i, j] * k[i, j - 1] / (k[i, j] + k[i, j - 1])
                kB = 2.0 * k[i, j + 1] * k[i, j] / (k[i, j + 1] + k[i, j])
                kC = 2.0 * k[i, j] * k[i - 1, j] / (k[i, j] + k[i - 1, j])
                kD = 2.0 * k[i + 1, j] * k[i, j] / (k[i + 1, j] + k[i, j])

                T1_coeff = -kA/dx**2
                T2_coeff = -kC/dy**2
                T3_coeff = kC/dy**2 + kD/dy**2 + kA/dx**2 + kB/dx**2 + rhocp[i, j]/dt
                T4_coeff = -kD/dy**2
                T5_coeff = -kB/dx**2

                # Insert coefficients into the matrix
                insert(mat, kij, idx(nx1, i, j-1), T1_coeff)
                insert(mat, kij, idx(nx1, i-1, j), T2_coeff)
                insert(mat, kij, idx(nx1, i, j), T3_coeff)
                insert(mat, kij, idx(nx1, i+1, j), T4_coeff)
                insert(mat, kij, idx(nx1, i, j+1), T5_coeff)

                # Right-hand side vector
                b[kij] = rhocp[i, j] * T0[i, j] / dt

    return mat[0], mat[1], mat[2], b

# Matrix scaling functions
@nb.njit(cache=True)
def row_norms_csr(data, indptr, n_rows):
    norms = np.empty(n_rows)
    for i in range(n_rows):
        start = indptr[i]
        end = indptr[i+1]
        s = 0.0
        for k in range(start, end):
            s += data[k] * data[k]
        norms[i] = np.sqrt(s)
    return norms

@nb.njit(cache=True)
def col_norms_csc(data, indptr, n_cols):
    norms = np.empty(n_cols)
    for j in range(n_cols):
        start = indptr[j]
        end = indptr[j+1]
        s = 0.0
        for k in range(start, end):
            s += data[k] * data[k]
        norms[j] = np.sqrt(s)
    return norms
       
