from PIL import Image
import numpy as np
import os

class GeometryHandler:
    def __init__(self, opt, geom):
        self.opt = opt
        self.geom = geom
        self.material_map = geom.get("materials", None)
        
        # Check if material map is defined
        if self.material_map is None:
            raise ValueError("Material map is not defined in geometry.")
        
        # Assign unique increasing IDs to each material
        for i, (name, material) in enumerate(self.material_map.items()):
            material['id'] = i + 1
        
    def load_geometry(self, input_file: str):
        mask = self.generate_geometry_map(input_file)
        self.ny, self.nx = mask.shape
        
        # Generate grid coordinates
        self.dx = self.geom.xsize / (1 - self.nx)
        self.dy = self.geom.ysize / (1 - self.ny)
        self.nx1 = self.nx + 1
        self.ny1 = self.ny + 1
        
        # Main nodal points
        self.x = np.linspace(0, self.geom.xsize + self.dx, self.nx1)
        self.y = np.linspace(0, self.geom.ysize + self.dy, self.ny1)

        # Staggered qx and qy points
        self.xqx = self.x - self.dx / 2
        self.yqx = self.y

        self.xqy = self.x
        self.yqy = self.y - self.dy / 2

        # Generate properties for rho, Cp, k and T on main nodal points
        self.rho = self.generate_meterial_properties(mask, 'rho')
        self.Cp = self.generate_meterial_properties(mask, 'Cp')
        self.k = self.generate_meterial_properties(mask, 'k')
        self.T_init = self.generate_meterial_properties(mask, 'T')
        
    def generate_geometry_map(self, input_file: str) -> np.ndarray:
        # Open image and convert to RGB
        png_image = Image.open(input_file).convert('RGB')
        pixels = np.array(png_image)
        rgb = (m['hex'] for m in self.material_map)
        pixels = np.apply_along_axis(lambda rgb: '#{:02x}{:02x}{:02x}'.format(*rgb), 2, pixels)
        
        # Map hex colors to material IDs
        material_mask = np.zeros(pixels.shape, dtype=np.int32)
        for i, (name, material) in enumerate(self.material_map.items()):
            hex_color = material['hex']
            mask = (pixels == hex_color)
            material_mask[mask] = material['id']
        
        return material_mask
    
    def generate_meterial_properties(self, mask: np.ndarray, prop: str) -> tuple:
        # Generate material properties based on the material map
        rax = np.zeros(mask.shape)
        
        for i, (name, material) in enumerate(self.material_map.items()):
            if prop in material:
                rax[mask == material['id']] = material[prop]
            else:
                raise ValueError(f"Property '{prop}' not found for material '{name}'")
            
        # Check if any points are still zero (unassigned)
        if np.any(rax == 0):
            raise ValueError(f"Some points in the geometry are not assigned a value for property '{prop}'")
        
        # Extend rax by one row and column of ghost zeros
        rax = np.pad(rax, ((0, 1), (0, 1)), mode='constant', constant_values=np.nan)

        return rax

        
