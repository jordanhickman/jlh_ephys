from ipywidgets import interactive, IntSlider, fixed, HBox, VBox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

from ipywidgets import Button
from IPython.display import display


class CCF_tools:
    def __init__(self, analysis_obj):
        self.analysis = analysis_obj
        if self.analysis.processed == True:
            self.trials = self.analysis.trials
            self.units = self.analysis.units
            self.probes = self.analysis.probes
            self.parameters = self.analysis.parameters
            self.path = self.analysis.path
            self.mouse = self.analysis.mouse
            self.date = self.analysis.date
            self.raw = self.analysis.raw
        
        
    def load_allen_tools(self):
        # Initialize the MouseConnectivityCache with 10um resolution
        self.mcc = MouseConnectivityCache(resolution=10)

        # Fetch the 10um annotation volume
        self.annotation_volume_10um, _ = self.mcc.get_annotation_volume()

        # Fetch the structure tree to map IDs to acronyms
        self.structure_tree = self.mcc.get_structure_tree()

        # Generate ids:brainregion pairs
        self.id_to_brainreg = {node['id']: node['name'] for node in self.structure_tree.nodes()}
    
    def get_regions_from_ccfs(self, ccf_coords_array, scale_factor=0.1):
        ccf_coords_array = (ccf_coords_array * scale_factor).astype(int)
        
        # Check for out-of-bounds coordinates
        in_bounds = np.all((ccf_coords_array >= 0) & (ccf_coords_array < np.array(self.annotation_volume_10um.shape)), axis=1)
        
        # Initialize results array
        results = np.array(["Coordinates out of bounds"] * ccf_coords_array.shape[0], dtype=object)
        
        # Extract region_ids for in-bounds coordinates
        in_bounds_coords = ccf_coords_array[in_bounds]
        region_ids = self.annotation_volume_10um[in_bounds_coords[:, 0], in_bounds_coords[:, 1], in_bounds_coords[:, 2]]
        
        # Map region_ids to names
        region_names = np.array([self.id_to_brainreg.get(region_id, "Unknown") for region_id in region_ids])
        
        # Update results array with region names for in-bounds coordinates
        results[in_bounds] = region_names
        
        return results
    def adjust_probe_depth(self, coords, target_length):
        """
        Adjusts the probe depth by either trimming or extending the coordinates.
        """
        current_length = len(coords)
        
        if target_length == current_length:
            return coords
        
        if target_length < current_length:
            return coords[:target_length]
        
        # Calculate the distance between each pair of adjacent coordinates
        diffs = np.diff(coords, axis=0)
        
        # Calculate the average difference
        avg_diff = np.mean(diffs, axis=0)
        
        # Extend the coordinates
        new_coords = [coords[-1]]
        while len(coords) + len(new_coords) < target_length:
            new_coords.append(new_coords[-1] + avg_diff)
        
        return np.vstack([coords, np.array(new_coords)])
    
    def apply_3d_transformations(self, full_coords, dAP=0, dML=0, rotation_ap=0, rotation_ml=0):
        """
        Apply 3D transformations and rotations around AP and ML axes to coordinates.
        
        Parameters:
            full_coords (numpy.ndarray): An array of shape (n, 3) for n coordinates in 3D (AP, DV, ML).
            dAP (float): Change in AP (translation).
            dML (float): Change in ML (translation).
            rotation_ap (float): Rotation around the AP axis (in degrees).
            rotation_ml (float): Rotation around the ML axis (in degrees).
            
        Returns:
            numpy.ndarray: Transformed coordinates with all 3 dimensions (AP, DV, ML).
        """
        
        # Translate
        translated_coords = full_coords + np.array([dAP, 0, dML])
        
        # Rotate around AP axis
        angle_ap_rad = np.radians(rotation_ap)
        rotation_matrix_ap = np.array([
            [1, 0, 0],
            [0, np.cos(angle_ap_rad), -np.sin(angle_ap_rad)],
            [0, np.sin(angle_ap_rad), np.cos(angle_ap_rad)]
        ])
        
        # Rotate around ML axis
        angle_ml_rad = np.radians(rotation_ml)
        rotation_matrix_ml = np.array([
            [np.cos(angle_ml_rad), 0, np.sin(angle_ml_rad)],
            [0, 1, 0],
            [-np.sin(angle_ml_rad), 0, np.cos(angle_ml_rad)]
        ])
        
        # Composite rotation
        rotation_matrix_composite = np.dot(rotation_matrix_ap, rotation_matrix_ml)
        
        # Apply rotation
        rotated_coords = np.dot(translated_coords, rotation_matrix_composite.T)
        
        return rotated_coords


    def plot_transformed_coords_multiple_probes(self, dAP, dML, rotation_ap, rotation_ml, custom_coords_dict, probeA_depth, probeB_depth, probeC_depth):
        global transformed_coords 
        plt.figure(figsize=(20, 8))
        
        for idx, (probe, custom_coords) in enumerate(custom_coords_dict.items()):
            # Adjust depth
            target_length = locals().get(f"{probe}_depth")
            adjusted_coords = self.adjust_probe_depth(custom_coords, target_length)
            
            scaled_coords = adjusted_coords / 10
            transformed_coords = self.apply_3d_transformations(scaled_coords, dAP, dML, rotation_ap, rotation_ml)
            brain_regions = self.get_regions_from_ccfs(transformed_coords, scale_factor=1)
            
            plt.subplot(1, len(custom_coords_dict), idx + 1)
            
            unique_regions, unique_ids = np.unique(brain_regions, return_inverse=True)
            plt.scatter(transformed_coords[:, 2], transformed_coords[:, 0], c=unique_ids, cmap='viridis')
            plt.xlim([min(transformed_coords[:, 2]) - 10, max(transformed_coords[:, 2]) + 10])
            plt.ylim([min(transformed_coords[:, 0]) - 10, max(transformed_coords[:, 0]) + 10])
            
            if idx == 0 or idx == 2:
                plt.gca().invert_yaxis()
            
            for i, region in enumerate(brain_regions):
                if i % 10 == 0:
                    plt.text(transformed_coords[i, 2], transformed_coords[i, 0], region, fontsize=12)
            
            plt.xlabel('ML coordinate')
            plt.ylabel('AP coordinate')
            plt.title(f'Transformed Coordinates in Brain Regions ({probe})')
            plt.grid(True)
            
        plt.tight_layout()
        plt.show()

    def save_coordinates(self, button):
        # Access the transformed_coords variable (make it global or pass in another way)
        global transformed_coords
    
        # Save the coordinates
        np.savetxt(os.path.join(self.path,'transformed_coordinates.csv'), transformed_coords, delimiter=',', header='AP, DV, ML')
        print("Coordinates saved.")

    def call_probe_plotter(self, coords_dict):
        save_button = Button(description="Save Coordinates")
        # Attach the save function to the button's click event
        save_button.on_click(self.save_coordinates)

        dAP_slider = IntSlider(min=-50, max=50, step=1, description='dAP:')
        dML_slider = IntSlider(min=-50, max=50, step=1, description='dML:')
        rotation_ap_slider = IntSlider(min=-15, max=15, step=1, description='Rotation AP:')
        rotation_ml_slider = IntSlider(min=-15, max=15, step=1, description='Rotation ML:')

        # Depth sliders for each probe
        probeA_depth_slider = IntSlider(min=1, max=600, step=1, value=len(coords_dict['probeA']), description='Probe A Depth:')
        probeB_depth_slider = IntSlider(min=1, max=600, step=1, value=len(coords_dict['probeB']), description='Probe B Depth:')
        probeC_depth_slider = IntSlider(min=1, max=600, step=1, value=len(coords_dict['probeC']), description='Probe C Depth:')

        # Create the interactive plot
        interactive_plot = interactive(
            self.plot_transformed_coords_multiple_probes, 
            dAP=(-50, 50, 1), 
            dML=(-50, 50, 1), 
            rotation_ap=(-15, 15, 1), 
            rotation_ml=(-15, 15, 1), 
            custom_coords_dict=coords_dict,
            probeA_depth=probeA_depth_slider,
            probeB_depth=probeB_depth_slider,
            probeC_depth=probeC_depth_slider
        )

        # Create a layout for the sliders and the plot
        layout = VBox([interactive_plot, save_button])

        layout

    def compare_brain_regions(list1, list2):
        discrepancies = []
        for i, (region1, region2) in enumerate(zip(list1, list2)):
            region1_cleaned = region1.replace(',', '')
            region2_cleaned = region2.replace(',', '')
            if region1_cleaned != region2_cleaned:
                discrepancies.append((i, region1_cleaned, region2_cleaned))
        return discrepancies
    
    def calculate_planned_NPcoords(self, entry, tip, 
                               x_bias=[-16, -8, 8, 16], 
                               y_bias=[30, 10, 30, 10], 
                               sites_distance=[40, 40, 40, 40], 
                               per_max_sites=[240, 240, 240, 240],
                               tip_length=175):
        """
        Calculate the Neuropixel coordinates from the tip to the entry point.

        Parameters:
        - entry, tip: Coordinates for the entry and tip of the probe.
        - x_bias, y_bias: Horizontal and vertical spacing between the columns.
        - sites_distance: Distance between each site in a column.
        - per_max_sites: Max number of sites per column.
        - tip_length: Length of the probe tip.

        Returns:
        - sorted_coordinates: Sorted coordinates of the Neuropixel contacts.
        """
        vector = np.array(tip) - np.array(entry)
        unit_vector = vector / np.linalg.norm(vector)
        
        # Calculate the total length of the probe from tip to entry
        total_length = np.linalg.norm(vector)
        
        # Calculate the effective length for contacts (total length - tip_length)
        effective_length = total_length - tip_length

        all_coordinates = []
        
        for col_idx, (x, y, distance, max_sites) in enumerate(zip(x_bias, y_bias, sites_distance, per_max_sites)):
            column_coordinates = []
            
            for site_idx in range(max_sites):
                pos_along_probe = entry + (tip_length + site_idx * distance) * unit_vector
                
                pos_along_probe[1] += x  # ML
                pos_along_probe[2] += y  # DV
                
                # Stop if we have reached the effective length
                if np.linalg.norm(pos_along_probe - entry) >= effective_length:
                    break
                
                column_coordinates.append(pos_along_probe)
            
            all_coordinates.append(column_coordinates)
        
        combined_coordinates = np.vstack(all_coordinates)
        sorted_coordinates = combined_coordinates[np.argsort(combined_coordinates[:, 2])]
        sorted_coordinates = np.round(sorted_coordinates).astype(int)
        
        return sorted_coordinates
        
    