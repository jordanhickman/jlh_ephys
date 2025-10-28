import numpy as np
#from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


import matplotlib.pyplot as plt
import pickle as pkl
import os


def load_allen_CCF(resolution = 10):
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    # initialize the MouseConnectivityCache with 10um resolution
    mcc = MouseConnectivityCache(resolution=10)

    #  10um annotation volume
    annotation_volume_10um, _ = mcc.get_annotation_volume()

    # the structure tree to map IDs to acronyms
    structure_tree = mcc.get_structure_tree()

    # Generate ids:brainregion pairs
    id_to_brainreg = {node['id']: node['name'] for node in structure_tree.nodes()}

    return annotation_volume_10um, id_to_brainreg 

def get_regions_from_ccfs(ccf_coords_array, annotation_volume, id_to_name, scale_factor=0.1):
    '''
    ccf_coords_array: numpy array of shape (n, 3) containing CCF coordinates
    annotation_volume: numpy array of shape (n, n, n) containing allen CCF brain coords
    
    '''
    ccf_coords_array = (ccf_coords_array * scale_factor).astype(int)
    
    # check for out-of-bounds coordinates
    in_bounds = np.all((ccf_coords_array >= 0) & (ccf_coords_array < np.array(annotation_volume.shape)), axis=1)
    results = np.array(["Coordinates out of bounds"] * ccf_coords_array.shape[0], dtype=object)
    
    # Extract region_ids for in-bounds coordinates
    in_bounds_coords = ccf_coords_array[in_bounds]
    region_ids = annotation_volume[in_bounds_coords[:, 0], in_bounds_coords[:, 1], in_bounds_coords[:, 2]]
    
    # Map region_ids to names
    region_names = np.array([id_to_name.get(region_id, "Unknown") for region_id in region_ids])
    
    # Update results array with region names for in-bounds coordinates
    results[in_bounds] = region_names
    
    return results

def compare_brain_regions(list1, list2):
    discrepancies = []
    for i, (region1, region2) in enumerate(zip(list1, list2)):
        region1_cleaned = region1.replace(',', '')
        region2_cleaned = region2.replace(',', '')
        if region1_cleaned != region2_cleaned:
            discrepancies.append((i, region1_cleaned, region2_cleaned))
    return discrepancies


def apply_3d_transformations(full_coords, dAP=0, dML=0, rotation_ap=0, rotation_ml=0):
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

def adjust_probe_depth(coords, target_length):
    """
    Adjusts the probe depth by either trimming or extending the coordinates.
    
    """
    current_length = len(coords)
    
    if target_length == current_length:
        return coords
    
    if target_length < current_length:
        return coords[:target_length]
    
    # Calculate the distance between each pair of adjacent coordinates
    col1 = coords[0::4]
    col2 = coords[1::4]
    col3 = coords[2::4]
    col4 = coords[3::4]

    diffs0 = np.mean(np.diff(col1, axis=0), axis = 0)
    diffs1 = np.mean(np.diff(col2, axis=0), axis = 0)
    diffs2 = np.mean(np.diff(col3, axis=0), axis = 0)
    diffs3 = np.mean(np.diff(col4, axis=0), axis = 0)
    
    contacts_added = len(coords) - target_length
    
    while len(coords) < target_length:
        coords = np.vstack([coords, (coords[-4] + diffs0)])
    
    return coords



def calculate_planned_NPcoords(entry, tip, 
                               x_bias=[-16, -8, 8, 16], 
                               y_bias=[30, 10, 30, 10], 
                               sites_distance=[40, 40, 40, 40], 
                               per_max_sites=[240, 240, 240, 240],
                               tip_length=175):    
    
    """
    Calculate the Neuropixel coordinates from the tip to the entry point.

    Parameters:
    - entry, tip: Coordinates for the entry and tip of the probe. (AP, DV, ML)
    - x_bias, y_bias: Horizontal and vertical spacing between the columns.
    - sites_distance: Distance between each site in a column.
    - per_max_sites: Max number of sites per column.
    - tip_length: Length of the probe tip.

    Returns:
    - sorted_coordinates: Sorted coordinates of neuropixel contacts.
    """
    vector = np.array(tip) - np.array(entry)
    unit_vector = vector / np.linalg.norm(vector)

    # find a vector perpendicular to unit_vector 
    perp_vector = np.cross(unit_vector, [1, 0, 0])
    perp_vector /= np.linalg.norm(perp_vector)

    # find another perpendicular vector to complete the basis (similar to x, y and z)
    second_perp_vector = np.cross(unit_vector, perp_vector)
    second_perp_vector /= np.linalg.norm(second_perp_vector)

    print(f'unit_vector{unit_vector}')
    print(f'perp_vector{perp_vector}')
    print(f'second_perp_vector{second_perp_vector}')
    
    total_length = np.linalg.norm(tip - entry)
    print(total_length)
    effective_length = total_length - tip_length

    all_coordinates = []

    for col_idx, (x, y, distance, max_sites) in enumerate(zip(x_bias, y_bias, sites_distance, per_max_sites)):
        for site_idx in range(max_sites):
            pos_along_probe = entry + (tip_length + site_idx * distance) * unit_vector
            
            # apply x and y biases in the local coordinate system
            pos_with_bias = pos_along_probe + x * perp_vector + y * second_perp_vector
            
            if np.linalg.norm(pos_with_bias - entry) >= effective_length:
                break

            all_coordinates.append(pos_with_bias)
    
    combined_coordinates = np.vstack(all_coordinates)
    sorted_coordinates = combined_coordinates[np.argsort(combined_coordinates[:, 1])] # sort by dv 
    sorted_coordinates = np.round(sorted_coordinates).astype(int)
    
    return sorted_coordinates


def calculate_planned_stimcoords(entry, tip, spacing = 50):
    direction_vector = entry - tip  # Calculate the direction vector
    magnitude = np.linalg.norm(direction_vector)  # Calculate the magnitude of the direction vector
    step_size = spacing / magnitude  # Calculate the step size
    stim_coords = []
    for i in range(1, 17):
        contact = tip + i * step_size * direction_vector
        stim_coords.append(contact)
        
    stim_coords = np.vstack(stim_coords)
    return stim_coords


def plot_transformed_coords_multiple_probes(dAP, dML, 
                                            rotation_ap, 
                                            rotation_ml, 
                                            custom_coords_dict, 
                                            probeA_depth, 
                                            probeB_depth, 
                                            probeC_depth, 
                                            stim_depth, 
                                            annotation_volume_10um, 
                                            id_to_brainreg,
                                            scale_factor = 1):
    transformed_coords_dict = {}
    transformed_regions_dict = {}

    plt.figure(figsize=(20, 8))

    for idx, (probe, custom_coords) in enumerate(custom_coords_dict.items()):
        # Adjust depth
        target_length = locals().get(f"{probe}_depth")
        
        adjusted_coords = adjust_probe_depth(custom_coords, target_length)
        
        # Apply transformations
        transformed_coords = apply_3d_transformations(adjusted_coords, dAP, dML, rotation_ap, rotation_ml)
        brain_regions = get_regions_from_ccfs(transformed_coords, annotation_volume_10um, id_to_brainreg, scale_factor=1)

        # Save transformed coordinates and regions
        transformed_coords_dict[probe] = transformed_coords
        transformed_regions_dict[probe] = brain_regions

        plt.subplot(1, len(custom_coords_dict), idx + 1)
        unique_regions, unique_ids = np.unique(brain_regions, return_inverse=True)
        if probe == 'stim':
            plt.scatter(transformed_coords[:, 2], transformed_coords[:, 1], c=unique_ids, cmap='viridis')
            plt.xlim([min(transformed_coords[:, 2]) - 10, max(transformed_coords[:, 2]) + 10])
            plt.ylim([min(transformed_coords[:, 1]) - 10, max(transformed_coords[:, 1]) + 10])
        else:
            plt.scatter(transformed_coords[:, 2], transformed_coords[:, 0], c=unique_ids, cmap='viridis')
            plt.xlim([min(transformed_coords[:, 2]) - 10, max(transformed_coords[:, 2]) + 10])
            plt.ylim([min(transformed_coords[:, 0]) - 10, max(transformed_coords[:, 0]) + 10])
        
        if idx == 0 or idx == 2 or probe == 'stim':
            plt.gca().invert_yaxis()
        
        for i, region in enumerate(brain_regions):
            if probe == 'stim':
                if i % 1 == 0:
                    plt.text(transformed_coords[i, 2], transformed_coords[i, 1], region, fontsize=12)
            else:
                if i % 10 == 0:
                    plt.text(transformed_coords[i, 2], transformed_coords[i, 0], region, fontsize=12)
        
        plt.xlabel('ML coordinate')
        plt.ylabel('AP coordinate')
        plt.title(f'Transformed Coordinates in Brain Regions ({probe})')
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    return transformed_coords_dict, transformed_regions_dict

# Function to save the transformed coordinates and brain regions
def save_coordinates(transformed_coords_dict, transformed_regions_dict, save_path):
    with open(os.path.join(save_path, 'coords_dict.pkl'), 'wb') as file:
        pkl.dump(transformed_coords_dict, file)
    print("Coordinates saved.")
   
    with open(os.path.join(save_path, 'regions_dict.pkl'), 'wb') as file:
        pkl.dump(transformed_regions_dict, file)
    print("Brain regions saved.")
    
    
    
'''
# Example usage
from ipywidgets import interactive, IntSlider, Button, VBox, Layout
from IPython.display import display
from transformation_plotting import plot_transformed_coords_multiple_probes, save_coordinates

# Create sliders for the interactive part
slider_layout = Layout(width='800px')
dAP_slider = IntSlider(min=-200, max=200, step=1, description='dAP:', layout=slider_layout)
dML_slider = IntSlider(min=-200, max=200, step=1, description='dML:', layout=slider_layout)
rotation_ap_slider = IntSlider(min=-15, max=15, step=1, description='Rotation AP:', layout=slider_layout)
rotation_ml_slider = IntSlider(min=-15, max=15, step=1, description='Rotation ML:', layout=slider_layout)

# Depth sliders for each probe
probeA_depth_slider = IntSlider(min=1, max=300, step=1, value=len(mouse_obj.probe_coords['probeA']), description='Probe A Depth:', layout=slider_layout)
probeB_depth_slider = IntSlider(min=1, max=300, step=1, value=len(mouse_obj.probe_coords['probeB']), description='Probe B Depth:', layout=slider_layout)
probeC_depth_slider = IntSlider(min=1, max=300, step=1, value=len(mouse_obj.probe_coords['probeC']), description='Probe C Depth:', layout=slider_layout)
stim_depth_slider = IntSlider(min=1, max=32, step=1, value=len(mouse_obj.probe_coords['stim']), description='Stim Depth:', layout=slider_layout)

# Create the save button
save_button = Button(description="Save Coordinates")

# Callback for interactive plotting
def update_plot(dAP, dML, rotation_ap, rotation_ml, probeA_depth, probeB_depth, probeC_depth, stim_depth):
    transformed_coords_dict, transformed_regions_dict = plot_transformed_coords_multiple_probes(
        dAP, dML, rotation_ap, rotation_ml, mouse_obj.probe_coords, 
        probeA_depth, probeB_depth, probeC_depth, stim_depth, annotation_volume_10um, id_to_brainreg
    )

    # Save function callback
    def save_callback(button):
        save_coordinates(transformed_coords_dict, transformed_regions_dict, mouse_obj.path)
    
    # Attach save callback to button
    save_button.on_click(save_callback)

# Create the interactive plot
interactive_plot = interactive(
    update_plot,
    dAP=dAP_slider,
    dML=dML_slider,
    rotation_ap=rotation_ap_slider,
    rotation_ml=rotation_ml_slider,
    probeA_depth=probeA_depth_slider,
    probeB_depth=probeB_depth_slider,
    probeC_depth=probeC_depth_slider,
    stim_depth=stim_depth_slider
)

# Layout for the sliders and save button
layout = VBox([interactive_plot, save_button])

# Display the layout
display(layout)


'''

import matplotlib as mpl
def plot_probe_depth_vs_column(units, column='no_spikes', save_path=None, save_name='probe_depth_vs_unit'):
    """
    Plot depth vs. a specified column for each probe in the provided dataframe.

    Parameters:
    - units: pd.DataFrame
        DataFrame containing 'probe', 'brain_reg', 'depth', and the column to be plotted (default: 'Amplitude').
    - column: str
        The column to be plotted on the x-axis (default: 'Amplitude').
    - save_path: str
        Directory path to save the figures (optional). If None, figures are not saved.
    - save_name: str
        Base name for the saved figures (default: 'probe_ch_vs_unit').
    """
    # Set up the figure and axes
    fig, ax = plt.subplots(1, len(units['probe'].unique()), sharex=True, sharey=True, figsize=(15, 15))
    
    # Ensure ax is iterable, even for a single subplot
    if len(units['probe'].unique()) == 1:
        ax = [ax]

    regions = units.sort_values('ch')['brain_reg'].unique()

    for p, probe in enumerate(units['probe'].unique()):
        data = units.loc[units['probe'] == probe]
        
        # Generate a colormap
        cmap = mpl.colormaps['viridis']
        colors = cmap(np.linspace(0, 1, len(regions)))

        # Plot data for each region
        for r, region in enumerate(regions):
            ax[p].scatter(
                data.loc[data['brain_reg'] == region, column],
                data.loc[data['brain_reg'] == region, 'ch'],
                color=colors[r],
                label=region
            )

        ax[p].set_title(f'Probe {probe}')
        #ax[p].invert_yaxis()
        ax[p].set_xlabel(column)

    # Set the y-axis label for the first plot
    ax[0].set_ylabel('Ch')

    # Add a legend to the figure
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{save_name}.png'))
        plt.savefig(os.path.join(save_path, f'{save_name}.pdf'))
        
    return fig, ax 

import matplotlib as mpl
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d

def plot_lfp_power_and_gamma(
    lfp_data_list,
    probe_names,
    freq_range=(0, 100),
    gamma_range=(30, 50),
    fs=2500,
    save_path=None,
    save_name='lfp_power_gamma'
):
    """
    Plot raw LFP data, LFP power heatmaps, and gamma power for up to 3 probes.

    Parameters:
    - lfp_data_list: list of np.ndarray
        List of LFP data arrays (one per probe), where each array is shaped as (time, channels).
    - probe_names: list of str
        Names of the probes (e.g., ['Probe A', 'Probe B', 'Probe C']).
    - freq_range: tuple
        Frequency range for power spectrum computation (default: (0, 100) Hz).
    - gamma_range: tuple
        Gamma frequency range for power extraction (default: (30, 50) Hz).
    - fs: int
        Sampling rate in Hz (default: 2500 Hz).
    - save_path: str
        Directory to save the figures (optional). If None, figures are not saved.
    - save_name: str
        Base name for the saved figures (default: 'lfp_power_gamma').
    """
    assert len(lfp_data_list) == len(probe_names), "Number of probes and LFP data arrays must match."
    assert 1 <= len(lfp_data_list) <= 3, "This function supports 1 to 3 probes."

    num_probes = len(lfp_data_list)
    fig, axs = plt.subplots(num_probes, 3, figsize=(13, 4 * num_probes))

    if num_probes == 1:  # Ensure axs is iterable for single probe
        axs = [axs]

    # Prepare the colormap
    cmap = mpl.colormaps['viridis']

    for i, (lfp_data, probe_name) in enumerate(zip(lfp_data_list, probe_names)):
        # Compute power matrix
        power_matrix = []
        for ch in range(lfp_data.shape[1]):
            ch_data = lfp_data[:, ch]
            f, Pxx = welch(ch_data, fs=fs)
            freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
            power_matrix.append(10 * np.log10(Pxx[freq_mask]))
        power_matrix = np.array(power_matrix)
        f_masked = f[freq_mask]

        # Extract gamma power
        gamma_mask = (f_masked >= gamma_range[0]) & (f_masked <= gamma_range[1])
        gamma_power = np.mean(power_matrix[:, gamma_mask], axis=1)
        gamma_power_smooth = gaussian_filter1d(gamma_power, sigma=2)

        # Plot raw LFP data
        axs[i][0].imshow(
            lfp_data.T, origin='lower', aspect='auto', vmin=-350, vmax=350, cmap=cmap
        )
        axs[i][0].set_title(f'{probe_name} Raw LFP', fontsize=12)
        axs[i][0].set_ylabel('Channel', fontsize=10)
        axs[i][0].set_xlabel('Time (Samples)', fontsize=10)

        # Plot LFP power heatmap
        im = axs[i][1].imshow(
            power_matrix,
            aspect='auto',
            extent=[f_masked[0], f_masked[-1], 0, lfp_data.shape[1]],
            cmap=cmap,
            origin='lower',
        )
        axs[i][1].set_title(f'{probe_name} LFP Power', fontsize=12)
        axs[i][1].set_xlabel('Frequency (Hz)', fontsize=10)

        # Plot gamma power
        axs[i][2].plot(gamma_power, np.arange(0, len(gamma_power)), label='Raw')
        axs[i][2].plot(gamma_power_smooth, np.arange(0, len(gamma_power_smooth)), label='Smoothed')
        axs[i][2].set_title(f'{probe_name} Gamma Power', fontsize=12)
        axs[i][2].set_xlabel('Power (dB)', fontsize=10)
        axs[i][2].legend(fontsize=8)

        # Add colorbar to the LFP power heatmap
        fig.colorbar(im, ax=axs[i][1], orientation='vertical', shrink=0.8, label='Power (dB)')

    # Adjust layout
    plt.tight_layout()

    # Save figures if save_path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'{save_name}.png'))
        plt.savefig(os.path.join(save_path, f'{save_name}.pdf'))
        
    return fig, axs