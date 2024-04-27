import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


def load_allen_CCF(resolution = 10):
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
