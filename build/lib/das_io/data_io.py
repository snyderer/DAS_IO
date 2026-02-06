"""
Methods for loading and rehydrating preprocessed files.
Eventually this will be replaced by preprocess_DAS.io, but the issue of 
incompatibility between DAS4Whales and PyQt6 needs to be resolved first.
"""

import os
import h5py
import numpy as np

# -----------------------------------------------
# one function to do it all
# -----------------------------------------------
def load_tx(filepath, nonzeros=None, original_shape=None, fs=None, dx=None, settings=None):
    fk_dehyd, timestamp = load_preprocessed_h5(filepath)

    if settings is None:
        # Require all individual parameters to be provided
        if None in (nonzeros, original_shape, fs, dx):
            # Fall back to loading settings from file
            settings_file = find_settings_h5(filepath)
            settings = load_settings_preprocessed_h5(settings_file)

    if settings is not None:
        # Use values from settings
        nonzeros = settings['rehydration_info']['nonzeros_mask']
        original_shape = settings['rehydration_info']['target_shape']
        dx = settings['processing_settings']['dx']
        fs = settings['processing_settings']['fs']
    else:
        # If settings is still None, check parameters explicitly
        if None in (nonzeros, original_shape, fs, dx):
            raise ValueError(
                "Either 'settings' must be provided or "
                "all of 'nonzeros', 'original_shape', 'fs', and 'dx' must be supplied."
            )

    # Now we know nonzeros, original_shape, fs, and dx are defined
    tx = 1e9 * rehydrate(fk_dehyd, nonzeros, original_shape, return_format='tx')
    x = np.arange(0, tx.shape[0]) * dx
    t = np.arange(0, tx.shape[1]) / fs

    return tx, t, x
    
# -----------------------------------------------
# load and rehydrate data from h5
# -----------------------------------------------
def load_preprocessed_h5(filepath):
    with h5py.File(filepath, 'r') as h:
        fk_dehyd = h['fk_dehyd'][...]
        timestamp = h['timestamp'][()]
    return fk_dehyd, timestamp

def rehydrate(fk_dehyd, nonzeros, original_shape, return_format='tx'):
    nx, nt = original_shape
    nf = nt // 2 + 1
    if nonzeros.shape != (nx, nf):
        raise ValueError("Mask shape mismatch")
    if len(fk_dehyd) != np.sum(nonzeros):
        raise ValueError("Nonzeros count mismatch")
    fk_positive = np.zeros((nx, nf), dtype=complex)
    fk_positive[nonzeros] = fk_dehyd
    if return_format == 'fk':
        return fk_positive
    elif return_format == 'tx':
        fx_domain = np.fft.ifft(fk_positive, axis=0)
        tx_data = np.fft.irfft(fx_domain, n=nt, axis=1)
        return tx_data
    else:
        raise ValueError("return_format must be 'tx' or 'fk'")
# -----------------------------------------------
# find, loading, and preparing settings from settings.h5
# -----------------------------------------------
def find_settings_h5(filepath):
    settings_filepath = os.path.join(os.path.dirname(filepath), 'settings.h5')
    if os.path.isfile(settings_filepath):
        return settings_filepath
    else:
        return None
    
def load_settings_preprocessed_h5(filepath):
    """
    Load settings and rehydration info.
    
    Returns:
    --------
    settings_data : dict
        Dictionary with all settings and rehydration info
    """
    with h5py.File(filepath, 'r') as f:
        settings_data = {
            'created': f.attrs.get('created', 'unknown'),
            'version': f.attrs.get('version', 'unknown')
        }
        
        # Load original metadata
        if 'original_metadata' in f:
            orig_meta = {}
            grp = f['original_metadata']
            
            for key in grp.keys():
                data = grp[key][...]
                # Convert single-element arrays back to scalars if appropriate
                if isinstance(data, np.ndarray) and data.size == 1 and data.dtype.kind in 'biufc':
                    orig_meta[key] = data.item()
                elif isinstance(data, bytes):
                    orig_meta[key] = data.decode()
                else:
                    orig_meta[key] = data
            
            settings_data['original_metadata'] = orig_meta
        
        # Load processing settings (now all datasets)
        if 'processing_settings' in f:
            proc_settings = {}
            grp = f['processing_settings']
            
            for key in grp.keys():
                if isinstance(grp[key], h5py.Group):  # it's a subgroup
                    if key == 'bandpass_filter':
                        bp_grp = grp['bandpass_filter']
                        filter_order = bp_grp['filter_order'][()]
                        cutoff_freqs = bp_grp['cutoff_freqs'][...]
                        filter_type = bp_grp['filter_type'][()].decode() \
                            if isinstance(bp_grp['filter_type'][()], bytes) else bp_grp['filter_type'][()]
                        proc_settings['bandpass_filter'] = [filter_order, list(cutoff_freqs), filter_type]
                else:
                    data = grp[key][...]
                    if isinstance(data, np.ndarray) and data.size == 1 and data.dtype.kind in 'biufc':
                        proc_settings[key] = data.item()
                    elif isinstance(data, bytes):
                        proc_settings[key] = data.decode()
                    else:
                        proc_settings[key] = data
            
            settings_data['processing_settings'] = proc_settings

        # Load rehydration info
        if 'rehydration_info' in f:
            rehyd_grp = f['rehydration_info']
            settings_data['rehydration_info'] = {
                'nonzeros_mask': rehyd_grp['nonzeros_mask'][...],
                'target_shape': tuple(rehyd_grp['target_shape'][...])
            }
        
        # Load axes
        if 'axes' in f:
            axes_grp = f['axes']
            settings_data['axes'] = {
                'frequency': axes_grp['frequency'][...],
                'wavenumber': axes_grp['wavenumber'][...]
            }
        
        # Load file_map (structured array)
        if 'file_map' in f:
            table = f['file_map'][...]  # structured numpy array
            table_filenames = np.array(
                [fn.decode() if isinstance(fn, bytes) else fn for fn in table['filename']],
                dtype=object
            )
            table = table.copy()
            table['filename'] = table_filenames
            settings_data['file_map'] = table
            
        return settings_data