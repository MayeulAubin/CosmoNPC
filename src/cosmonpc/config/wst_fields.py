"""Edit paths and geometry, then run through example.py or run_stats(config)."""
CONFIG = {
    'statistic': 'wst',
    'input_type': 'density',
    'geometry': 'box-like',
    'fields': {'data_a': {'path': '/path/to/density.h5', 'kind': 'overdensity'}},
    # Alternatives: a global NumPy array, or {'tiles': [...], 'missing_tiles': 'zero'}.
    'nmesh': [256, 256, 256],
    'boxsize': [2000., 2000., 2000.],
    'field_origin': [-1000., -1000., -1000.],
    'precision': 'single',
    'wst': {
        'J': 4, 'L': 4, 'sigma': .8, 'q': [.8], 'max_order': 2,
        # Reduce scales/degrees for runtime; inclusive J and L conventions.
        # 'scales': [0, 2, 4], 'ells': [0, 1, 2],
        # 'mask': binary_array_or_HDF5_Field,
        'm_resolved': False,
    },
    # For a survey set geometry='survey-like' and exactly one selection:
    # 'survey_weight': {'path': '/path/to/weight.h5'},  # same grid, values [0,1]
    # 'survey_randoms': random_catalogue,
    'output_dir': './output/wst',
}
