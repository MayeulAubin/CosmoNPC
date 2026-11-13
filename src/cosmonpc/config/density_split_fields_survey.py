"""Edit paths and global grid geometry for tiled DM survey density splits."""
CONFIG = {
    'input_type': 'density',
    'geometry': 'survey-like',
    'statistic': 'pk',
    'fields': {
        'data_a': {
            'tiles': ['tile1_density_field.h5', 'tile3_density_field.h5'],
            'kind': 'overdensity',
            'missing_tiles': 'zero',
        },
    },
    'survey_weight': 'survey_weight.h5',  # Same grid; values in [0,1].
    # Alternatively remove survey_weight and set survey_randoms to a catalogue.
    'nmesh': [256, 256, 256],
    'boxsize': [2000., 2000., 2000.],
    'field_origin': [-1000., -1000., -1000.],
    'precision': 'single',
    'normalization_scheme': 'mesh',
    'k_min': .005,
    'k_max': .15,
    'k_bins': 20,
    'poles': [0, 2],
    'output_dir': './survey_density_split_results',
    'density_split': {
        'nquantiles': 5,
        'smoothing_radius': 10.,
        'measurements': [['Q', 'data_a'], ['Q', 'Q']],
        # For statistic='bk_sugi', set angu_config=[0,0,0] and use e.g.
        # [['Q','data_a','data_a'], ['Q','Q','data_a'], ['Q','Q','Q']].
    },
}
