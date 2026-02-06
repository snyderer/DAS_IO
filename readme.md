Includes tools for loading DAS data that have been compressed with https://github.com/snyderer/preprocess_DAS

install with:
pip install git+https://github.com/snyderer/DAS_IO.git

See example.ipynb for an example of loading and plotting DAS data.

The data h5 files store data in the F-K domain, restructured into a vector to minimize memory. The settings.h5 contains information on reconstructing the original time-distance data. This can all be done in one line:

from das_io import data_io as io
tx, t, x = io.load_tx(filepath)

Data from the same dataset must be saved in the same directory. The code will look for the settings.h5 file inside the path to the file you are trying to load. If there is no settings file there, it will not work.

If loading several files from the same directory/dataset, you can use: 
settings = io.load_settings_preprocessed_h5(settings_filepath)
tx, t, x = io.load_tx(filepath, settings=settings)

This will avoid having to load the settings file multiple times for different files within the same dataset.