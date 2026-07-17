"""
data_sources -- pluggable dataset loaders behind one canonical interface.

Each module exposes load_events(data_dir) and load_item_properties(data_dir)
returning our canonical schema, so phases never know which dataset they're on.
Select with the DATASET env var (see load_data.py).
"""
