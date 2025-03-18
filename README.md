# RKNS
## An "Adapter" Format for medical time-series (EEG, ECG, ...)

> **Note**: RKNS is still in early development and not yet ready for production use. APIs may change frequently.

## Overview

This repository contains the reference implementation of the RKNS file format in python.

With RKNS we are develping a flexible data format for storing and processing medical time series, e.g. electrophysiological (EXG) data, designed to address the challenges of working with various ExG file formats (EDF, EDF+, BDF, BDF+, etc.).

Rather than creating just another rigid standard, RKNS serves as a universal adapter between different formats while preserving all original information and enabling optional standardization.

## Long-term Goals

- **Format Preservation**: Store original input files as raw byte blobs, ensuring data integrity and reproducibility
- **Flexible Standardization**: Support optional channel name mapping and data normalization
- **Efficient Processing**: Leverage Zarr V3 for chunked reading/writing and compression of large arrays
- **Rich Metadata**: Store comprehensive metadata with a flexible schema
- **Cross-Platform**: Support multiple backend options (file, memory, cloud)
- **Cross-Language**: Based on Zarr with implementations in Python, JavaScript, and other languages
- **Future-Proof**: Hierarchical structure allows for extensions without breaking backward compatibility

# Install from GitHub (development version)
pip install git+https://github.com/rekonas-com/rkns-python


# Contributing
Contributions are welcome! Please see CONTRIBUTING.md for details on how to contribute to this project.
