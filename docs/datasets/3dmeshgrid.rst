3D Meshgrid provides four dense room impulse-response grids, one for each loudspeaker position. Each call downloads exactly one approximately 1 GB provider HDF5 file at 16 kHz.

IRDL writes the moving-microphone data as a reciprocal ``SingleRoomMIMOSRIR`` SOFA representation: ``source_coordinates`` are the measured microphone-grid locations and the sole ``receiver_coordinates`` position is the physical loudspeaker. The original HDF5 file remains available through ``output_format="raw"``.

The source dataset is licensed under CC BY 4.0. See the `dataset page <https://yh-audio.github.io/meshgrid-ir.html>`_ for measurement details.
