import numpy as np
import pyroomacoustics as pra
from typing import Tuple

FS = 16000
MARGIN = 0.5

ROOM_SIZES = [
    (4, 5, 3),    # Small
    (6, 8, 3),    # Medium
    (10, 12, 4),  # Large
    (15, 20, 5)   # Very Large
]

ABSORPTION_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
REFLECTION_ORDERS = [0, 1, 2, 3, 4, 5]

def simulate_rir() -> Tuple[np.ndarray, dict]:
    """
    Simulates a room impulse response and returns the RIR and metadata.
    """
    room_dim = ROOM_SIZES[np.random.randint(len(ROOM_SIZES))]
    absorption = np.random.choice(ABSORPTION_LEVELS)
    max_order = np.random.choice(REFLECTION_ORDERS)

    # Generate valid source and mic positions
    source_pos = np.random.uniform([MARGIN]*3, [d - MARGIN for d in room_dim])
    mic_pos = np.random.uniform([MARGIN]*3, [d - MARGIN for d in room_dim])

    # Create room
    materials = pra.Material(absorption)
    room = pra.ShoeBox(
        room_dim,
        fs=FS,
        materials=materials,
        max_order=max_order,
        ray_tracing=False,
        air_absorption=True
    )

    room.add_source(source_pos)
    room.add_microphone_array(pra.MicrophoneArray(mic_pos[:, np.newaxis], FS))
    room.compute_rir()
    rir = room.rir[0][0]

    metadata = {
        "room_dim": room_dim,
        "absorption": absorption,
        "sample_rate": FS,
        "source_pos": source_pos.tolist(),
        "mic_pos": mic_pos.tolist(),
        "max_order": max_order
    }

    return rir, metadata
