"""Uncompressed NPZ arrays with a checksum computed during the single write."""
import hashlib
import io
import os
import zipfile
from pathlib import Path

import numpy as np


class HashingWriteStream:
    """Forward-only ZIP output; data descriptors avoid header seek rewrites."""
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.position = 0

    def write(self, value):
        written = self.stream.write(value)
        self.digest.update(memoryview(value).cast("B")[:written])
        self.position += written
        return written

    def tell(self):
        return self.position

    def flush(self):
        return self.stream.flush()

    def seekable(self):
        return False

    def read(self, *args):
        # NumPy uses the presence of read() to recognize a file-like ZIP target.
        raise io.UnsupportedOperation("write-only checksum stream")


def save_npz_with_hash(path, arrays, writer=np.savez, compression_level=None):
    """Array values/dtypes/order are untouched; the ZIP container is streaming."""
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with open(temporary, "wb") as stream:
        hashing = HashingWriteStream(stream)
        if compression_level is None:
            writer(hashing, **arrays)
        else:
            with zipfile.ZipFile(hashing, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=compression_level, allowZip64=True) as archive:
                for name, value in arrays.items():
                    with archive.open(name + ".npy", "w", force_zip64=True) as member:
                        np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)
        digest = hashing.digest.hexdigest()
    os.replace(temporary, path)
    return digest
