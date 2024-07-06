import Imath
import OpenEXR as exr
from collections import UserDict
from pathlib import Path

import numpy


class AttrDict(UserDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)
            elif isinstance(value, list):
                self[key] = [AttrDict(item) for item in value]

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == "data":
            return super().__setattr__(key, value)
        self[key] = value

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        return super().__getitem__(key)


def read_depth_exr_file(filepath: Path):
    exrfile = exr.InputFile(filepath.as_posix())
    raw_bytes = exrfile.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map = numpy.reshape(depth_vector, (height, width))
    return depth_map


if __name__ == "__main__":
    p = Path("../data/cleargrasp-dataset-train/cup-with-waves-train/camera-normals/000000000-cameraNormals.exr")
    dm = read_depth_exr_file(p)
    print(dm)