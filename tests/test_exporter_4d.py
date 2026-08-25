"""Unit tests for 4DGS dataset exporter logic and camera conversions."""

import numpy as np
from fdanyone.nerfstudio.cameras import camera_to_nerfstudio


def test_camera_to_nerfstudio_opengl_conversion():
    # Identity OpenCV matrix (Z-forward, Y-down, X-right)
    opencv_c2w = np.eye(4, dtype=np.float64)
    opengl_c2w = camera_to_nerfstudio(opencv_c2w)

    # Nerfstudio converts Y-up to Z-up and OpenCV to OpenGL
    expected = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert np.allclose(opengl_c2w, expected)


def test_camera_to_nerfstudio_translation():
    opencv_c2w = np.eye(4, dtype=np.float64)
    opencv_c2w[0, 3] = 1.5
    opencv_c2w[1, 3] = 2.5
    opencv_c2w[2, 3] = 3.5

    opengl_c2w = camera_to_nerfstudio(opencv_c2w)
    # y_up to z_up maps y' = -z and z' = y
    assert np.isclose(opengl_c2w[0][3], 1.5)
    assert np.isclose(opengl_c2w[1][3], -3.5)
    assert np.isclose(opengl_c2w[2][3], 2.5)
