import numpy as np
from math import sqrt, cos, sin


def get_quaternion(theta):
    rv = theta[0]
    q1 = rv[0] * np.sin(np.pi * theta[1] / 360.0)
    q2 = rv[1] * np.sin(np.pi * theta[1] / 360.0)
    q3 = rv[2] * np.sin(np.pi * theta[1] / 360.0)
    # Constraint to creat quaternion
    q0 = np.sqrt(1 - q1 * q1 - q2 * q2 - q3 * q3)
    # Quaternion
    q = np.array([q0, q1, q2, q3], dtype=np.float64)
    return q


def q_normalised(rv):
    q02 = 1 - np.sum(n * n for n in rv[1:])
    q2 = np.array([q02, abs(rv[1]), abs(rv[2]), abs(rv[3])], dtype=np.float64)
    q = np.sqrt(q2)
    return q


def get_RM(q):
    RM = np.array(
        [
            [
                1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2),
                2.0 * (q[1] * q[2] - q[3] * q[0]),
                2.0 * (q[1] * q[3] + q[2] * q[0]),
            ],
            [
                2.0 * (q[1] * q[2] + q[3] * q[0]),
                1.0 - 2.0 * (q[1] ** 2 + q[3] ** 2),
                2.0 * (q[2] * q[3] - q[1] * q[0]),
            ],
            [
                2.0 * (q[1] * q[3] - q[2] * q[0]),
                2.0 * (q[2] * q[3] + q[1] * q[0]),
                1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2),
            ],
        ],
        dtype=np.float64,
    )
    return RM


def derivatives_wrt_q(q):
    # First-order derivatives of R wrt q
    dRdq1 = np.array(
        [
            [
                0.0,
                2.0 * q[2] + 2.0 * (q[1] * q[3]) / q[0],
                2.0 * q[3] - 2.0 * (q[1] * q[2]) / q[0],
            ],
            [
                2.0 * q[2] - 2.0 * (q[1] * q[3]) / q[0],
                -2.0 * q[1],
                -2.0 * q[0] + 2.0 * (q[1] * q[1]) / q[0],
            ],
            [
                2.0 * q[3] + 2.0 * (q[1] * q[2]) / q[0],
                2.0 * q[0] - 2.0 * (q[1] * q[1]) / q[0],
                -2.0 * q[1],
            ],
        ],
        dtype=np.float64,
    )
    dRdq2 = np.array(
        [
            [
                -2.0 * q[2],
                2.0 * q[1] + 2.0 * (q[2] * q[3]) / q[0],
                2.0 * q[0] - 2.0 * (q[2] * q[2]) / q[0],
            ],
            [
                2.0 * q[1] - 2.0 * (q[2] * q[3]) / q[0],
                0.0,
                2.0 * q[3] - 2.0 * (q[1] * q[2]) / q[0],
            ],
            [
                -2.0 * q[0] + 2.0 * (q[2] * q[2]) / q[0],
                2.0 * q[3] - 2.0 * (q[1] * q[2]) / q[0],
                -2.0 * q[1],
            ],
        ],
        dtype=np.float64,
    )
    dRdq3 = np.array(
        [
            [
                -2.0 * q[3],
                -2.0 * q[0] + 2.0 * (q[3] * q[3]) / q[0],
                2.0 * q[1] - 2.0 * (q[2] * q[3]) / q[0],
            ],
            [
                2.0 * q[0] - 2.0 * (q[3] * q[3]) / q[0],
                -2.0 * q[3],
                2.0 * q[2] - 2.0 * (q[1] * q[3]) / q[0],
            ],
            [
                2.0 * q[1] + 2.0 * (q[2] * q[3]) / q[0],
                2.0 * q[2] - 2.0 * (q[1] * q[3]) / q[0],
                0.0,
            ],
        ],
        dtype=np.float64,
    )
    return np.array([dRdq1, dRdq2, dRdq3])


# source https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    import math

    assert isRotationMatrix(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def rotmat_from_axisangle(axis, theta):
    """
    Returns rotation matrix of counterclockwise rotation about axis.

    Rotation matrix is obtained from expanding Rodiguez formula.
    This requires angle in radians.
    """
    import math

    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    x, y, z = axis
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    ca, sa = math.cos(theta), math.sin(theta)
    aa = 1 - ca
    return np.array(
        [
            [ca + xx * aa, xy * aa - z * sa, xz * aa + y * sa],
            [xy * aa + z * sa, ca + yy * aa, yz * aa - x * sa],
            [xz * aa - y * sa, yz * aa + x * sa, ca + zz * aa],
        ],
        dtype="float",
    )