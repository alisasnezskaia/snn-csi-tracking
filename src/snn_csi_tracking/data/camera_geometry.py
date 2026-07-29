"""Full camera-pose calibration from floor-plane point correspondences, an
upgrade over a flat single-plane homography (see scripts/
calibrate_head_homography.py, superseded by this module once real room
geometry became available -- see scripts/calibrate_camera_pose.py).

Given >=4 floor points (world Z=0, from the room's floor plan) and their
matching pixel locations in a reference frame from the fixed session
camera, cv2.solvePnP recovers the camera's full pose (rotation + position
in the room). That pose lets us reproject an image pixel through a
ray-plane intersection at ANY assumed real-world height -- not just the
floor -- which is exactly the fix for the head-height-vs-floor-plane
mismatch: rather than calibrating one homography for a fixed head height
(which only holds while standing), we can intersect the same camera ray at
whatever height is appropriate for a given frame (e.g. lower for a seated
subject), if/when that height is known or estimated.

No lens calibration (checkerboard) was done for this dataset's camera, so
the camera matrix here is an ESTIMATE from image size and an assumed
horizontal field of view (a typical smartphone default, adjustable) --
this is a real, honestly-reported source of approximation, on top of the
floor-plan point positions themselves being read off a sketch rather than
a precise survey. Both are documented at the call site
(scripts/calibrate_camera_pose.py) via reprojection error.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_HORIZONTAL_FOV_DEG = 68.0  # typical smartphone rear/front camera, adjust with real data if known


def estimate_camera_matrix(image_width: int, image_height: int,
                            horizontal_fov_deg: float = DEFAULT_HORIZONTAL_FOV_DEG) -> np.ndarray:
    """Pinhole camera matrix from image size + assumed horizontal FOV (no
    real lens calibration available -- see module docstring)."""
    fx = (image_width / 2) / np.tan(np.deg2rad(horizontal_fov_deg) / 2)
    fy = fx  # square pixels assumed
    cx, cy = image_width / 2, image_height / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def deproject_pixel_to_camera_frame(pixel_xy_norm: np.ndarray, depth_m: np.ndarray,
                                     image_width: int, image_height: int,
                                     K: np.ndarray | None = None) -> np.ndarray:
    """Standard pinhole back-projection: given a normalized [0,1] MediaPipe
    (u, v) and a metric depth at that pixel (see depth_extraction.py's
    metric-indoor checkpoint), returns the real-world (X, Y, Z) in the
    CAMERA's own frame (meters) -- not room/floor-plan coordinates, since
    that would additionally need the camera's pose (solve_camera_pose),
    which needs floor-plan point correspondences we're deliberately not
    collecting here.

    This is a valid real-world coordinate system on its own: as long as the
    camera doesn't move between capture sessions (a standing assumption of
    this dataset -- see calibrate_head_homography.py), it's just a rigid
    rotation+translation away from room coordinates, which changes no
    distance, angle, or velocity -- only where the origin/axes sit. That's
    what makes skipping the room-pose calibration a legitimate
    simplification here, unlike skipping metric depth calibration would be.

    Follows the standard monocular-depth-benchmark convention (NYU-Depth-v2,
    which Depth-Anything-V2's metric-indoor checkpoint is trained against):
    depth_m is Z, the distance along the camera's optical axis -- not the
    radial/Euclidean distance from the camera to the point.

    pixel_xy_norm: (..., 2) normalized [0,1] image coordinates.
    depth_m: (...,) metric depth in meters, same leading shape as pixel_xy_norm.
    K: camera intrinsics; if None, estimated from image size via estimate_camera_matrix.
    Returns (..., 3) (X, Y, Z) in meters, camera frame.
    """
    if K is None:
        K = estimate_camera_matrix(image_width, image_height)

    pixel_xy_norm = np.asarray(pixel_xy_norm, dtype=np.float64)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    u = pixel_xy_norm[..., 0] * (image_width - 1)
    v = pixel_xy_norm[..., 1] * (image_height - 1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m
    return np.stack([X, Y, Z], axis=-1)


def solve_camera_pose(image_points_px: np.ndarray, world_points_xy_m: np.ndarray,
                       K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """image_points_px: (N,2) pixel coords. world_points_xy_m: (N,2) real-world
    floor (X,Y) in meters (Z=0 implied -- these are floor-plane points).
    Returns (rvec, tvec) such that P_camera = R @ P_world + tvec."""
    object_points = np.hstack([world_points_xy_m, np.zeros((len(world_points_xy_m), 1))]).astype(np.float64)
    image_points = image_points_px.astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, K, distCoeffs=None)
    if not ok:
        raise RuntimeError("solvePnP failed to converge -- check point correspondences")
    return rvec, tvec


def camera_center_world(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """The camera's own position in world (room) coordinates -- a useful
    sanity check against wherever 'C' was placed on the floor plan."""
    R, _ = cv2.Rodrigues(rvec)
    return (-R.T @ tvec).ravel()


def project_pixel_to_world(pixel_xy: np.ndarray, height_m: float, K: np.ndarray,
                            rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Ray-plane intersection: given a pixel (u,v) and an assumed real-world
    height above the floor (e.g. ~1.6m for a standing head), returns the
    real-world (X, Y) floor position whose vertical ray through that height
    projects to this pixel. This is what makes the calibration usable for
    the head/shoulder landmark (which sits well above the floor plane the
    correspondence points were measured on), and -- if a seated height is
    ever estimated separately -- for seated frames too, without needing a
    second dedicated calibration.

    pixel_xy: (2,) or (N,2) pixel coordinates.
    """
    pixel_xy = np.atleast_2d(pixel_xy).astype(np.float64)
    R, _ = cv2.Rodrigues(rvec)
    K_inv = np.linalg.inv(K)
    cam_center = camera_center_world(rvec, tvec)  # (3,)

    homogeneous = np.hstack([pixel_xy, np.ones((len(pixel_xy), 1))])  # (N,3)
    cam_rays = (K_inv @ homogeneous.T).T  # (N,3), camera-space ray direction
    world_dirs = (R.T @ cam_rays.T).T  # (N,3), same ray in world coordinates

    s = (height_m - cam_center[2]) / world_dirs[:, 2]
    world_points = cam_center[None, :] + s[:, None] * world_dirs  # (N,3)
    return world_points[:, :2]  # drop Z, it's == height_m by construction


def reprojection_error_m(image_points_px: np.ndarray, world_points_xy_m: np.ndarray,
                          K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Per-point floor-plane reprojection error in meters -- projects each
    input pixel back to the floor (height=0) and compares to the world
    point it was paired with. This is the calibration's own precision
    floor; report it, don't hide it."""
    recovered = project_pixel_to_world(image_points_px, height_m=0.0, K=K, rvec=rvec, tvec=tvec)
    return np.linalg.norm(recovered - world_points_xy_m, axis=1)
