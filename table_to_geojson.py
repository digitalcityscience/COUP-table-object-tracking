from collections.abc import Mapping, Sequence

import numpy as np
from pyproj import Transformer

from pixel_to_utm import BasemapHomography, project_pixels_to_utm

"""
for now just return the centroids
building calibration comes later.
"""

# Building a Transformer is expensive (CRS parsing/grid lookups), so create it
# once and reuse it across calls instead of going through geopandas' to_crs,
# which rebuilds one internally on every invocation.
_UTM_TO_WGS84 = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)


def markers_json_to_geojson(markers_json: Mapping[int, Sequence[float]], basemap_homography: BasemapHomography) -> dict:
    if not markers_json:
        return {"type": "FeatureCollection", "features": []}

    marker_ids = list(markers_json.keys())
    positions = markers_json.values()
    pixel_points = np.array([[p[0], p[1]] for p in positions], dtype=np.float64)
    rotations = [p[2] for p in positions]

    # Project all markers in a single batched homography call instead of one
    # cv2.perspectiveTransform call per marker.
    utm_points = project_pixels_to_utm(basemap_homography, pixel_points)
    lon, lat = _UTM_TO_WGS84.transform(utm_points[:, 0], utm_points[:, 1])

    features = [
        {
            "type": "Feature",
            "properties": {"marker_id": marker_id, "rotation": rotation},
            "geometry": {"type": "Point", "coordinates": [lon_i, lat_i]},
        }
        for marker_id, rotation, lon_i, lat_i in zip(marker_ids, rotations, lon, lat)
    ]

    return {"type": "FeatureCollection", "features": features}
