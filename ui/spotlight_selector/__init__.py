from pathlib import Path

import streamlit.components.v1 as components


_COMPONENT_DIR = Path(__file__).resolve().parent / "frontend"


spotlight_selector = components.declare_component(
    "cropai_spotlight_selector",
    path=str(_COMPONENT_DIR),
)


def select_spotlight(
    image_url,
    regions=None,
    active_index=None,
    active_region=None,
    key=None,
):
    if regions is None:
        regions = []

    if active_index is None or not isinstance(active_index, int) or active_index < 0:
        active_index = -1
        if active_region is not None and regions:
            try:
                active_index = regions.index(active_region)
            except ValueError:
                active_index = len(regions) - 1
    elif regions and active_index >= len(regions):
        active_index = len(regions) - 1

    return spotlight_selector(
        image_url=image_url,
        regions=regions,
        active_index=active_index,
        key=key,
        default={
            "regions": regions,
            "active_index": active_index,
            "active": regions[active_index] if 0 <= active_index < len(regions) else active_region,
            "action": None,
        },
    )