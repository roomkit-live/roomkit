"""The llama.cpp build the llamacpp provider downloads — generated, do not edit.

llama.cpp v1.0.0 (build b12359).

Regenerate with ``uv run python scripts/update_llamacpp_build.py``.
"""

# ruff: noqa: E501

from __future__ import annotations

BUILD = "b12359"

# variant -> ((archive, sha256), ...): the build, then its CUDA runtime if any.
ASSETS: dict[str, tuple[tuple[str, str], ...]] = {
    "linux-x64-cpu": (
        (
            "llama-b12359-bin-ubuntu-x64.tar.gz",
            "0bc1f091e8916102b6d8f2cf944233a620e90a0f856c72d0ce41150c65c2e72e",
        ),
    ),
    "linux-x64-cuda-12": (
        (
            "llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "50c937bec5bbd9f74c99138034f7403283ee21f1943f2429d7b1e4c079eafc11",
        ),
        (
            "cudart-llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz",
            "75a12d227f44a391608992329241765343bd143ee9445b90dc62e154ee726e5d",
        ),
    ),
    "linux-x64-cuda-13": (
        (
            "llama-b12359-bin-ubuntu-cuda-13.5-x64.tar.gz",
            "9440f417615bed0e191acef03bff0249117d2be74335358265a6d28475a9baf0",
        ),
        (
            "cudart-llama-b12359-bin-ubuntu-cuda-13.5-x64.tar.gz",
            "976a9146bb8a487a6ea701224d7fb14b05e494ab3c8f5c1799ffc4e5805d640c",
        ),
    ),
    "linux-x64-vulkan": (
        (
            "llama-b12359-bin-ubuntu-vulkan-x64.tar.gz",
            "19b60f00395a3f4954e46a5037695bc1d843404fc9d3124f82a612ecd8fbe9f0",
        ),
    ),
    "linux-arm64-cpu": (
        (
            "llama-b12359-bin-ubuntu-arm64.tar.gz",
            "dc7aa17a67884619b7223870aca9fd6addbc01717713b04a714add1c31137359",
        ),
    ),
    "linux-arm64-cuda-13": (
        (
            "llama-b12359-bin-ubuntu-cuda-13.5-arm64.tar.gz",
            "e12fd3ca9a599349cac5ec767825e17f8cab9dc2dab8d8e5b85d50ce21ef6d7d",
        ),
        (
            "cudart-llama-b12359-bin-ubuntu-cuda-13.5-arm64.tar.gz",
            "998f62318999710d398dad40821d1d037731f360791e41628d9dfcde207909d1",
        ),
    ),
    "macos-arm64": (
        (
            "llama-b12359-bin-macos-arm64.tar.gz",
            "82b00abc740160e1b9980e6cc79c117345fa34af058c80e6c0a69aa8ddccde75",
        ),
    ),
    "macos-x64": (
        (
            "llama-b12359-bin-macos-x64.tar.gz",
            "25320d529077ef5a3d16e7b5da26438f221d88c8c0930a32b0c79457f5eede56",
        ),
    ),
    "windows-x64-cpu": (
        (
            "llama-b12359-bin-win-cpu-x64.zip",
            "837c1655e87df3b66989f4be33a792a1399c785bf259038dc75c900096683e69",
        ),
    ),
    "windows-x64-cuda-12": (
        (
            "llama-b12359-bin-win-cuda-12.4-x64.zip",
            "e9c0abf442e8339c06728366c1ad2c711332f9939c0c00603fb75d85fd44c914",
        ),
        (
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
            "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
        ),
    ),
    "windows-x64-cuda-13": (
        (
            "llama-b12359-bin-win-cuda-13.5-x64.zip",
            "1de9ca1ff408d6e348ea0007c6880ad2393311a02e7799bcde26dc16daef03dd",
        ),
        (
            "cudart-llama-bin-win-cuda-13.5-x64.zip",
            "c2bf4cc202bfbaaa7a0b75f5082457c3811326947bb216059b72839d844e9cba",
        ),
    ),
    "windows-arm64-cpu": (
        (
            "llama-b12359-bin-win-cpu-arm64.zip",
            "d27564c754478d2832789dfcebfc5ace018e36a224c41d810b7cbc1ba8243ec0",
        ),
    ),
}
