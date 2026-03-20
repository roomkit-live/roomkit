"""OmniView client for precise UI element detection via GPU service.

Wraps the OmniView API (YOLO + EasyOCR + Florence-2) and applies DPI
scaling so coordinates work correctly on Retina / HiDPI displays.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

logger = logging.getLogger("screen_assistant.omniview")

# -- Tool definitions exposed to the realtime voice model -------------------

OBSERVE_TOOL: dict[str, object] = {
    "name": "observe",
    "description": (
        "Detect ALL UI elements on screen with bounding boxes and IDs. "
        "Returns numbered elements you can click with click_result(element_id). "
        "Use this for precise element detection before clicking."
    ),
    "parameters": {"type": "object", "properties": {}},
}

CLICK_RESULT_TOOL: dict[str, object] = {
    "name": "click_result",
    "description": (
        "Click a UI element by its ID from the last observe() call. "
        "More precise than click_element — uses exact bounding box coordinates. "
        "Pass element_id from observe results. Optionally set button or double-click."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "element_id": {
                "type": "integer",
                "description": "Element ID from the last observe() result.",
            },
            "button": {
                "type": "string",
                "description": "Mouse button: left (default), right, middle.",
                "enum": ["left", "right", "middle"],
            },
            "double": {
                "type": "boolean",
                "description": "Double-click if true (default: false).",
            },
        },
        "required": ["element_id"],
    },
}


class OmniViewClient:
    """Client for the OmniView API (screenshot -> UI elements with bboxes).

    Applies DPI scaling so clicks land at the correct logical coordinates
    on Retina / HiDPI displays.
    """

    def __init__(self, base_url: str, monitor: int = 1) -> None:
        self.base_url = base_url.rstrip("/")
        self.monitor = monitor
        self.last_elements: list[dict[str, object]] = []

    # -- Screen capture -----------------------------------------------------

    def _capture_b64(self) -> str | None:
        """Capture screen as base64 PNG."""
        import base64
        import io

        from roomkit.video.vision.screen_tool import capture_screen_frame

        frame = capture_screen_frame(self.monitor)
        if frame is None:
            return None
        from PIL import Image

        img = Image.frombytes("RGB", (frame.width, frame.height), frame.data)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    # -- DPI-safe clicking --------------------------------------------------

    @staticmethod
    def _scale_coords(cx: int, cy: int) -> tuple[int, int]:
        """Apply DPI scale factor to convert physical -> logical coords."""
        from roomkit.video.vision.screen_input import _get_scale_factor

        sx, sy = _get_scale_factor()
        return int(cx * sx), int(cy * sy)

    def click_at(
        self,
        cx: int,
        cy: int,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        """Click at physical pixel coords, applying DPI correction."""
        import pyautogui  # type: ignore[import-untyped]

        pyautogui.FAILSAFE = False
        lx, ly = self._scale_coords(cx, cy)
        logger.info("click_at phys=(%d,%d) -> logical=(%d,%d)", cx, cy, lx, ly)
        pyautogui.click(lx, ly, button=button, clicks=clicks)

    # -- API calls ----------------------------------------------------------

    async def parse(self) -> dict[str, object]:
        """Capture screen -> OmniView /parse -> all elements."""
        import urllib.request

        b64 = self._capture_b64()
        if b64 is None:
            return {"status": "error", "error": "No screen frame"}

        req_body = json.dumps({"image": b64}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/parse",
            data=req_body,
            headers={"Content-Type": "application/json"},
        )
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(req, timeout=30),  # noqa: ASYNC210
        )
        result = json.loads(resp.read())
        self.last_elements = result.get("elements", [])
        return result

    async def locate(self, query: str) -> dict[str, object]:
        """Capture screen -> OmniView /locate -> best matching element."""
        import urllib.request

        b64 = self._capture_b64()
        if b64 is None:
            return {"found": False, "error": "No screen frame"}

        req_body = json.dumps({"image": b64, "query": query}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/locate",
            data=req_body,
            headers={"Content-Type": "application/json"},
        )
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(req, timeout=30),  # noqa: ASYNC210
        )
        return json.loads(resp.read())

    def get_element_by_id(self, element_id: int) -> dict[str, object] | None:
        """Get an element from the last parse result by ID."""
        for el in self.last_elements:
            if el.get("id") == element_id:
                return el
        return None

    # -- Convenience --------------------------------------------------------

    async def locate_and_click(
        self,
        query: str,
        button: str = "left",
        double: bool = False,
    ) -> dict[str, object]:
        """Locate an element by description and click it (DPI-safe).

        Returns the /locate response with an added ``clicked`` key.
        """
        result = await self.locate(query)
        if not result.get("found"):
            return result
        center = result.get("center", [0, 0])
        cx, cy = int(center[0]), int(center[1])
        self.click_at(cx, cy, button=button, clicks=2 if double else 1)
        result["clicked"] = True
        return result

    @staticmethod
    def clean_ocr(text: str) -> str:
        """Fix common OCR artifacts from EasyOCR."""
        text = re.sub(r"Jl(?=www|[a-z])", "//", text)
        text = re.sub(r"https?\s*:\s*//", "https://", text)
        text = text.replace(" .", ".").replace(". ", ".")
        return text
