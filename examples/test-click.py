"""
vision_click.py — Cliquer sur des éléments UI via Gemini Vision
Cross-platform : Linux, macOS (Retina), Windows (HiDPI)

Dépendances :
    pip install google-genai mss pyautogui pillow

Usage :
    from vision_click import VisionClick

    vc = VisionClick(api_key="VOTRE_CLE_GEMINI")
    vc.click("icône Google Chrome dans la barre latérale")
    vc.double_click("dossier Documents sur le bureau")
    vc.right_click("barre de titre du terminal")
    x, y = vc.find("bouton Envoyer")  # sans cliquer
"""

import io
import json
import os
import platform
import re
import sys
import time

import mss
import pyautogui
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("pip install google-genai")


# ---------------------------------------------------------------------------
# Détection DPI / scale par OS
# ---------------------------------------------------------------------------


def _get_scale_factor() -> tuple[float, float]:
    system = platform.system()
    if system == "Darwin":
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            phys_w = monitor["width"]
            phys_h = monitor["height"]
        logical_w, logical_h = pyautogui.size()
        return logical_w / phys_w, logical_h / phys_h
    elif system == "Windows":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        return 1.0, 1.0
    else:
        gdk_scale = os.environ.get("GDK_SCALE", "1")
        try:
            s = float(gdk_scale)
            return 1.0 / s, 1.0 / s
        except ValueError:
            return 1.0, 1.0


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------


def _take_screenshot() -> tuple[bytes, int, int]:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), sct_img.width, sct_img.height


# ---------------------------------------------------------------------------
# Prompt Gemini
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Tu es un assistant spécialisé dans la localisation d'éléments graphiques d'interface utilisateur (GUI).
Tu analyses des screenshots de bureau et tu identifies des éléments VISUELS : icônes, boutons, fenêtres, widgets.

Règles importantes :
- Tu cherches des ÉLÉMENTS GRAPHIQUES (icônes, boutons, images), PAS du texte affiché dans les applications.
- Si la description mentionne une icône ou un logo, cherche la représentation graphique, pas le mot écrit.
- Les coordonnées doivent TOUJOURS être en pixels absolus dans l'image fournie.
- Retourne UNIQUEMENT un objet JSON valide, sans markdown, sans explication.
"""

# NOTE: on injecte img_w et img_h dans le prompt ET dans les contraintes
# pour forcer le modèle à rester dans l'espace pixel absolu.
_USER_PROMPT_TEMPLATE = """\
Image size: {w}x{h} pixels.

Élément à trouver : "{element}"

Consignes :
- Cherche un ÉLÉMENT VISUEL (icône, bouton, widget graphique) correspondant à la description.
- Ignore le texte affiché dans les fenêtres d'application (terminal, éditeur, navigateur, gestionnaire de fichiers).
- Concentre-toi sur les barres d'outils, barre des tâches, barre latérale, bureau, menus système.
- IMPORTANT : cx doit être un entier entre 0 et {w}. cy doit être un entier entre 0 et {h}.
- IMPORTANT : NE PAS normaliser les coordonnées entre 0 et 1. Retourner des pixels absolus.

Retourne UNIQUEMENT ce JSON (pas de markdown, pas de texte autour) :
{{"found": true, "cx": <pixels absolus x>, "cy": <pixels absolus y>, "box": {{"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>}}, "label": "<nom visible>"}}

Si introuvable :
{{"found": false, "cx": 0, "cy": 0, "box": {{"x1": 0, "y1": 0, "x2": 0, "y2": 0}}, "label": ""}}
"""


def _parse_gemini_response(text: str) -> dict:
    """Extrait le JSON de la réponse Gemini même s'il y a du texte autour."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Impossible de parser la réponse Gemini : {text!r}")


def _denormalize_if_needed(result: dict, img_w: int, img_h: int) -> dict:
    """
    Détecte si Gemini a retourné des coords normalisées (0-1) au lieu de pixels
    et les convertit automatiquement.
    Certains modèles ignorent les consignes et normalisent quand même.
    """
    cx, cy = result.get("cx", 0), result.get("cy", 0)
    # Si les deux valeurs sont entre 0 et 1 → normalisées
    if 0 <= cx <= 1 and 0 <= cy <= 1 and (cx > 0 or cy > 0):
        result["cx"] = int(cx * img_w)
        result["cy"] = int(cy * img_h)
        b = result.get("box", {})
        result["box"] = {
            "x1": int(b.get("x1", 0) * img_w),
            "y1": int(b.get("y1", 0) * img_h),
            "x2": int(b.get("x2", 0) * img_w),
            "y2": int(b.get("y2", 0) * img_h),
        }
    return result


# ---------------------------------------------------------------------------
# Classe principale
# ---------------------------------------------------------------------------


class VisionClick:
    """
    Localise et clique sur des éléments UI décrits en langage naturel,
    en utilisant Gemini Vision pour trouver les coordonnées.

    Paramètres
    ----------
    api_key : str, optionnel
        Clé API Gemini. Si absent, lit GEMINI_API_KEY dans l'environnement.
        N'utilise PAS GOOGLE_API_KEY pour éviter les conflits.
    model : str
        Modèle Gemini à utiliser. Défaut : "gemini-3.1-flash-image-preview".
    pause : float
        Délai (secondes) après l'action pyautogui. Défaut : 0.1.
    debug : bool
        Si True, affiche les coords et sauvegarde le screenshot annoté.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.1-flash-image-preview",
        pause: float = 0.1,
        debug: bool = False,
    ):
        # Priorité explicite : paramètre > GEMINI_API_KEY (jamais GOOGLE_API_KEY)
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Fournir api_key= ou définir GEMINI_API_KEY "
                "(GOOGLE_API_KEY est ignoré volontairement)"
            )
        self._client = genai.Client(api_key=key)
        self._model_name = model
        self.pause = pause
        self.debug = debug
        self._scale_x, self._scale_y = _get_scale_factor()
        if self.debug:
            print(
                f"[VisionClick] OS={platform.system()}, model={model}, scale=({self._scale_x:.3f}, {self._scale_y:.3f})"
            )

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def find(self, element: str) -> tuple[int, int]:
        """
        Trouve l'élément décrit et retourne ses coordonnées écran (x, y).
        Lève ElementNotFoundError si introuvable.
        """
        png_bytes, img_w, img_h = _take_screenshot()
        result = self._ask_gemini(element, png_bytes, img_w, img_h)
        result = _denormalize_if_needed(result, img_w, img_h)

        if not result.get("found"):
            raise ElementNotFoundError(f"Élément introuvable : {element!r}")

        cx_img = result["cx"]
        cy_img = result["cy"]

        screen_x = int(cx_img * self._scale_x)
        screen_y = int(cy_img * self._scale_y)

        if self.debug:
            print(f"[VisionClick] '{element}'")
            print(f"  image coords  : ({cx_img}, {cy_img})")
            print(f"  screen coords : ({screen_x}, {screen_y})")
            print(f"  label         : {result.get('label')!r}")
            print(f"  box           : {result.get('box')}")
            self._save_debug_image(png_bytes, result, element)

        return screen_x, screen_y

    def click(self, element: str) -> tuple[int, int]:
        """Click gauche sur l'élément décrit."""
        x, y = self.find(element)
        pyautogui.click(x, y)
        time.sleep(self.pause)
        return x, y

    def double_click(self, element: str) -> tuple[int, int]:
        """Double-click sur l'élément décrit."""
        x, y = self.find(element)
        pyautogui.doubleClick(x, y)
        time.sleep(self.pause)
        return x, y

    def right_click(self, element: str) -> tuple[int, int]:
        """Click droit sur l'élément décrit."""
        x, y = self.find(element)
        pyautogui.rightClick(x, y)
        time.sleep(self.pause)
        return x, y

    def move_to(self, element: str, duration: float = 0.3) -> tuple[int, int]:
        """Déplace la souris sur l'élément sans cliquer."""
        x, y = self.find(element)
        pyautogui.moveTo(x, y, duration=duration)
        return x, y

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------

    def _ask_gemini(self, element: str, png_bytes: bytes, img_w: int, img_h: int) -> dict:
        prompt = _USER_PROMPT_TEMPLATE.format(w=img_w, h=img_h, element=element)
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[
                types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
            ),
        )
        return _parse_gemini_response(response.text)

    def _save_debug_image(self, png_bytes: bytes, result: dict, label: str):
        """Sauvegarde le screenshot avec le bounding box et le centre annotés."""
        try:
            from PIL import ImageDraw

            img = Image.open(io.BytesIO(png_bytes))
            draw = ImageDraw.Draw(img)
            b = result["box"]
            draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline="red", width=3)
            cx, cy = result["cx"], result["cy"]
            r = 8
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="red")
            # Crosshair
            draw.line([cx - 20, cy, cx + 20, cy], fill="yellow", width=2)
            draw.line([cx, cy - 20, cx, cy + 20], fill="yellow", width=2)
            safe_label = re.sub(r"[^\w]", "_", label)[:40]
            path = f"debug_vision_{safe_label}.png"
            img.save(path)
            print(f"[VisionClick] debug image → {path}")
        except Exception as e:
            print(f"[VisionClick] debug image failed: {e}")


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ElementNotFoundError(Exception):
    """L'élément demandé n'a pas été trouvé dans le screenshot."""

    pass


# ---------------------------------------------------------------------------
# CLI rapide pour tester
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python vision_click.py "description de l\'élément"')
        print('       python vision_click.py "description" --right')
        print('       python vision_click.py "description" --double')
        print('       python vision_click.py "description" --find')
        sys.exit(1)

    element = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "--click"

    vc = VisionClick(debug=True)

    if mode == "--right":
        x, y = vc.right_click(element)
    elif mode == "--double":
        x, y = vc.double_click(element)
    elif mode == "--find":
        x, y = vc.find(element)
        print(f"Trouvé à : ({x}, {y})")
        sys.exit(0)
    else:
        x, y = vc.click(element)

    print(f"Action '{mode}' effectuée à ({x}, {y})")
