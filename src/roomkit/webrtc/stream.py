# Vendored from gradio-app/fastrtc@v0.0.34 (Apache-2.0); see LICENSE in this
# directory. Trimmed to the headless WebRTC transport: the Gradio UI builder,
# the `gr.WebRTC` component, the colab/spaces helpers and `fastphone` tunneling
# were removed so roomkit ships WebRTC without depending on gradio.
import inspect
import logging
import re
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Literal,
    TypedDict,
    cast,
)

import anyio
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing_extensions import NotRequired

from .tracks import HandlerType, StreamHandlerImpl
from .utils import RTCConfigurationCallable
from .webrtc_connection_mixin import WebRTCConnectionMixin
from .websocket import WebSocketHandler

logger = logging.getLogger(__name__)

curr_dir = Path(__file__).parent


class Body(BaseModel):
    sdp: str | None = None
    candidate: dict[str, Any] | None = None
    type: str
    webrtc_id: str


class UIArgs(TypedDict):
    """
    UI customization arguments for the Gradio Blocks UI of the Stream class
    """

    title: NotRequired[str]
    """Title of the demo"""
    subtitle: NotRequired[str]
    """Subtitle of the demo. Text will be centered and displayed below the title."""
    icon: NotRequired[str]
    """Icon to display on the button instead of the wave animation. The icon should be a path/url to a .svg/.png/.jpeg file."""
    icon_button_color: NotRequired[str]
    """Color of the icon button. Default is var(--color-accent) of the demo theme."""
    pulse_color: NotRequired[str]
    """Color of the pulse animation. Default is var(--color-accent) of the demo theme."""
    icon_radius: NotRequired[int]
    """Border radius of the icon button expressed as a percentage of the button size. Default is 50%."""
    send_input_on: NotRequired[Literal["submit", "change"]]
    """When to send the input to the handler. Default is "change".
    If "submit", the input will be sent when the submit event is triggered by the user.
    If "change", the input will be sent whenever the user changes the input value.
    """
    hide_title: NotRequired[bool]
    """If True, the title and subtitle will not be displayed."""
    full_screen: NotRequired[bool]
    """If False, the component will be contained within its parent instead of full screen. Default is True."""


class Stream(WebRTCConnectionMixin):
    """
    Define an audio or video stream with a built-in UI, mountable on a FastAPI app.

    This class encapsulates the logic for handling real-time communication (WebRTC)
    streams, including setting up peer connections, managing tracks, generating
    a Gradio user interface, and integrating with FastAPI for API endpoints.
    It supports different modes (send, receive, send-receive) and modalities
    (audio, video, audio-video), and can optionally handle additional Gradio
    input/output components alongside the stream. It also provides functionality
    for telephone integration via the FastPhone method.

    Attributes:
        mode (Literal["send-receive", "receive", "send"]): The direction of the stream.
        modality (Literal["video", "audio", "audio-video"]): The type of media stream.
        rtp_params (dict[str, Any] | None): Parameters for RTP encoding.
        event_handler (HandlerType): The main function to process stream data.
        concurrency_limit (int): The maximum number of concurrent connections allowed.
        time_limit (float | None): Time limit in seconds for the event handler execution.
        allow_extra_tracks (bool): Whether to allow extra tracks beyond the specified modality.
        additional_output_components (list[Component] | None): Extra Gradio output components.
        additional_input_components (list[Component] | None): Extra Gradio input components.
        additional_outputs_handler (Callable | None): Handler for additional outputs.
        track_constraints (dict[str, Any] | None): Constraints for media tracks (e.g., resolution).
        webrtc_component (WebRTC): The underlying Gradio WebRTC component instance.
        rtc_configuration (dict[str, Any] | None): Configuration for the RTCPeerConnection (e.g., ICE servers).
        _ui (Blocks): The Gradio Blocks UI instance.
    """

    def __init__(
        self,
        handler: HandlerType,
        *,
        additional_outputs_handler: Callable | None = None,
        mode: Literal["send-receive", "receive", "send"] = "send-receive",
        modality: Literal["video", "audio", "audio-video"] = "video",
        concurrency_limit: int | None | Literal["default"] = "default",
        time_limit: float | None = None,
        allow_extra_tracks: bool = False,
        rtp_params: dict[str, Any] | None = None,
        rtc_configuration: RTCConfigurationCallable | None = None,
        server_rtc_configuration: dict[str, Any] | None = None,
        track_constraints: dict[str, Any] | None = None,
        additional_inputs: list[Any] | None = None,
        additional_outputs: list[Any] | None = None,
        ui_args: UIArgs | None = None,
        verbose: bool = True,
    ):
        """
        Initialize the Stream instance.

        Args:
            handler: The function to handle incoming stream data and return output data.
            additional_outputs_handler: An optional function to handle updates to additional output components.
            mode: The direction of the stream ('send', 'receive', or 'send-receive').
            modality: The type of media ('video', 'audio', or 'audio-video').
            concurrency_limit: Maximum number of concurrent connections. 'default' maps to 1.
            time_limit: Maximum execution time for the handler function in seconds.
            allow_extra_tracks: If True, allows connections with tracks not matching the modality.
            rtp_params: Optional dictionary of RTP encoding parameters.
            rtc_configuration: Optional Callable or dictionary for RTCPeerConnection configuration (e.g., ICE servers).
                               Required when deploying on Colab or Spaces.
            server_rtc_configuration: Optional dictionary for RTCPeerConnection configuration on the server side. Note
                                      that setting iceServers to be an empty list will mean no ICE servers will be used in the server.
            track_constraints: Optional dictionary of constraints for media tracks (e.g., resolution, frame rate).
            additional_inputs: Optional list of extra Gradio input components.
            additional_outputs: Optional list of extra Gradio output components. Requires `additional_outputs_handler`.
            ui_args: Optional dictionary to customize the default UI appearance (title, subtitle, icon, etc.).
            verbose: Whether to print verbose logging on startup.

        Raises:
            ValueError: If `additional_outputs` are provided without `additional_outputs_handler`.
        """
        WebRTCConnectionMixin.__init__(self)
        self.mode = mode
        self.modality = modality
        self.rtp_params = rtp_params
        self.event_handler = handler
        if ui_args and ui_args.get("variant") == "textbox" and hasattr(handler, "needs_args"):
            self.event_handler.needs_args = True  # type: ignore
        else:
            self.event_handler.needs_args = False  # type: ignore

        self.concurrency_limit = cast(
            (int),
            1 if concurrency_limit in ["default", None] else concurrency_limit,
        )
        self.time_limit = time_limit
        self.allow_extra_tracks = allow_extra_tracks
        self.additional_output_components = additional_outputs
        self.additional_input_components = additional_inputs
        self.additional_outputs_handler = additional_outputs_handler
        self.track_constraints = track_constraints
        self.rtc_configuration = rtc_configuration
        self.server_rtc_configuration = self.convert_to_aiortc_format(server_rtc_configuration)
        self.verbose = verbose

    def mount(self, app: FastAPI, path: str = "", tags: list[str | Enum] | None = None) -> None:
        """
        Mount the stream's API endpoints onto a FastAPI application.

        This method adds the necessary routes (`/webrtc/offer`, `/telephone/handler`,
        `/telephone/incoming`, `/websocket/offer`) to the provided FastAPI app,
        prefixed with the optional `path`. It also injects a startup message
        into the app's lifespan.

        Args:
            app: The FastAPI application instance.
            path: An optional URL prefix for the mounted routes.
            tags: Optional tags to FastAPI endpoints.
        """
        from fastapi import APIRouter

        router = APIRouter(prefix=path)
        router.post("/webrtc/offer", tags=tags)(self.offer)
        router.websocket("/telephone/handler")(self.telephone_handler)
        router.post("/telephone/incoming", tags=tags)(self.handle_incoming_call)
        router.websocket("/websocket/offer")(self.websocket_offer)
        lifespan = self._inject_startup_message(app.router.lifespan_context)
        app.router.lifespan_context = lifespan
        app.include_router(router)

    def _inject_startup_message(
        self, lifespan: Callable[[FastAPI], AbstractAsyncContextManager] | None = None
    ):
        """
        Create a FastAPI lifespan context manager to print startup messages and check environment.

        Args:
            lifespan: An optional existing lifespan context manager to wrap.

        Returns:
            An async context manager function suitable for `FastAPI(lifespan=...)`.
        """
        import contextlib

        import click

        def print_startup_message():
            if self.verbose:
                print(
                    click.style("INFO", fg="green")
                    + ":\t  Visit "
                    + click.style("https://fastrtc.org/userguide/api/", fg="cyan")
                    + " for WebRTC or Websocket API docs."
                )

        @contextlib.asynccontextmanager
        async def new_lifespan(app: FastAPI):
            if lifespan is None:
                print_startup_message()
                yield
            else:
                async with lifespan(app):
                    print_startup_message()
                    yield

        return new_lifespan

    async def offer(self, body: Body):
        """
        Handle an incoming WebRTC offer via HTTP POST.

        Processes the SDP offer and ICE candidates from the client to establish
        a WebRTC connection.

        Args:
            body: A Pydantic model containing the SDP offer, optional ICE candidate,
                  type ('offer'), and a unique WebRTC ID.

        Returns:
            A dictionary containing the SDP answer generated by the server.
        """
        return await self.handle_offer(
            body.model_dump(), set_outputs=self.set_additional_outputs(body.webrtc_id)
        )

    async def get_rtc_configuration(self):
        if inspect.isfunction(self.rtc_configuration):
            if inspect.iscoroutinefunction(self.rtc_configuration):
                return await self.rtc_configuration()
            else:
                return anyio.to_thread.run_sync(self.rtc_configuration)  # type: ignore
        else:
            return self.rtc_configuration

    async def handle_incoming_call(self, request: Request):
        """
        Handle incoming telephone calls (e.g., via Twilio).

        Generates TwiML instructions to connect the incoming call to the
        WebSocket handler (`/telephone/handler`) for audio streaming.

        Args:
            request: The FastAPI Request object for the incoming call webhook.

        Returns:
            An HTMLResponse containing the TwiML instructions as XML.
        """
        from twilio.twiml.voice_response import Connect, VoiceResponse

        response = VoiceResponse()
        response.say("Connecting to the AI assistant.")
        connect = Connect()
        path = request.url.path.removesuffix("/telephone/incoming")
        connect.stream(url=f"wss://{request.url.hostname}{path}/telephone/handler")
        response.append(connect)
        response.say("The call has been disconnected.")
        return HTMLResponse(content=str(response), media_type="application/xml")

    async def telephone_handler(self, websocket: WebSocket):
        """
        The websocket endpoint for streaming audio over Twilio phone.

        Args:
            websocket: The incoming WebSocket connection object.
        """
        handler = cast(StreamHandlerImpl, self.event_handler.copy())  # type: ignore
        handler.phone_mode = True

        async def set_handler(s: str, a: WebSocketHandler):
            if len(self.connections) >= self.concurrency_limit:  # type: ignore
                await cast(WebSocket, a.websocket).send_json(
                    {
                        "status": "failed",
                        "meta": {
                            "error": "concurrency_limit_reached",
                            "limit": self.concurrency_limit,
                        },
                    }
                )
                await websocket.close()
                return

        ws = WebSocketHandler(handler, set_handler, lambda s: None, lambda s: lambda a: None)
        await ws.handle_websocket(websocket)

    async def websocket_offer(self, websocket: WebSocket):
        """
        Handle WebRTC signaling over a WebSocket connection.

        Provides an alternative to the HTTP POST `/webrtc/offer` endpoint for
        exchanging SDP offers/answers and ICE candidates via WebSocket messages.

        Args:
            websocket: The incoming WebSocket connection object.
        """
        handler = cast(StreamHandlerImpl, self.event_handler.copy())  # type: ignore
        handler.phone_mode = False

        async def set_handler(s: str, a: WebSocketHandler):
            if len(self.connections) >= self.concurrency_limit:  # type: ignore
                await cast(WebSocket, a.websocket).send_json(
                    {
                        "status": "failed",
                        "meta": {
                            "error": "concurrency_limit_reached",
                            "limit": self.concurrency_limit,
                        },
                    }
                )
                await websocket.close()
                return

            self.connections[s] = [a]  # type: ignore

        def clean_up(s):
            self.clean_up(s)

        ws = WebSocketHandler(
            handler, set_handler, clean_up, lambda s: self.set_additional_outputs(s)
        )
        await ws.handle_websocket(websocket)
