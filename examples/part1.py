from typing import Any

from roomkit import RoomKit
from roomkit.core.inbound_router import InboundRoomRouter
from roomkit.models.enums import ChannelType


class SupportRouter(InboundRoomRouter):
    def __init__(self, db):
        self.db = db

    async def route(
        self,
        channel_id: str,
        channel_type: ChannelType,
        participant_id: str | None = None,
        channel_data: dict[str, Any] | None = None,
    ) -> str | None:
        # Look up existing open case by sender identity
        if participant_id:
            case = await self.db.find_open_case(participant_id)
            if case:
                return case.room_id

        # Return None to let RoomKit auto-create a new room
        return None


# Pass the router when creating the RoomKit instance
kit = RoomKit(inbound_router=SupportRouter(db=case_database))
