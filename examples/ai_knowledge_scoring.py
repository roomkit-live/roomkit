"""Knowledge retrieval and response scoring example.

Demonstrates:
1. KnowledgeSource — pluggable retrieval backend
2. RetrievalMemory — enriches AI context with external knowledge
3. ConversationScorer — automatic quality scoring
4. ScoringHook — wires scorers to the AFTER_AI_RESPONSE hook
5. kit.submit_feedback() — user quality ratings

Run with:
    uv run python examples/ai_knowledge_scoring.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from roomkit import (
    ChannelCategory,
    InboundMessage,
    RoomEvent,
    RoomKit,
    TextContent,
    WebSocketChannel,
)
from roomkit.channels.ai import AIChannel
from roomkit.knowledge import KnowledgeResult, KnowledgeSource
from roomkit.memory import RetrievalMemory, SlidingWindowMemory
from roomkit.providers.ai.mock import MockAIProvider
from roomkit.scoring import ConversationScorer, Score, ScoringHook

# ---------------------------------------------------------------------------
# 1. Custom knowledge source (simulates a FAQ database)
# ---------------------------------------------------------------------------


class FAQKnowledgeSource(KnowledgeSource):
    """In-memory FAQ search — replace with vector DB in production."""

    def __init__(self) -> None:
        self._faqs = [
            (
                "How do I get a refund?",
                "Refunds are available within 30 days of purchase. Contact support@example.com.",
            ),
            (
                "What payment methods do you accept?",
                "We accept Visa, Mastercard, PayPal, and bank transfer.",
            ),
            (
                "How do I reset my password?",
                "Click 'Forgot Password' on the login page. A reset link will be sent to your email.",
            ),
            ("What are your business hours?", "We are available Monday-Friday, 9am-5pm EST."),
            (
                "Do you offer free shipping?",
                "Free shipping on orders over $50. Standard shipping is $5.99.",
            ),
        ]

    @property
    def name(self) -> str:
        return "FAQSource"

    async def search(
        self, query: str, *, room_id: str | None = None, limit: int = 5
    ) -> list[KnowledgeResult]:
        # Simple keyword matching — replace with embeddings in production
        query_lower = query.lower()
        results = []
        for question, answer in self._faqs:
            words = set(query_lower.split())
            match_words = set(question.lower().split()) & words
            if match_words:
                score = len(match_words) / len(words) if words else 0
                results.append(
                    KnowledgeResult(
                        content=f"Q: {question}\nA: {answer}",
                        score=score,
                        source="faq",
                    )
                )
        return sorted(results, key=lambda r: r.score, reverse=True)[:limit]


# ---------------------------------------------------------------------------
# 2. Custom scorer (rule-based quality check)
# ---------------------------------------------------------------------------


class LengthAndRelevanceScorer(ConversationScorer):
    """Simple rule-based scorer — checks response length and keyword overlap."""

    @property
    def name(self) -> str:
        return "LengthAndRelevanceScorer"

    async def score(
        self,
        *,
        response_content: str,
        query: str,
        room_id: str,
        channel_id: str,
        usage: dict[str, Any] | None = None,
        thinking: str = "",
    ) -> list[Score]:
        scores = []

        # Length score: penalize very short or very long responses
        length = len(response_content)
        if length < 20:
            scores.append(Score(value=0.3, dimension="length", reason="Too short"))
        elif length > 2000:
            scores.append(Score(value=0.5, dimension="length", reason="Too long"))
        else:
            scores.append(Score(value=0.9, dimension="length", reason="Good length"))

        # Relevance score: keyword overlap between query and response
        query_words = set(query.lower().split())
        response_words = set(response_content.lower().split())
        overlap = query_words & response_words
        relevance = len(overlap) / len(query_words) if query_words else 0
        scores.append(
            Score(
                value=min(1.0, relevance + 0.3),  # baseline 0.3
                dimension="relevance",
                reason=f"{len(overlap)} keyword overlaps",
            )
        )

        return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    kit = RoomKit()

    ws = WebSocketChannel("ws-user")
    kit.register_channel(ws)

    inbox: list[RoomEvent] = []

    async def on_recv(_conn: str, event: RoomEvent) -> None:
        inbox.append(event)

    ws.register_connection("user-conn", on_recv)

    # --- Set up knowledge-augmented AI with scoring ---
    provider = MockAIProvider(
        responses=[
            "Based on our FAQ, refunds are available within 30 days of purchase. "
            "Please contact support@example.com to initiate the process.",
        ]
    )

    memory = RetrievalMemory(
        sources=[FAQKnowledgeSource()],
        inner=SlidingWindowMemory(max_events=50),
        max_results=3,
    )

    ai = AIChannel("ai-support", provider=provider, memory=memory)
    kit.register_channel(ai)

    # --- Attach scoring ---
    scorer = LengthAndRelevanceScorer()
    hook = ScoringHook(scorers=[scorer])
    hook.attach(kit)

    # --- Create room and send message ---
    await kit.create_room(room_id="support-room")
    await kit.attach_channel("support-room", "ws-user")
    await kit.attach_channel("support-room", "ai-support", category=ChannelCategory.INTELLIGENCE)

    print("=== Knowledge Retrieval + Response Scoring ===\n")

    # User asks a question
    print("User: How do I get a refund?\n")
    await kit.process_inbound(
        InboundMessage(
            channel_id="ws-user",
            sender_id="user-1",
            content=TextContent(body="How do I get a refund?"),
        )
    )
    await asyncio.sleep(0.3)

    # Show AI response
    for event in inbox:
        if isinstance(event.content, TextContent) and event.content.body:
            print(f"AI: {event.content.body}\n")

    # Show what the AI provider received (knowledge was injected)
    if provider.calls:
        ctx = provider.calls[0]
        print("--- Knowledge Injected into Context ---")
        for msg in ctx.messages:
            if isinstance(msg.content, str) and "knowledge sources" in msg.content:
                print(msg.content)
                print()

    # Show scores
    print("--- Quality Scores ---")
    for score_entry in hook.recent_scores:
        print(
            f"  [{score_entry['dimension']}] {score_entry['value']:.2f} — {score_entry['reason']}"
        )

    # Submit user feedback
    await kit.submit_feedback(
        "support-room",
        rating=0.95,
        comment="Very helpful, exactly what I needed",
        dimension="helpfulness",
    )

    # Show stored observations (scores + feedback)
    obs = await kit._store.list_observations("support-room")
    print(f"\n--- Stored Observations ({len(obs)}) ---")
    for o in obs:
        print(f"  [{o.category}] confidence={o.confidence:.2f}: {o.content}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
