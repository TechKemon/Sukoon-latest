from __future__ import annotations

import logging

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai


logger = logging.getLogger("myagent")
logger.setLevel(logging.INFO)

model=openai.realtime.RealtimeModel(
    voice="alloy",
    temperature=0.8,
    instructions="You are a helpful assistant",
    turn_detection=openai.realtime.ServerVadOptions(
        threshold=0.6, prefix_padding_ms=200, silence_duration_ms=500
    ),
)
agent = multimodal.MultimodalAgent(
    model=model
)
agent.start(ctx.room)


######
async def entrypoint(ctx: JobContext):
    logger.info("starting entrypoint")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    model = openai.realtime.RealtimeModel(
        instructions="You are a helpful assistant and you love kittens",
        voice="shimmer",
        temperature=0.8,
        modalities=["audio", "text"],
    )
    assistant = MultimodalAgent(model=model)
    assistant.start(ctx.room)

    logger.info("starting agent")

    session = model.sessions[0]
    session.conversation.item.create(
      llm.ChatMessage(
        role="assistant",
        content="Please begin the interaction with the user in a manner consistent with your instructions.",
      )
    )
    session.response.create()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

