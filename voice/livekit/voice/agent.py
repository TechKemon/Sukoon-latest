from __future__ import annotations

import logging
from dotenv import load_dotenv

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


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    run_multimodal_agent(ctx, participant)

    logger.info("agent started")


def run_multimodal_agent(ctx: JobContext, participant: rtc.RemoteParticipant):
    logger.info("starting multimodal agent")

    model = openai.realtime.RealtimeModel(
        instructions=(
            "You are a supportive friend who's here to listen and chat about mental health concerns. "
            "Speak naturally and conversationally, as if talking to a close friend. "
            "If the user speaks in Hindi or Marathi, respond in the same language they use. "
            "For Hindi/Marathi conversations, use casual, friendly terms like 'dost' or 'मित्रा' as appropriate. "
            "Common greetings to use: "
            "- Hindi: 'कैसे हो?', 'क्या हाल है?', 'सब ठीक?' "
            "- Marathi: 'कसे आहात?', 'काय चाललंय?', 'सगळं ठीक आहे का?' "
            "Show genuine care and empathy in your responses. "
            "Use a warm, friendly tone while being mindful of emotional cues in their voice. "
            "Keep responses brief and natural - like a real conversation. "
            "If they share difficult emotions or experiences, validate their feelings and offer gentle support. "
            "For serious concerns, kindly suggest professional help while maintaining the friendly dynamic. "
            "Remember to: "
            "- Listen actively and reflect their emotions "
            "- Use casual, friendly language in English/Hindi/Marathi "
            "- Share brief, supportive responses "
            "- Be genuine and warm "
            "- Stay within mental health support boundaries"
        ),
        modalities=["audio", "text"],
    )

    chat_ctx = llm.ChatContext()
    chat_ctx.append(
        text="Context about the user: you are talking to someone who needs a friendly ear for mental health support. "
        "They may speak in English, Hindi, or Marathi. Match their language choice. "
        "Start with a warm, casual greeting and ask how they're doing today.",
        role="assistant",
    )

    agent = MultimodalAgent(
        model=model,
        chat_ctx=chat_ctx,
    )
    agent.start(ctx.room, participant)

    # to enable the agent to speak first
    agent.generate_reply()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
