"""
Social Media Content Agent using CrewAI + Groq.

Generates platform-optimized content for:
- Twitter/X
- LinkedIn
- Instagram

Uses Groq-hosted GPT-OSS 120B.
"""

import argparse
import os
from typing import Optional

# ============================================================
# LITELLM / GROQ COMPATIBILITY PATCH
# ============================================================

try:
    import litellm

    # Ignore unsupported parameters passed by CrewAI/LiteLLM
    litellm.drop_params = True

    _orig_litellm_completion = litellm.completion
    _orig_litellm_acompletion = litellm.acompletion

    def _strip_cache_breakpoints(messages):
        """Remove cache_breakpoint parameters unsupported by Groq."""

        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    message.pop("cache_breakpoint", None)

        elif isinstance(messages, dict):
            messages.pop("cache_breakpoint", None)

    def _safe_completion(*args, **kwargs):
        """Safe wrapper for synchronous LiteLLM calls."""

        if "messages" in kwargs:
            _strip_cache_breakpoints(kwargs["messages"])

        elif len(args) > 1 and isinstance(args[1], list):
            _strip_cache_breakpoints(args[1])

        return _orig_litellm_completion(*args, **kwargs)

    async def _safe_acompletion(*args, **kwargs):
        """Safe wrapper for asynchronous LiteLLM calls."""

        if "messages" in kwargs:
            _strip_cache_breakpoints(kwargs["messages"])

        elif len(args) > 1 and isinstance(args[1], list):
            _strip_cache_breakpoints(args[1])

        return await _orig_litellm_acompletion(*args, **kwargs)

    litellm.completion = _safe_completion
    litellm.acompletion = _safe_acompletion

except ImportError:
    pass


# ============================================================
# CREWAI CACHE BREAKPOINT PATCH
# ============================================================

try:
    import crewai.llms.cache as _c

    _c.mark_cache_breakpoint = lambda msg: msg

except Exception:
    pass


try:
    import crewai.agents.crew_agent_executor as _cae

    _cae.mark_cache_breakpoint = lambda msg: msg

except Exception:
    pass


try:
    import crewai.experimental.agent_executor as _eae

    _eae.mark_cache_breakpoint = lambda msg: msg

except Exception:
    pass


# ============================================================
# IMPORT CREWAI
# ============================================================

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv


# Load variables from .env
load_dotenv()


# ============================================================
# BUILD GROQ LLM
# ============================================================

def build_llm(
    model: str = "openai/gpt-oss-120b",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
) -> LLM:
    """
    Create a CrewAI LLM using Groq only.

    The model is hosted by Groq.
    """

    # Get API key from argument or .env
    key = api_key or os.getenv("GROQ_API_KEY")

    # Check API key
    if not key:
        raise ValueError(
            "Groq API key is missing.\n"
            "Please add GROQ_API_KEY to your .env file."
        )

    # Store key in environment
    os.environ["GROQ_API_KEY"] = key

    # Make sure model has the Groq prefix
    if not model.startswith("groq/"):
        model = f"groq/{model}"

    # Create CrewAI LLM
    return LLM(
        model=model,
        api_key=key,
        temperature=temperature,
    )


# ============================================================
# GENERATE SOCIAL MEDIA CONTENT
# ============================================================

def generate_social_content(
    topic: str,
    brand: str = "",
    platforms: Optional[list[str]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:

    # Default platforms
    if platforms is None:
        platforms = [
            "twitter",
            "linkedin",
            "instagram",
        ]

    # --------------------------------------------------------
    # CREATE LLM
    # --------------------------------------------------------

    llm = build_llm(
        model=model or "openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.7,
    )

    # ========================================================
    # AGENT 1: SOCIAL MEDIA STRATEGIST
    # ========================================================

    strategist = Agent(
        role="Social Media Strategist",

        goal=(
            "Analyze the topic and create a clear social media "
            "strategy for each requested platform."
        ),

        backstory=(
            "You are an experienced social media strategist "
            "specializing in audience engagement, content strategy, "
            "platform-specific communication, hooks, and hashtags."
        ),

        llm=llm,

        verbose=False,
    )

    # ========================================================
    # AGENT 2: SOCIAL MEDIA COPYWRITER
    # ========================================================

    writer = Agent(
        role="Social Media Copywriter",

        goal=(
            "Create engaging, platform-optimized social media "
            "content that matches the strategy."
        ),

        backstory=(
            "You are an expert social media copywriter who understands "
            "Twitter/X, LinkedIn, and Instagram formats, hooks, "
            "storytelling, hashtags, and audience engagement."
        ),

        llm=llm,

        verbose=False,
    )

    # ========================================================
    # TASK 1: CREATE STRATEGY
    # ========================================================

    strategy_task = Task(

        description=f"""
Analyze the following topic for social media marketing.

Topic:
{topic}

Brand:
{brand or "Not specified"}

Platforms:
{", ".join(platforms)}

Create a social media strategy containing:

1. Core message
2. Target audience
3. Emotional hook
4. Tone of voice
5. Key points to communicate
6. Five relevant hashtags

Make the strategy practical and platform-aware.
""",

        agent=strategist,

        expected_output=(
            "A clear content strategy containing "
            "the core message, audience, emotional hook, "
            "tone, key points, and hashtags."
        ),
    )

    # ========================================================
    # TASK 2: WRITE CONTENT
    # ========================================================

    writing_task = Task(

        description=f"""
Create social media content for the following topic.

Topic:
{topic}

Brand:
{brand or "General"}

Platforms:
{", ".join(platforms)}

Use the strategy created by the Social Media Strategist.

IMPORTANT:

For Twitter/X:

- Create 2 tweet variations.
- Each tweet must be under 280 characters.
- Make them short, punchy, and engaging.
- Also create a thread opener.

For LinkedIn:

- Create one professional post.
- 150–200 words.
- Start with a strong storytelling hook.
- Use short paragraphs.
- Make it informative and professional.
- End with an engaging question or call to action.

For Instagram:

- Create one caption.
- 100–150 words.
- Make it visual and engaging.
- Use an appropriate call to action.
- Add 15 relevant hashtags.

Make every platform version unique.

Do NOT simply copy the same content across platforms.

Return the final answer with clear headings:

TWITTER/X
LINKEDIN
INSTAGRAM
""",

        agent=writer,

        expected_output=(
            "Platform-optimized social media content for "
            "Twitter/X, LinkedIn, and Instagram."
        ),

        context=[strategy_task],
    )

    # ========================================================
    # CREATE CREW
    # ========================================================

    crew = Crew(

        agents=[
            strategist,
            writer,
        ],

        tasks=[
            strategy_task,
            writing_task,
        ],

        process=Process.sequential,

        verbose=False,
    )

    # ========================================================
    # RUN CREW
    # ========================================================

    result = crew.kickoff()

    return str(result)


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Social Media Content Agent "
            "(CrewAI + Groq)"
        )
    )

    # --------------------------------------------------------
    # TOPIC
    # --------------------------------------------------------

    parser.add_argument(
        "--topic",
        default=(
            "How AI is transforming "
            "software development in 2026"
        ),
        help="Content topic",
    )

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    parser.add_argument(
        "--brand",
        default="",
        help="Brand name (optional)",
    )

    # --------------------------------------------------------
    # PLATFORMS
    # --------------------------------------------------------

    parser.add_argument(
        "--platforms",
        default="twitter,linkedin,instagram",
        help="Comma-separated platforms",
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        default="openai/gpt-oss-120b",
        help="Groq-hosted model",
    )

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    parser.add_argument(
        "--api-key",
        default=None,
        help="Groq API key",
    )

    # Parse arguments
    args = parser.parse_args()

    # Convert platform string to list
    platforms = [
        platform.strip()
        for platform in args.platforms.split(",")
    ]

    # ========================================================
    # DISPLAY SETTINGS
    # ========================================================

    print()
    print("=" * 60)
    print("🤖 SOCIAL MEDIA CONTENT AGENT")
    print("=" * 60)

    print(f"⚡ Provider  : GROQ")
    print(f"🤖 Model     : {args.model}")
    print(f"📱 Platforms : {', '.join(platforms)}")
    print(f"📌 Topic     : {args.topic}")

    print("=" * 60)
    print()

    # ========================================================
    # GENERATE CONTENT
    # ========================================================

    try:

        content = generate_social_content(
            topic=args.topic,
            brand=args.brand,
            platforms=platforms,
            api_key=args.api_key,
            model=args.model,
        )

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        print("=" * 60)
        print("✍️ SOCIAL MEDIA CONTENT")
        print("=" * 60)

        print(content)

        print()
        print("=" * 60)
        print("✅ Content generation completed successfully!")
        print("=" * 60)

    except Exception as e:

        print()
        print("=" * 60)
        print("❌ ERROR GENERATING CONTENT")
        print("=" * 60)

        print(str(e))

        print()
        print(
            "Please check your GROQ_API_KEY and "
            "Groq model availability."
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()