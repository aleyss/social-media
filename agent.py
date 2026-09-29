import os
from typing import Optional

from dotenv import load_dotenv

# ============================================================
# LITELLM COMPATIBILITY PATCH FOR GROQ
# ============================================================

try:
    import litellm

    # Ignore unsupported parameters
    litellm.drop_params = True

    # Save original functions
    _original_completion = litellm.completion
    _original_acompletion = litellm.acompletion

    def remove_cache_breakpoint(messages):
        """
        Remove cache_breakpoint because Groq does not
        accept this parameter.
        """

        if isinstance(messages, list):

            for message in messages:

                if isinstance(message, dict):
                    message.pop("cache_breakpoint", None)

                    # Also check nested content
                    if isinstance(message.get("content"), list):

                        for item in message["content"]:

                            if isinstance(item, dict):
                                item.pop(
                                    "cache_breakpoint",
                                    None
                                )

        elif isinstance(messages, dict):

            messages.pop(
                "cache_breakpoint",
                None
            )

    # --------------------------------------------------------
    # Safe synchronous completion
    # --------------------------------------------------------

    def safe_completion(*args, **kwargs):

        if "messages" in kwargs:
            remove_cache_breakpoint(
                kwargs["messages"]
            )

        elif len(args) > 1:
            if isinstance(args[1], list):
                remove_cache_breakpoint(
                    args[1]
                )

        return _original_completion(
            *args,
            **kwargs
        )

    # --------------------------------------------------------
    # Safe asynchronous completion
    # --------------------------------------------------------

    async def safe_acompletion(*args, **kwargs):

        if "messages" in kwargs:
            remove_cache_breakpoint(
                kwargs["messages"]
            )

        elif len(args) > 1:
            if isinstance(args[1], list):
                remove_cache_breakpoint(
                    args[1]
                )

        return await _original_acompletion(
            *args,
            **kwargs
        )

    # Replace LiteLLM functions
    litellm.completion = safe_completion
    litellm.acompletion = safe_acompletion

except ImportError:

    pass


# ============================================================
# CREWAI CACHE PATCH
# ============================================================

try:

    import crewai.llms.cache as crew_cache

    crew_cache.mark_cache_breakpoint = (
        lambda message: message
    )

except Exception:

    pass


try:

    import crewai.agents.crew_agent_executor as executor

    executor.mark_cache_breakpoint = (
        lambda message: message
    )

except Exception:

    pass


try:

    import crewai.experimental.agent_executor as experimental_executor

    experimental_executor.mark_cache_breakpoint = (
        lambda message: message
    )

except Exception:

    pass


# ============================================================
# CREWAI IMPORT
# ============================================================

from crewai import (
    Agent,
    Crew,
    LLM,
    Process,
    Task
)


# Load .env
load_dotenv()


# ============================================================
# BUILD GROQ LLM
# ============================================================

def build_llm(
    model: str = "openai/gpt-oss-120b",
    api_key: Optional[str] = None,
    temperature: float = 0.7
):

    # Get API key
    key = (
        api_key
        or os.getenv("GROQ_API_KEY")
    )

    if not key:

        raise ValueError(
            "Groq API key is missing. "
            "Please add GROQ_API_KEY to your .env file."
        )

    # Save key to environment
    os.environ["GROQ_API_KEY"] = key

    # Make sure Groq provider prefix exists
    if not model.startswith("groq/"):

        model = f"groq/{model}"

    # Create CrewAI LLM
    llm = LLM(

        model=model,

        api_key=key,

        temperature=temperature

    )

    return llm


# ============================================================
# GENERATE SOCIAL MEDIA CONTENT
# ============================================================

def generate_social_content(
    topic: str,
    brand: str = "",
    platforms: Optional[list[str]] = None,
    api_key: Optional[str] = None,
    model: str = "openai/gpt-oss-120b"
):

    # --------------------------------------------------------
    # DEFAULT PLATFORMS
    # --------------------------------------------------------

    if platforms is None:

        platforms = [
            "twitter",
            "linkedin",
            "instagram"
        ]

    # --------------------------------------------------------
    # CREATE GROQ LLM
    # --------------------------------------------------------

    llm = build_llm(

        model=model,

        api_key=api_key,

        temperature=0.7

    )

    # ========================================================
    # AGENT 1
    # SOCIAL MEDIA STRATEGIST
    # ========================================================

    strategist = Agent(

        role="Social Media Strategist",

        goal=(
            "Analyze the topic and create a practical "
            "social media strategy for the selected platforms."
        ),

        backstory=(
            "You are an experienced social media strategist "
            "who understands audience engagement, hooks, "
            "content strategy, tone of voice and hashtags."
        ),

        llm=llm,

        verbose=False

    )

    # ========================================================
    # AGENT 2
    # SOCIAL MEDIA COPYWRITER
    # ========================================================

    writer = Agent(

        role="Social Media Copywriter",

        goal=(
            "Create engaging and platform-specific "
            "social media content."
        ),

        backstory=(
            "You are an expert social media copywriter "
            "specializing in Twitter/X, LinkedIn and Instagram."
        ),

        llm=llm,

        verbose=False

    )

    # ========================================================
    # TASK 1
    # CREATE STRATEGY
    # ========================================================

    strategy_task = Task(

        description=f"""

Analyze the following topic for social media.

TOPIC:
{topic}

BRAND:
{brand if brand else "Not specified"}

SELECTED PLATFORMS:
{", ".join(platforms)}

Create a strategy containing:

1. Core message
2. Target audience
3. Emotional hook
4. Tone of voice
5. Key points
6. Five relevant hashtags

Make the strategy practical and platform-specific.

""",

        agent=strategist,

        expected_output=(
            "A clear social media strategy containing "
            "the core message, audience, hook, tone, "
            "key points and hashtags."
        )

    )

    # ========================================================
    # TASK 2
    # CREATE CONTENT
    # ========================================================

    writing_task = Task(

        description=f"""

Create social media content using the strategy
created by the Social Media Strategist.

TOPIC:
{topic}

BRAND:
{brand if brand else "General"}

SELECTED PLATFORMS:
{", ".join(platforms)}

IMPORTANT:

Only generate content for the platforms selected
by the user.

Do not generate content for unselected platforms.

--------------------------------------------------
TWITTER / X
--------------------------------------------------

If Twitter/X is selected:

- Create 2 tweet variations.
- Each tweet must be under 280 characters.
- Make them short and engaging.
- Create one thread opener.

--------------------------------------------------
LINKEDIN
--------------------------------------------------

If LinkedIn is selected:

- Create one professional post.
- 150–200 words.
- Start with a strong storytelling hook.
- Use short paragraphs.
- Include useful insights.
- End with a question or call to action.

--------------------------------------------------
INSTAGRAM
--------------------------------------------------

If Instagram is selected:

- Create one engaging caption.
- 100–150 words.
- Use a strong hook.
- Include a call to action.
- Add exactly 15 relevant hashtags.

--------------------------------------------------

Make every platform version unique.

Use clear headings:

TWITTER/X

LINKEDIN

INSTAGRAM

""",

        agent=writer,

        expected_output=(
            "High-quality platform-specific social media "
            "content for the selected platforms."
        ),

        context=[strategy_task]

    )

    # ========================================================
    # CREATE CREW
    # ========================================================

    crew = Crew(

        agents=[
            strategist,
            writer
        ],

        tasks=[
            strategy_task,
            writing_task
        ],

        process=Process.sequential,

        verbose=False

    )

    # ========================================================
    # RUN CREW
    # ========================================================

    result = crew.kickoff()

    return str(result)
