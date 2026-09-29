"""
Streamlit Web UI for Social Media Content Agent (CrewAI).

Defaults to Groq (Free & Fast Llama 3.3 70B), with optional support for
OpenAI, Google Gemini, or local Ollama.

Run with:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
import os

from agent import generate_social_content

# Load environment variables
load_dotenv()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Social Media Content Agent",
    page_icon="⚡",
    layout="wide"
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("⚡ AI Social Media Content Agent")

st.markdown(
    """
    Generate engaging social media content using
    **CrewAI + Groq LLM**.
    
    Create content for **Twitter/X, LinkedIn, and Instagram**
    from a single topic.
    """
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚡ Groq Configuration")

    # Get API key from .env
    env_key = os.getenv("GROQ_API_KEY", "")

    api_key = st.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
        help="Enter your Groq API key."
    )

    st.markdown(
        "[Get your Groq API Key](https://console.groq.com/keys)"
    )

    st.divider()

    model = st.text_input(
        "Groq Model",
        value="openai/gpt-oss-120b"
    )

    st.caption(
        "Groq provides fast inference for the selected model."
    )

    st.divider()

    st.subheader("Content Rules")

    st.markdown(
        """
        **Twitter/X**
        - 2 tweet variations
        - Thread opener
        - Under 280 characters

        **LinkedIn**
        - 150–200 words
        - Storytelling hook
        - Professional tone

        **Instagram**
        - 100–150 words
        - Engaging caption
        - 15 hashtags
        """
    )

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("📝 Create Your Content")

col1, col2 = st.columns([2, 1])

with col1:

    topic = st.text_area(
        "📌 Content Topic",
        placeholder="Example: The future of Artificial Intelligence",
        height=120
    )

with col2:

    brand = st.text_input(
        "🏷️ Brand Name",
        placeholder="Optional"
    )

# --------------------------------------------------
# PLATFORM SELECTION
# --------------------------------------------------

st.write("### 📱 Select Platforms")

col1, col2, col3 = st.columns(3)

with col1:
    twitter = st.checkbox(
        "Twitter / X",
        value=True
    )

with col2:
    linkedin = st.checkbox(
        "LinkedIn",
        value=True
    )

with col3:
    instagram = st.checkbox(
        "Instagram",
        value=True
    )

# Create platform list
platforms = []

if twitter:
    platforms.append("twitter")

if linkedin:
    platforms.append("linkedin")

if instagram:
    platforms.append("instagram")

# --------------------------------------------------
# GENERATE BUTTON
# --------------------------------------------------

st.divider()

generate_button = st.button(
    "🚀 Generate Social Media Content",
    type="primary",
    use_container_width=True
)

# --------------------------------------------------
# GENERATE CONTENT
# --------------------------------------------------

if generate_button:

    # Check API key
    if not api_key:

        st.error(
            "❌ Please enter your Groq API key."
        )

    # Check topic
    elif not topic.strip():

        st.error(
            "❌ Please enter a content topic."
        )

    # Check platforms
    elif not platforms:

        st.error(
            "❌ Please select at least one platform."
        )

    else:

        with st.spinner(
            "🤖 AI agents are generating your content..."
        ):

            try:

                result = generate_social_content(
                    topic=topic.strip(),
                    brand=brand.strip(),
                    platforms=platforms,
                    api_key=api_key.strip(),
                    model=model.strip()
                )

                st.success(
                    "✅ Content generated successfully!"
                )

                st.divider()

                st.subheader(
                    "✍️ Generated Social Media Content"
                )

                # Display result
                st.markdown(result)

                # Download button
                st.download_button(
                    label="💾 Download Content",
                    data=result,
                    file_name="social_media_content.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"❌ Error generating content: {str(e)}"
                )
