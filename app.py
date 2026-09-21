"""
Streamlit Web UI for Social Media Content Agent (CrewAI).

Defaults to Groq (Free & Fast Llama 3.3 70B), with optional support for
OpenAI, Google Gemini, or local Ollama.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from agent import generate_social_content

load_dotenv()

st.set_page_config(
    page_title="Social Media Content Agent (Groq)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("⚡ LLM Configuration")

    provider_choice = st.selectbox(
        "Select Provider",
        options=[
            "Groq (Free & Fast — Llama 3.3 70B)",
            "Google Gemini (Free Tier — Gemini 2.0 Flash)",
            "OpenAI (GPT-4o-mini)",
            "Ollama (Local Offline)",
        ],
        index=0,
        help="Select which AI provider to use.",
    )

    api_key = None
    base_url = None
    provider_id = "groq"
    model_id = "groq/llama-3.3-70b-versatile"

    if "Groq" in provider_choice:
        provider_id = "groq"
        model_id = st.text_input("Groq Model", value="groq/llama-3.3-70b-versatile")
        env_key = os.getenv("GROQ_API_KEY", "")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=env_key,
            help="Free key (no credit card required) from console.groq.com/keys",
        )
        if not api_key:
            st.warning("⚠️ Paste your Groq API key here or add GROQ_API_KEY to `.env`.")
        st.markdown("[👉 **Get a Free Groq API Key** (instant)](https://console.groq.com/keys)")

    elif "Gemini" in provider_choice:
        provider_id = "gemini"
        model_id = st.text_input("Gemini Model", value="gemini/gemini-3.6-flash")
        env_key = os.getenv("GEMINI_API_KEY", "")
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            value=env_key,
            help="Free key at aistudio.google.com/app/apikey",
        )
        if not api_key:
            st.warning("⚠️ Paste your Gemini API key here or add GEMINI_API_KEY to `.env`.")
        st.markdown("[👉 **Get a Free Gemini Key**](https://aistudio.google.com/app/apikey)")

    elif "OpenAI" in provider_choice:
        provider_id = "openai"
        model_id = "gpt-4o-mini"
        env_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=env_key,
            help="Requires credit balance at platform.openai.com",
        )
        if not api_key:
            st.warning("⚠️ Enter your OpenAI API key.")

    elif "Ollama" in provider_choice:
        provider_id = "ollama"
        model_id = st.text_input("Ollama Model", value="ollama/llama3.2")
        base_url = st.text_input("Ollama Base URL", value="http://localhost:11434")
        st.caption("Ensure `ollama serve` is active.")

    st.divider()
    st.markdown(
        """
        ### 💡 Content Rules:
        - **Twitter/X:** 2 punchy tweets + thread opener.
        - **LinkedIn:** Storytelling hook & takeaways (150-200 words).
        - **Instagram:** Engaging caption with 15 hashtags.
        """
    )

# --- Main Interface ---
st.title("⚡ Social Media Content Agent")
st.markdown(
    "Generate platform-optimized social media content (**Twitter/X**, **LinkedIn**, **Instagram**) powered by **CrewAI & Groq Llama 3.3 70B**."
)

col1, col2 = st.columns([2, 1])

with col1:
    topic = st.text_area(
        "📌 Content Topic / Theme",
        value="The future of remote work",
        placeholder="Enter your topic, product announcement, or theme...",
        height=110,
    )

with col2:
    brand = st.text_input(
        "🏷️ Brand Name (Optional)",
        value="",
        placeholder="e.g. CloudSync, Acme Inc, or leave blank",
    )

st.write("**Target Platforms:**")
p1, p2, p3 = st.columns(3)
with p1:
    tw_selected = st.checkbox("Twitter / X", value=True)
with p2:
    li_selected = st.checkbox("LinkedIn", value=True)
with p3:
    ig_selected = st.checkbox("Instagram", value=True)

platforms = []
if tw_selected:
    platforms.append("twitter")
if li_selected:
    platforms.append("linkedin")
if ig_selected:
    platforms.append("instagram")

generate_btn = st.button("🚀 Generate Social Content", type="primary", use_container_width=True)

if generate_btn:
    if provider_id in ["groq", "openai", "gemini"] and not api_key:
        st.error(f"Please provide your {provider_choice.split(' ')[0]} API key in the sidebar.")
    elif not topic.strip():
        st.error("Please provide a topic to generate content for.")
    elif not platforms:
        st.error("Please select at least one social media platform.")
    else:
        with st.spinner(f"🤖 CrewAI Agents at work using {provider_choice.split(' ')[0]}..."):
            try:
                result = generate_social_content(
                    topic=topic.strip(),
                    brand=brand.strip(),
                    platforms=platforms,
                    api_key=api_key.strip() if api_key else None,
                    provider=provider_id,
                    model=model_id,
                    base_url=base_url,
                )

                st.success("✅ Content generated successfully!")

                st.divider()
                st.subheader("✍️ Generated Content")

                tab_preview, tab_raw = st.tabs(["✨ Formatted Output", "📋 Raw Markdown"])
                with tab_preview:
                    st.markdown(result)
                with tab_raw:
                    st.code(result, language="markdown")

                st.download_button(
                    label="💾 Download Generated Posts (.md)",
                    data=result,
                    file_name=f"social_posts_{brand or 'content'}.md",
                    mime="text/markdown",
                )

            except Exception as e:
                err_msg = str(e)
                st.error(f"Error generating content: {err_msg}")
