"""
Premium CSS theme for the NotebookLM application.

Obsidian-dark theme with glassmorphism, micro-animations,
and a polished cinematic feel.
"""

from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    """Inject the full CSS theme into the Streamlit app."""
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
/* ================================================================
   NotebookLM — Obsidian Dark Premium Theme
   ================================================================ */

/* --- Fonts ---------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* --- Animations ----------------------------------------------- */
@keyframes slideUpFade {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 5px rgba(0, 255, 255, 0.1); }
    50%      { box-shadow: 0 0 20px rgba(0, 255, 255, 0.15); }
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* --- Base ----------------------------------------------------- */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #09090B;
    color: #E4E4E7;
}

/* --- Sidebar -------------------------------------------------- */
[data-testid="stSidebar"] {
    background-color: #0C0C0F;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #71717A;
    font-weight: 600;
    margin-top: 1.5rem;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    font-size: 0.82rem;
    color: #A1A1AA;
}

/* --- Buttons -------------------------------------------------- */
.stButton > button {
    border: 1px solid rgba(0, 255, 255, 0.15);
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.85rem;
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.05), rgba(0, 255, 255, 0.02));
    color: #00E5FF;
    padding: 0.5rem 1.2rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    letter-spacing: 0.01em;
}
.stButton > button:hover {
    border-color: rgba(0, 255, 255, 0.5);
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.12), rgba(0, 255, 255, 0.05));
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.12), 0 4px 12px rgba(0, 0, 0, 0.3);
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0) scale(0.98);
}

/* --- Inputs --------------------------------------------------- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background-color: rgba(255, 255, 255, 0.03) !important;
    color: #E4E4E7;
    font-size: 0.88rem;
    transition: all 0.25s ease;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(0, 255, 255, 0.4);
    box-shadow: 0 0 0 3px rgba(0, 255, 255, 0.08);
    outline: none;
}

/* --- File uploader -------------------------------------------- */
.stFileUploader > div > div {
    border-radius: 10px;
    border: 1px dashed rgba(255, 255, 255, 0.1);
    background: rgba(255, 255, 255, 0.02);
    transition: all 0.3s ease;
}
.stFileUploader > div > div:hover {
    border-color: rgba(0, 255, 255, 0.3);
    background: rgba(0, 255, 255, 0.03);
}

/* --- Dividers ------------------------------------------------- */
hr {
    margin: 1.5rem 0;
    border: none;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

/* --- Chat messages -------------------------------------------- */
.stChatMessage {
    background-color: transparent;
    padding: 1.2rem 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    animation: slideUpFade 0.4s ease-out forwards;
}

/* Assistant avatar */
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.15), rgba(0, 200, 255, 0.1));
    border: 1px solid rgba(0, 255, 255, 0.3);
    color: #00E5FF;
}

/* User avatar */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(139, 92, 246, 0.1));
    border: 1px solid rgba(168, 85, 247, 0.3);
    color: #A855F7;
}

/* --- Chat input ----------------------------------------------- */
.stChatInputContainer {
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    background-color: #09090B;
    transition: all 0.3s ease;
}
.stChatInputContainer:focus-within {
    border-color: rgba(0, 255, 255, 0.3);
    box-shadow: 0 -4px 20px rgba(0, 255, 255, 0.05);
}

/* --- Typography ----------------------------------------------- */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.03em;
    color: #FAFAFA !important;
    font-size: 1.8rem !important;
}
h2 {
    font-weight: 600 !important;
    letter-spacing: -0.02em;
    color: #E4E4E7 !important;
}
h3 {
    font-weight: 600 !important;
    color: #D4D4D8 !important;
}
p, li {
    color: #A1A1AA;
    line-height: 1.7;
    font-size: 0.92rem;
}

/* --- Code blocks ---------------------------------------------- */
code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    background-color: rgba(255, 255, 255, 0.06);
    padding: 0.15em 0.4em;
    border-radius: 4px;
    color: #00E5FF;
}

/* --- Metrics / Stats ------------------------------------------ */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 1rem;
}

/* --- Expander ------------------------------------------------- */
.streamlit-expanderHeader {
    font-size: 0.88rem;
    font-weight: 500;
    color: #A1A1AA;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.02);
}

/* --- Tabs ----------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.stTabs [data-baseweb="tab"] {
    color: #71717A;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.6rem 1.2rem;
    border-bottom: 2px solid transparent;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    color: #00E5FF !important;
    border-bottom-color: #00E5FF !important;
    background: transparent !important;
}

/* --- Success / Error / Warning / Info alerts ------------------ */
.stAlert {
    border-radius: 8px;
    font-size: 0.85rem;
    animation: fadeIn 0.3s ease;
}

/* --- Scrollbar ------------------------------------------------ */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}

/* --- Source citation cards ------------------------------------- */
.source-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    transition: all 0.2s ease;
    animation: fadeIn 0.3s ease;
}
.source-card:hover {
    border-color: rgba(0, 255, 255, 0.2);
    background: rgba(0, 255, 255, 0.03);
}
.source-card .source-title {
    color: #00E5FF;
    font-weight: 500;
    font-size: 0.82rem;
}
.source-card .source-preview {
    color: #71717A;
    font-size: 0.78rem;
    margin-top: 0.3rem;
    line-height: 1.5;
}

/* --- Notebook selector ---------------------------------------- */
.notebook-chip {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    background: rgba(0, 255, 255, 0.08);
    border: 1px solid rgba(0, 255, 255, 0.2);
    color: #00E5FF;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.2rem 0.3rem;
}

/* --- Empty state ---------------------------------------------- */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #52525B;
    animation: fadeIn 0.5s ease;
}
.empty-state .emoji {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: block;
}
.empty-state .title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #71717A;
    margin-bottom: 0.5rem;
}
.empty-state .subtitle {
    font-size: 0.85rem;
    color: #52525B;
}
</style>
"""
