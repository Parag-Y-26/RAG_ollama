"""
Application theme injection.
"""

from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    """Inject the full CSS theme into the Streamlit app."""
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
/* ================================================================
   NotebookLM — Strict Monochrome Minimal Theme
   ================================================================ */

@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* --- Reset & Base --------------------------------------------- */
*, *::before, *::after {
    box-sizing: border-box;
}

.stApp {
    font-family: 'DM Mono', 'Fira Mono', monospace;
    background-color: #000000;
    color: #FFFFFF;
    font-size: 13px;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* --- Sidebar -------------------------------------------------- */
[data-testid="stSidebar"] {
    background-color: #000000;
    border-right: 1px solid #1C1C1C;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #555555;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
}

/* --- Typography ----------------------------------------------- */
h1 {
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 1.4rem !important;
    letter-spacing: -0.03em !important;
    color: #FFFFFF !important;
}
h2 {
    font-family: 'DM Mono', monospace !important;
    font-weight: 400 !important;
    font-size: 1rem !important;
    letter-spacing: -0.01em !important;
    color: #FFFFFF !important;
}
h3 {
    font-family: 'DM Mono', monospace !important;
    font-weight: 400 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #555555 !important;
}
p, li, .stMarkdown p {
    font-family: 'DM Mono', monospace;
    font-weight: 300;
    font-size: 13px;
    color: #A0A0A0;
    line-height: 1.8;
}
strong, b {
    font-weight: 500;
    color: #FFFFFF;
}
caption, .stCaption, small {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 300 !important;
    color: #555555 !important;
    letter-spacing: 0.04em !important;
}

/* --- Inputs --------------------------------------------------- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 300 !important;
    background-color: #0A0A0A !important;
    border: 1px solid #1C1C1C !important;
    border-radius: 2px !important;
    color: #FFFFFF !important;
    padding: 0.6rem 0.75rem !important;
    transition: border-color 0.15s ease !important;
    caret-color: #FFFFFF;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #FFFFFF !important;
    box-shadow: none !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
    color: #333333 !important;
    font-weight: 300 !important;
}

/* --- Selectbox ------------------------------------------------ */
.stSelectbox > div > div {
    background-color: #0A0A0A !important;
    border: 1px solid #1C1C1C !important;
    border-radius: 2px !important;
    color: #FFFFFF !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 300 !important;
    transition: border-color 0.15s ease !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #FFFFFF !important;
    box-shadow: none !important;
}

/* --- Buttons -------------------------------------------------- */
.stButton > button {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    background-color: transparent !important;
    border: 1px solid #1C1C1C !important;
    border-radius: 2px !important;
    color: #A0A0A0 !important;
    padding: 0.45rem 1rem !important;
    transition: border-color 0.15s ease, color 0.15s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    border-color: #FFFFFF !important;
    color: #FFFFFF !important;
    background-color: transparent !important;
    transform: none !important;
    box-shadow: none !important;
}
.stButton > button:active {
    background-color: #111111 !important;
}

/* Primary / full-width action buttons */
.stButton > button[kind="primary"],
div[data-testid="stButton"] > button[use_container_width="true"] {
    border-color: #2E2E2E !important;
    color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #111111 !important;
    border-color: #FFFFFF !important;
}

/* --- Download button ------------------------------------------ */
.stDownloadButton > button {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid #1C1C1C !important;
    border-radius: 2px !important;
    color: #555555 !important;
    transition: border-color 0.15s ease, color 0.15s ease !important;
}
.stDownloadButton > button:hover {
    border-color: #FFFFFF !important;
    color: #FFFFFF !important;
}

/* --- File uploader -------------------------------------------- */
.stFileUploader > div > div {
    background-color: #0A0A0A !important;
    border: 1px dashed #1C1C1C !important;
    border-radius: 2px !important;
    transition: border-color 0.15s ease !important;
}
.stFileUploader > div > div:hover {
    border-color: #555555 !important;
}

/* --- Chat messages -------------------------------------------- */
.stChatMessage {
    background-color: transparent !important;
    border: none !important;
    border-bottom: 1px solid #0F0F0F !important;
    border-radius: 0 !important;
    padding: 1.25rem 0.5rem !important;
    animation: msgReveal 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

@keyframes msgReveal {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* === AVATAR SELECTORS FOR STREAMLIT ≥1.40 === */

/* User avatar */
div[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarUser"] > div,
[data-testid="chatAvatarIcon-user"] {
    background-color: #000000 !important;
    background: #000000 !important;
    border: 1px solid #2E2E2E !important;
    border-radius: 2px !important;
    color: #FFFFFF !important;
    font-size: 11px !important;
}

/* Assistant avatar */
div[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarAssistant"] > div,
[data-testid="chatAvatarIcon-assistant"] {
    background-color: #111111 !important;
    background: #111111 !important;
    border: 1px solid #2E2E2E !important;
    border-radius: 2px !important;
    color: #FFFFFF !important;
    font-size: 11px !important;
}

/* Nuke any Streamlit SVG fills on avatars */
[data-testid="stChatMessageAvatarUser"] svg,
[data-testid="stChatMessageAvatarAssistant"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

/* --- Chat input (correct Streamlit ≥1.40 selectors) ---------- */
div[data-testid="stChatInput"] {
    background-color: #000000 !important;
}
div[data-testid="stChatInput"] > div {
    border-radius: 2px !important;
    border: 1px solid #1C1C1C !important;
    background-color: #0A0A0A !important;
    transition: border-color 0.15s ease !important;
}
div[data-testid="stChatInput"] > div:focus-within {
    border-color: #FFFFFF !important;
    box-shadow: none !important;
}
div[data-testid="stChatInput"] textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 300 !important;
    color: #FFFFFF !important;
    caret-color: #FFFFFF !important;
}

/* Legacy selectors (keep for older Streamlit) */
.stChatInputContainer {
    background-color: #000000 !important;
    border-top: 1px solid #1C1C1C !important;
    padding: 0.75rem 0 !important;
}
.stChatInputContainer > div > div {
    background-color: #0A0A0A !important;
    border: 1px solid #1C1C1C !important;
    border-radius: 2px !important;
    transition: border-color 0.15s ease !important;
}
.stChatInputContainer > div > div:focus-within {
    border-color: #FFFFFF !important;
}
.stChatInputContainer textarea {
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 300 !important;
    color: #FFFFFF !important;
    background: transparent !important;
}

/* --- Expander ------------------------------------------------- */
.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #555555 !important;
    border: 1px solid #1C1C1C !important;
    border-radius: 2px !important;
    background-color: transparent !important;
    padding: 0.5rem 0.75rem !important;
    transition: border-color 0.15s ease, color 0.15s ease !important;
}
.streamlit-expanderHeader:hover {
    border-color: #2E2E2E !important;
    color: #A0A0A0 !important;
}
[data-testid="stExpander"] > div:last-child {
    border: 1px solid #1C1C1C !important;
    border-top: none !important;
    border-radius: 0 0 2px 2px !important;
    background-color: #0A0A0A !important;
}

/* --- Tabs ----------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1px solid #1C1C1C !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 400 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #555555 !important;
    padding: 0.6rem 1.25rem !important;
    border: none !important;
    border-bottom: 1px solid transparent !important;
    background: transparent !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    border-bottom-color: #FFFFFF !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #A0A0A0 !important;
}

/* --- Metrics -------------------------------------------------- */
[data-testid="stMetric"] {
    background-color: #0A0A0A !important;
    border: 1px solid #1C1C1C !important;
    border-radius: 2px !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 400 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #555555 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.4rem !important;
    font-weight: 500 !important;
    color: #FFFFFF !important;
}

/* --- Alerts --------------------------------------------------- */
.stAlert {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 300 !important;
    border-radius: 2px !important;
    border-left-width: 2px !important;
}
/* Override Streamlit's colorful alerts to monochrome */
[data-testid="stAlert"][kind="info"] {
    background-color: #0A0A0A !important;
    border-color: #2E2E2E !important;
    color: #A0A0A0 !important;
}
[data-testid="stAlert"][kind="success"] {
    background-color: #0A0A0A !important;
    border-color: #2E2E2E !important;
    color: #FFFFFF !important;
}
[data-testid="stAlert"][kind="warning"] {
    background-color: #0A0A0A !important;
    border-color: #555555 !important;
    color: #A0A0A0 !important;
}
[data-testid="stAlert"][kind="error"] {
    background-color: #0A0A0A !important;
    border: 1px solid #1C1C1C !important;
    border-left: 2px solid #FF4444 !important;
    border-radius: 2px !important;
    color: #FF4444 !important;
}
[data-testid="stAlert"][kind="error"] p {
    color: #FF4444 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}
/* Also target baseweb notification for error alerts */
div[data-testid="stAlert"][data-baseweb="notification"] {
    background-color: #0A0A0A !important;
    border: 1px solid #1C1C1C !important;
    border-left: 2px solid #FF4444 !important;
    border-radius: 2px !important;
}

/* --- Divider -------------------------------------------------- */
hr, .stDivider {
    border: none !important;
    border-top: 1px solid #1C1C1C !important;
    margin: 1.25rem 0 !important;
}

/* --- Popover -------------------------------------------------- */
[data-testid="stPopover"] > div {
    background-color: #0A0A0A !important;
    border: 1px solid #1C1C1C !important;
    border-radius: 2px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.6) !important;
}

/* --- Spinner -------------------------------------------------- */
.stSpinner > div {
    border-color: #1C1C1C #1C1C1C #FFFFFF #FFFFFF !important;
}

/* --- Toggle --------------------------------------------------- */
.stToggle > label {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #555555 !important;
}

/* --- Scrollbar ------------------------------------------------ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #000000; }
::-webkit-scrollbar-thumb { background: #1C1C1C; border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: #2E2E2E; }

/* --- Code ----------------------------------------------------- */
code {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    background-color: #111111 !important;
    border: 1px solid #1C1C1C !important;
    border-radius: 2px !important;
    padding: 0.1em 0.4em !important;
    color: #FFFFFF !important;
}
pre code {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}
pre {
    background-color: #0A0A0A !important;
    border: 1px solid #1C1C1C !important;
    border-radius: 2px !important;
    padding: 1rem !important;
}

/* --- Source citation cards ------------------------------------ */
.source-card {
    background-color: #0A0A0A;
    border: 1px solid #1C1C1C;
    border-radius: 2px;
    padding: 0.75rem 1rem;
    margin: 0.35rem 0;
    transition: border-color 0.15s ease;
}
.source-card:hover {
    border-color: #2E2E2E;
}
.source-card .source-title {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    font-weight: 400;
    color: #FFFFFF;
}
.source-card .source-preview {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 300;
    color: #555555;
    margin-top: 0.3rem;
    line-height: 1.6;
}

/* --- Empty state ---------------------------------------------- */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    animation: msgReveal 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
.empty-state .emoji {
    font-size: 2rem;
    margin-bottom: 1.5rem;
    display: block;
    opacity: 0.5;
}
.empty-state .title {
    font-family: 'DM Mono', monospace;
    font-size: 14px;
    font-weight: 400;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #FFFFFF;
    margin-bottom: 0.75rem;
}
.empty-state .subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    font-weight: 300;
    color: #A0A0A0;
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.8;
}

/* --- Notebook chip -------------------------------------------- */
.notebook-chip {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border: 1px solid #1C1C1C;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 400;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #555555;
}

/* --- Mode badge ----------------------------------------------- */
.mode-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 9px;
    font-weight: 400;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #333333;
    border: 1px solid #1C1C1C;
    border-radius: 2px;
    padding: 0.15rem 0.5rem;
    margin-top: 0.35rem;
}

/* --- Sidebar app title ---------------------------------------- */
.sidebar-title {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #FFFFFF;
}

/* --- Status indicator ----------------------------------------- */
.status-online {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #555555;
    letter-spacing: 0.04em;
}
.status-offline {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #FF4444;
    letter-spacing: 0.04em;
}

</style>
"""
