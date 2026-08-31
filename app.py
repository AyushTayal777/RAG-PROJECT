"""
Lumen — Meeting Intelligence UI
A Streamlit front-end for the AI video/meeting assistant pipeline in main.py.

Run with:
    streamlit run app.py
"""

import tempfile
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summary, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gist - Feed it a recording. Get the gist.",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# Theme — warm editorial paper & terracotta, deliberately not "AI purple"
# ────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;0,700;1,500&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --paper: #F6F1E7;
    --paper-alt: #EFE7D8;
    --card: #FFFDF8;
    --ink: #211D1A;
    --ink-soft: #5B554C;
    --line: #DFD5BF;
    --accent: #B5502F;
    --accent-dark: #8F3D22;
    --accent-soft: #F0DACB;
    --pine: #2E3B32;
    --pine-soft: #E4E9E2;
    --gold: #A9843A;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

/* Overall app background */
.stApp {
    background: var(--paper);
}

/* Kill default top padding a bit */
.block-container {
    padding-top: 2.2rem;
    max-width: 1180px;
}

h1, h2, h3, h4 {
    font-family: 'Lora', serif !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
}

p, li, span, label, div {
    color: var(--ink);
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--pine);
    border-right: 1px solid #1c241e;
}
[data-testid="stSidebar"] * {
    color: #F3EFE4 !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    background: #3A4A3E;
    border: 1px solid #4C5E50;
    border-radius: 6px;
    color: #F6F1E7 !important;
}
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] .stFileUploader label {
    color: #E9E3D3 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #46554A;
}

/* Sidebar brand block */
.lumen-brand {
    font-family: 'Lora', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #F6F1E7 !important;
    letter-spacing: 0.02em;
    margin-bottom: 0px;
}
.lumen-tagline {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #B7C4B9 !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: -6px;
    margin-bottom: 1.4rem;
}

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    background: var(--accent);
    color: #FBF3EA !important;
    border: none;
    border-radius: 6px;
    padding: 0.55rem 1.1rem;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    transition: background 0.15s ease, transform 0.05s ease;
    box-shadow: none;
}
.stButton > button:hover {
    background: var(--accent-dark);
    color: #FBF3EA !important;
}
.stButton > button:active {
    transform: translateY(1px);
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: var(--accent);
}

/* Secondary / download buttons */
.stDownloadButton > button {
    background: transparent;
    border: 1.5px solid var(--ink);
    color: var(--ink) !important;
    border-radius: 6px;
    font-weight: 600;
}
.stDownloadButton > button:hover {
    background: var(--ink);
    color: var(--paper) !important;
}

/* ── Tabs ─────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    gap: 1.6rem;
    border-bottom: 1.5px solid var(--line);
}
[data-baseweb="tab"] {
    background: transparent !important;
    font-family: 'Lora', serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--ink-soft) !important;
    padding-bottom: 0.6rem;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--accent) !important;
    border-bottom: 2.5px solid var(--accent) !important;
}

/* ── Cards ────────────────────────────────────────────── */
.lumen-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.1rem;
}
.lumen-card h4 {
    margin-top: 0;
    font-size: 1.05rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent) !important;
}

.lumen-hero {
    border: 1px dashed var(--line);
    border-radius: 12px;
    padding: 3.2rem 2.4rem;
    text-align: center;
    background: var(--paper-alt);
}
.lumen-hero h2 {
    font-size: 1.7rem;
    margin-bottom: 0.4rem;
}
.lumen-hero p {
    color: var(--ink-soft);
    font-size: 0.95rem;
    max-width: 480px;
    margin: 0 auto;
}

/* Title header */
.lumen-title-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 2px solid var(--ink);
    padding-bottom: 0.8rem;
    margin-bottom: 1.6rem;
}
.lumen-title-row h1 {
    font-size: 2rem;
    margin: 0;
}
.lumen-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--ink-soft);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Pill list items for action/decision/question extracts */
.lumen-item {
    display: flex;
    gap: 0.75rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid var(--line);
    font-size: 0.95rem;
    line-height: 1.5;
}
.lumen-item:last-child { border-bottom: none; }
.lumen-marker {
    flex-shrink: 0;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    color: var(--accent);
}

/* Transcript block */
.lumen-transcript {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.86rem;
    line-height: 1.7;
    white-space: pre-wrap;
    max-height: 560px;
    overflow-y: auto;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 10px;
}

/* Status widget */
[data-testid="stStatusWidget"] {
    border-radius: 8px;
}

/* Divider */
hr {
    border-color: var(--line);
}

/* Small caption / label styling */
.lumen-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--gold);
    margin-bottom: 0.3rem;
    display: block;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────────────────────────────────
defaults = {
    "results": None,
    "chat_history": [],
    "processing_log": [],
    "source_label": "",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def as_items(value):
    """Normalize extractor output (str or list) into a clean list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip("-• \n") for v in value if str(v).strip()]
    lines = [ln.strip("-• \n") for ln in str(value).split("\n")]
    return [ln for ln in lines if ln]


def render_item_list(items, empty_label="Nothing surfaced here."):
    items = as_items(items)
    if not items:
        st.markdown(f"<p style='color:var(--ink-soft); font-style:italic;'>{empty_label}</p>", unsafe_allow_html=True)
        return
    html = "<div class='lumen-card'>"
    for i, item in enumerate(items, start=1):
        html += f"<div class='lumen-item'><span class='lumen-marker'>{i:02d}</span><span>{item}</span></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Sidebar — input & pipeline trigger
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='lumen-brand'>◆ GIST</div>", unsafe_allow_html=True)
    st.markdown("<div class='lumen-tagline'>Video Intelligence, Distilled</div>", unsafe_allow_html=True)

    mode = st.radio(
        "Source",
        ["YouTube URL", "Upload a file", "Local file path"],
        label_visibility="visible",
    )

    source = None
    if mode == "YouTube URL":
        source = st.text_input("Paste a YouTube link", placeholder="https://youtube.com/watch?v=...")
    elif mode == "Upload a file":
        uploaded = st.file_uploader("Audio or video file", type=["mp3", "wav", "m4a", "mp4", "mov", "mkv"])
        if uploaded is not None:
            tmp_dir = Path(tempfile.gettempdir()) / "lumen_uploads"
            tmp_dir.mkdir(exist_ok=True)
            tmp_path = tmp_dir / uploaded.name
            tmp_path.write_bytes(uploaded.getbuffer())
            source = str(tmp_path)
            st.caption(f"Saved to `{tmp_path.name}`")
    else:
        source = st.text_input("Local file path", placeholder="/path/to/recording.mp4")

    st.markdown("<br>", unsafe_allow_html=True)
    run_clicked = st.button("Analyze  →", use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.session_state.results:
        if st.button("↺  New session", use_container_width=True):
            st.session_state.results = None
            st.session_state.chat_history = []
            st.rerun()

    st.markdown(
        "<p style='font-size:0.75rem; color:#9FB0A2; margin-top:2rem;'>"
        "Pipeline: ingest → transcribe → summarize → extract → index."
        "</p>",
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ────────────────────────────────────────────────────────────────────────────
if run_clicked:
    if not source:
        st.error("Give me a YouTube URL, an uploaded file, or a local file path first.")
    else:
        st.session_state.chat_history = []
        st.session_state.source_label = source
        try:
            with st.status("Running the pipeline…", expanded=True) as status:
                status.write("📥  Ingesting source & chunking audio…")
                chunks = process_input(source)

                status.write("🎙️  Transcribing…")
                transcript = transcribe_all(chunks)

                status.write("🏷️  Naming the session…")
                title = generate_title(transcript)

                status.write("📝  Summarizing…")
                summarized = summary(transcript)

                status.write("✅  Extracting action items…")
                action_items = extract_action_items(transcript)

                status.write("🔑  Extracting key decisions…")
                decisions = extract_key_decisions(transcript)

                status.write("❓  Extracting open questions…")
                questions = extract_questions(transcript)

                status.write("🧠  Building the knowledge base for chat…")
                rag_chain = build_rag_chain(transcript)

                status.update(label="Done.", state="complete", expanded=False)

            st.session_state.results = {
                "title": title,
                "transcript": transcript,
                "summary": summarized,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
                "generated_at": datetime.now().strftime("%b %d, %Y · %H:%M"),
            }
            st.rerun()
        except Exception as e:
            st.error(f"Something broke mid-pipeline: {e}")

# ────────────────────────────────────────────────────────────────────────────
# Main content
# ────────────────────────────────────────────────────────────────────────────
results = st.session_state.results

if not results:
    st.markdown(
        """
        <div class="lumen-hero">
            <h2>◆ Bring in a recording</h2>
            <p>Drop a YouTube link, upload a file, or point to a local path in the
            sidebar. GIST will transcribe it, summarize it, pull out action items
            and decisions, and hand you a chat box to interrogate the whole thing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="lumen-title-row">
            <h1>{results['title']}</h1>
            <div class="lumen-meta">{results['generated_at']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["Summary", "Action Items", "Decisions", "Open Questions", "Transcript", "Ask the Meeting"]
    )

    with tab_summary:
        st.markdown(
            f"<div class='lumen-card'><h4>Summary</h4>{results['summary']}</div>",
            unsafe_allow_html=True,
        )

        report_md = (
            f"# {results['title']}\n\n"
            f"_Generated {results['generated_at']}_\n\n"
            f"## Summary\n{results['summary']}\n\n"
            f"## Action Items\n"
            + "\n".join(f"- {i}" for i in as_items(results["action_items"])) + "\n\n"
            f"## Key Decisions\n"
            + "\n".join(f"- {i}" for i in as_items(results["key_decisions"])) + "\n\n"
            f"## Open Questions\n"
            + "\n".join(f"- {i}" for i in as_items(results["open_questions"])) + "\n"
        )
        st.download_button(
            "⤓  Download full report (.md)",
            data=report_md,
            file_name=f"{results['title'].strip().replace(' ', '_')[:60]}.md",
            mime="text/markdown",
        )

    with tab_actions:
        st.markdown("<span class='lumen-label'>Things to do</span>", unsafe_allow_html=True)
        render_item_list(results["action_items"], "No action items detected.")

    with tab_decisions:
        st.markdown("<span class='lumen-label'>What got decided</span>", unsafe_allow_html=True)
        render_item_list(results["key_decisions"], "No firm decisions detected.")

    with tab_questions:
        st.markdown("<span class='lumen-label'>Still unresolved</span>", unsafe_allow_html=True)
        render_item_list(results["open_questions"], "No open questions detected.")

    with tab_transcript:
        st.markdown("<span class='lumen-label'>Full transcript</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='lumen-transcript'>{results['transcript']}</div>", unsafe_allow_html=True)
        st.download_button(
            "⤓  Download transcript (.txt)",
            data=results["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
        )

    with tab_chat:
        st.markdown("<span class='lumen-label'>Chat with this recording</span>", unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask something about the meeting…")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = ask_question(results["rag_chain"], question)
                    except Exception as e:
                        answer = f"Couldn't get an answer: {e}"
                    st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
