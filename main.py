"""Streamlit interface for the AI Study Assistant."""
import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.pipeline import answer_question, process_document

APP_TITLE = "AI Study Assistant"
CREATOR_NAME = "Lim Wen Le"
STYLESHEET_PATH = Path(__file__).parent / "assets" / "styles.css"


def initialize_session_state() -> None:
    """Create the temporary values used during the current app session."""
    if "embedded_chunks" not in st.session_state:
        st.session_state.embedded_chunks = None

    if "document_name" not in st.session_state:
        st.session_state.document_name = None

    if "upload_hash" not in st.session_state:
        st.session_state.upload_hash = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "active_page" not in st.session_state:
        st.session_state.active_page = "Home"


def create_openai_client() -> OpenAI:
    """Load the API key into the environment and create an OpenAI client."""
    load_dotenv()
    return OpenAI()


def reset_active_document(upload_hash: str) -> None:
    """Clear the current document and chat when a different PDF is selected."""
    st.session_state.upload_hash = upload_hash
    st.session_state.embedded_chunks = None
    st.session_state.document_name = None
    st.session_state.messages = []


def display_sources(sources: list[dict]) -> None:
    """Display the source chunks used to create an answer."""
    if not sources:
        return

    st.markdown("#### Sources")

    for index, source in enumerate(sources, start=1):
        label = (
            f"Source {index} — {source['document']}, "
            f"page {source['page']}"
        )

        with st.expander(label):
            st.write(source["text"])
            st.caption(f"Chunk {source['chunk']}")


def display_chat_history() -> None:
    """Display questions and answers from the current session."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                display_sources(message["sources"])


def process_uploaded_pdf(uploaded_file) -> None:
    """Temporarily save an uploaded PDF and send it through the pipeline."""
    safe_filename = Path(uploaded_file.name).name

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory) / safe_filename
        temporary_path.write_bytes(uploaded_file.getvalue())

        client = create_openai_client()
        embedded_chunks = process_document(temporary_path, client)

    st.session_state.embedded_chunks = embedded_chunks
    st.session_state.document_name = safe_filename
    st.session_state.messages = []


def render_home_page() -> None:
    """Display PDF upload controls and the question-answer interface."""
    st.caption("YOUR STUDY SPACE")
    st.title("Study with your documents")
    st.write(
        "Upload a text-based PDF, then ask questions grounded in its content."
    )

    with st.container(border=True, key="document_status"):
        if st.session_state.embedded_chunks is None:
            st.caption("NO ACTIVE DOCUMENT")
            st.write("Upload and process a PDF to begin asking questions.")
        else:
            chunk_count = len(st.session_state.embedded_chunks)
            st.caption("ACTIVE DOCUMENT")
            st.write(
                f"{st.session_state.document_name} is ready — "
                f"{chunk_count} chunks indexed."
            )

    display_chat_history()

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question about your PDF",
            placeholder="What would you like to understand?",
            disabled=st.session_state.embedded_chunks is None,
            key="question_input",
        )
        with st.container(horizontal=True, horizontal_alignment="right"):
            question_submitted = st.form_submit_button(
                "Ask",
                type="primary",
                width="content",
                disabled=st.session_state.embedded_chunks is None,
            )

    if question_submitted:
        cleaned_question = question.strip()

        if not cleaned_question:
            with st.container(border=True, key="question_feedback"):
                st.write("Enter a question before pressing Ask.")
        else:
            st.session_state.messages.append({
                "role": "user",
                "content": cleaned_question,
            })

            try:
                with st.spinner("Finding the most relevant information..."):
                    client = create_openai_client()
                    result = answer_question(
                        cleaned_question,
                        st.session_state.embedded_chunks,
                        client,
                    )
                    
            except Exception as error:
                with st.container(border=True, key="question_feedback"):
                    st.write(f"The question could not be answered: {error}")
            
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
                st.rerun()

    st.divider()
    st.subheader("Add or change your PDF")

    uploaded_file = st.file_uploader(
        "Click here to upload your PDF",
        type=["pdf"],
        accept_multiple_files=False,
        key="pdf_uploader",
    )

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        upload_hash = hashlib.sha256(uploaded_bytes).hexdigest()

        if upload_hash != st.session_state.upload_hash:
            reset_active_document(upload_hash)
            st.rerun()

    with st.container(horizontal=True, horizontal_alignment="right"):
        process_clicked = st.button(
            "Process PDF",
            type="primary",
            width="content",
            disabled=uploaded_file is None,
            key="process_pdf_button",
        )

    if process_clicked:
        try:
            with st.spinner("Preparing your PDF..."):
                process_uploaded_pdf(uploaded_file)
        except Exception as error:
            with st.container(border=True, key="pdf_feedback"):
                st.write(f"The PDF could not be processed: {error}")
        else:
            st.rerun()


def render_pdfs_page() -> None:
    """Display the placeholder for the saved PDF library."""
    st.caption("YOUR LIBRARY")
    st.title("PDFs")
    with st.container(border=True, key="library_notice"):
        st.caption("COMING NEXT")
        st.write(
            "Your saved PDF library is currently in progress. "
            "It will be connected to the SQLite database in the next stage."
        )


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📕",
        layout="centered",
    )
    st.html(STYLESHEET_PATH)
    initialize_session_state()

    with st.sidebar:
        with st.container(key="sidebar_brand"):
            st.caption("Your Document Space")
            st.title(APP_TITLE)
            st.caption(f"By {CREATOR_NAME}")
        st.divider()

        if st.button(
            "Home",
            type=(
                "primary"
                if st.session_state.active_page == "Home"
                else "secondary"
            ),
            use_container_width=True,
            key="nav_home",
        ):
            st.session_state.active_page = "Home"
            st.rerun()

        if st.button(
            "PDFs",
            type=(
                "primary"
                if st.session_state.active_page == "PDFs"
                else "secondary"
            ),
            use_container_width=True,
            key="nav_pdfs",
        ):
            st.session_state.active_page = "PDFs"
            st.rerun()

    if st.session_state.active_page == "Home":
        render_home_page()
    else:
        render_pdfs_page()


if __name__ == "__main__":
    main()
