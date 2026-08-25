import re
import gradio as gr
from ingestion import get_timestamped_docs
from vectorstore import get_vectorstore, index_docs
from retrieval import get_retriever
from chain import build_chain, clear_session

_chain = None
_current_video_id = None

def extract_video_id(url: str) -> str:
    match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def load_video(url: str) -> str:
    """
    Called when the user clicks 'Load Video'.
    Fetches transcript, indexes into Chroma, builds retriever + chain.
    Returns a status message shown in the UI.
    """
    global _chain, _current_video_id

    if not url.strip():
        return "Please enter a YouTube URL."

    try:
        video_id = extract_video_id(url)

        if video_id == _current_video_id:
            return "This video is already loaded. Go ahead and ask questions!"

        print(f"[load_video] Loading video: {video_id}")
        docs = get_timestamped_docs(video_id)

        vs = get_vectorstore()
        index_docs(vectorstore=vs, docs=docs, video_id=video_id)

        retriever = get_retriever(vectorstore=vs, docs=docs, video_id=video_id)

        clear_session()
        _chain = build_chain(retriever)
        _current_video_id = video_id

        print(f"[load_video] Ready. Indexed {len(docs)} chunks.")
        return f"Video loaded ({len(docs)} chunks indexed). Ask your questions below!"

    except ValueError as e:
        return f"{str(e)}"
    except Exception as e:
        return f"Error loading video: {str(e)}"

def answer_question(question: str, history: list) -> str:
    """
    Called by gr.ChatInterface on every user message.
    Reuses the persistent chain so message history is preserved.
    """
    if _chain is None:
        return "Please load a video first using the URL box above."

    if not question.strip():
        return "Please enter a question."

    try:
        response = _chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": "default"}},
        )
        return response
    except Exception as e:
        return f" Error: {str(e)}"

with gr.Blocks(title="YouTube RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 YouTube Video Q&A")
    gr.Markdown("Load a video once, then ask as many questions as you like.")

    with gr.Row():
        url_input = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            scale=4,
        )
        load_btn = gr.Button("Load Video", variant="primary", scale=1)

    status_box = gr.Textbox(
        label="Status",
        interactive=False,
        lines=1,
    )

    load_btn.click(fn=load_video, inputs=[url_input], outputs=[status_box])
    url_input.submit(fn=load_video, inputs=[url_input], outputs=[status_box])

    gr.Markdown("---")

    chat = gr.ChatInterface(
        fn=answer_question,
        chatbot=gr.Chatbot(height=450, label="Conversation"),
        textbox=gr.Textbox(
            placeholder="Ask anything about the video...",
            label="Your Question",
        )
    )

if __name__ == "__main__":
    demo.launch()