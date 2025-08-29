import os, json, gradio as gr
from dotenv import load_dotenv
from src.config import AppConfig
from src.ingest import ingest_files
from src.rag import rag_chat
from src.vectorstore import ensure_index
from src.embeddings import get_embedder

load_dotenv()

cfg = AppConfig.from_env()
embedder = get_embedder(cfg.EMBED_MODEL)

# Ensure vector index exists (dimension must match embedding model)
ensure_index(cfg, dim=embedder.get_sentence_embedding_dimension())

# --- UI CALLBACKS ---

def ui_ingest(files, namespace):
    if not files:
        return "No files selected."
    report = ingest_files(cfg, embedder, files, namespace or cfg.NAMESPACE)
    return json.dumps(report, indent=2)

def ui_costs(monthly_queries, prompt_tk, completion_tk, price_per_1k, base_fixed):
    try:
        mq = float(monthly_queries or 0)
        pt = float(prompt_tk or 0)
        ct = float(completion_tk or 0)
        ppk = float(price_per_1k or 0)
        base = float(base_fixed or 0)
        token_cost = (mq * (pt + ct) / 1000.0) * ppk
        return f"Estimated monthly: ${base + token_cost:,.2f} (Base=${base:.2f}, Tokens=${token_cost:,.2f})"
    except Exception as e:
        return f"Error computing estimate: {e}"

def get_chat_fn():
    def _chat(message, history):
        return rag_chat(cfg, embedder, message, history)
    return _chat

with gr.Blocks(title="Personal RAG (Propositional)") as demo:
    gr.Markdown("# Personal RAG (Propositional) — Local")
    with gr.Tab("Chat"):
        gr.Markdown("Ask questions about your uploaded docs. Answers cite `[doc:page:start-end]`.")
        chat = gr.ChatInterface(
            fn=get_chat_fn(),
            type="messages",
            title="Propositional RAG",
            theme="soft",
            retry_btn="Retry",
            undo_btn="Undo",
        )
    with gr.Tab("Ingest"):
        gr.Markdown("Upload PDF/TXT/MD. The app extracts **propositions** via an LLM, embeds them (BGE-small-en v1.5), and upserts to Pinecone.")
        files = gr.Files(file_types=[".pdf", ".txt", ".md"], label="Files", height=150)
        ns = gr.Textbox(label="Namespace (optional)", value=os.getenv("NAMESPACE","default"))
        out = gr.Textbox(label="Ingest Report", lines=12)
        gr.Button("Ingest").click(ui_ingest, inputs=[files, ns], outputs=out)
    with gr.Tab("Settings"):
        gr.Markdown("Configured via `.env`. Display-only here for safety.")
        with gr.Row():
            gr.Textbox(label="OpenRouter model", value=os.getenv("OPENROUTER_MODEL","openrouter/auto"), interactive=False)
            gr.Textbox(label="Pinecone index", value=os.getenv("PINECONE_INDEX","personal-rag"), interactive=False)
            gr.Textbox(label="Embedding model", value=os.getenv("EMBED_MODEL","BAAI/bge-small-en-v1.5"), interactive=False)
    with gr.Tab("Costs"):
        gr.Markdown("Simple estimator (edit numbers). Always verify current pricing.")
        monthly_queries = gr.Number(label="Monthly Qs", value=200)
        prompt_tk = gr.Number(label="Avg Prompt Tokens", value=600)
        completion_tk = gr.Number(label="Avg Completion Tokens", value=400)
        price_per_1k = gr.Number(label="LLM Price per 1K tokens (USD)", value=0.5)
        base_fixed = gr.Number(label="Base fixed (e.g., vector DB minimum)", value=50.0)
        estimate = gr.Textbox(label="Estimate", lines=2)
        gr.Button("Estimate").click(ui_costs, [monthly_queries, prompt_tk, completion_tk, price_per_1k, base_fixed], estimate)

if __name__ == "__main__":
    demo.launch()
