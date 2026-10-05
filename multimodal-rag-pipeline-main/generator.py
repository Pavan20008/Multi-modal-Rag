"""
Generates the final answer using a local, free, open-source Vision-Language
Model served through Ollama (e.g. `ollama pull llava`).
"""
import base64
import ollama

import config


def _encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_prompt(query, text_chunks):
    context = "\n\n".join(
        f"[Page {c['page']}]: {c['content']}" for c in text_chunks
    )
    return (
        "You are a helpful assistant answering questions about a technical "
        "document. Use the text context and the attached images (diagrams, "
        "screenshots, tables) to answer as accurately as possible.\n\n"
        "Rules for using the evidence:\n"
        "- If the question refers to a specific diagram, figure, table, or "
        "section, prioritize evidence from that specific source.\n"
        "- If diagram, table, or figure images are attached, visually "
        "inspect them and describe what they actually show - layout, "
        "icons, labels, values, structure - as part of your answer. Do "
        "not just summarize nearby text instead of looking at the image; "
        "if the question asks what an image/diagram shows, your answer "
        "must describe the image's visual content specifically.\n"
        "- Do not combine information from unrelated retrieved pages.\n"
        "- If multiple retrieved pages conflict or contain information from "
        "different sections, use only the page(s) relevant to the question.\n"
        "- Do not add information merely because it is semantically related "
        "— only use it if it directly answers the question.\n"
        "- When the context contains a numbered or bulleted list, steps, a "
        "set of conditions (e.g. warning-light triggers, procedure steps), "
        "or a table with multiple rows/variants, include every relevant "
        "item or row in your answer, not just the first or most prominent "
        "one. Dropping list items or table rows when summarizing is a "
        "common mistake - double check your answer against the context "
        "and add back anything you left out before responding. That said, "
        "do not pad your answer with fabricated restatements or "
        "paraphrased summaries beyond what the context actually says - "
        "every added detail must be traceable to the context, with the "
        "same specific numbers/wording, not an approximation of them.\n"
        "- Only cite a specific section/chapter reference (e.g. \"refer to "
        "X section\") if it appears in the context directly attached to "
        "the topic being asked about. If the context contains multiple "
        "distinct subsections (e.g. two different warning lights "
        "described back to back), do not attach one subsection's "
        "reference note to another subsection's answer.\n"
        "- Be careful to distinguish between different but similarly-"
        "labeled quantities in the context (e.g. total fluid capacity vs. "
        "a smaller top-up/refill amount, or different engine/vehicle "
        "variants). Only pair a specific number with a specific label "
        "(engine name, variant, part) if that exact pairing appears "
        "together in the same table row or sentence in the context - "
        "never attach a number to a label just because both appear "
        "somewhere in the retrieved context.\n"
        "- Dense multi-cell grids (e.g. fuse/relay panels) are hard to read "
        "precisely even when visible. If you report a specific numeric "
        "rating (amps, etc.) from such a grid, add a brief note that the "
        "user should confirm the exact value against the diagram image "
        "directly before relying on it (e.g. for actual fuse replacement).\n"
        "- If the answer isn't in the provided context, say so explicitly "
        "rather than guessing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def generate_answer(query, text_chunks, images, model=config.VLM_MODEL):
    prompt = build_prompt(query, text_chunks)
    image_data = [_encode_image(img["path"]) for img in images]

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": image_data,
            }
        ],
        options={
            "num_ctx": 8192,  # default Ollama context (4096) is too small once
                               # several text chunks + image tokens are combined
        },
    )
    return response["message"]["content"]