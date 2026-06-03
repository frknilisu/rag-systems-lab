# n8n Blueprint · Standard RAG

This page describes how to rebuild the Standard RAG pipeline as an n8n workflow.
Each Python step maps to one n8n node. The `RAGState` object between steps maps
to the n8n item JSON flowing between nodes.

## Workflow overview

```
Webhook (POST /query)
  → Embed Question
  → Chroma Vector Search
  → Build Prompt
  → LLM (OpenAI / Ollama / Anthropic)
  → Respond
```

---

## Node 1 · Webhook

**Type:** Webhook  
**Method:** POST  
**Path:** `/query`

**Input body:**
```json
{
  "question": "What is the maximum torque for the XR-500 motor?",
  "session_id": "optional-string"
}
```

**Output item (passed to next node):**
```json
{
  "question": "What is the maximum torque for the XR-500 motor?",
  "session_id": null,
  "retrieval_results": [],
  "answer": ""
}
```

---

## Node 2 · Embed Question

**Type:** HTTP Request (POST to `http://localhost:8000/embed`)  
*— or —*  
**Type:** Code (if embedding runs inline)

Calls the RAG-Lab embedding endpoint:

```
POST /embed
{ "text": "{{ $json.question }}" }
→ { "embedding": [0.023, -0.418, ...] }   // 384-dim vector
```

If using the RAG-Lab FastAPI service (`make serve`), the `/embed` endpoint is
available automatically. Otherwise, call your embedding provider (OpenAI
`text-embedding-3-small`, Cohere, etc.) directly.

**Output item:**
```json
{
  "question": "...",
  "question_embedding": [0.023, -0.418, ...]
}
```

---

## Node 3 · Vector Search (Chroma)

**Type:** HTTP Request  

```
POST http://localhost:8000/search
{
  "vector": {{ $json.question_embedding }},
  "top_k": 5
}
→ {
    "results": [
      { "chunk_id": "doc1__chunk_0003", "text": "...", "score": 0.82 },
      ...
    ]
  }
```

Alternatively, call the Chroma HTTP API directly (`http://localhost:8001`).
The RAG-Lab service wraps Chroma with the same interface and score normalization
(cosine distance → similarity) that the Python implementation uses.

**Output item:**
```json
{
  "question": "...",
  "retrieval_results": [
    { "text": "...", "score": 0.82, "rank": 0 },
    ...
  ]
}
```

---

## Node 4 · Build Prompt

**Type:** Code (JavaScript)

Formats the retrieved chunks and question into the LLM message list. Paste this
code node verbatim — the system prompt text comes from
`generation/prompts/rag_system.txt`:

```javascript
const systemPrompt = `You are a helpful assistant. Answer the question using ONLY the provided context.

Rules:
- Base your answer entirely on the context below. Do not use outside knowledge.
- If the context does not contain enough information, say exactly:
  "The provided documents don't contain enough information to answer this question."
- Be concise. Quote or paraphrase the relevant part of the context when helpful.
- Never make up facts or invent details not present in the context.`;

const contexts = $json.retrieval_results
  .map((r, i) => `[Source ${i + 1}]\n${r.text}`)
  .join('\n\n---\n\n');

const userContent = `Context:\n${contexts}\n\nQuestion: ${$json.question}`;

return [{
  json: {
    ...$json,
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user",   content: userContent  }
    ]
  }
}];
```

---

## Node 5 · LLM

**Type:** OpenAI Chat Model (or Ollama, Anthropic, etc.)

Connect the `messages` array from the previous node:

| Setting | Value |
|---------|-------|
| Messages | `{{ $json.messages }}` |
| Model | `gpt-4o-mini` / `qwen2.5:3b` / etc. |
| Temperature | `0` (for factual, grounded answers) |
| Max tokens | `1024` |

**Output item:**
```json
{
  "question": "...",
  "retrieval_results": [...],
  "answer": "The XR-500 motor has a maximum continuous torque of 42 N·m..."
}
```

---

## Node 6 · Respond

**Type:** Respond to Webhook

```json
{
  "answer": "{{ $json.answer }}",
  "contexts": "{{ $json.retrieval_results }}",
  "question": "{{ $json.question }}"
}
```

---

## Item JSON shape between nodes

The item flowing through the workflow mirrors `RAGState` exactly:

```json
{
  "question":           "string — the user's question",
  "session_id":         "string | null",
  "question_embedding": "[float, ...]  — added by Embed node",
  "retrieval_results": [
    { "text": "string", "score": 0.82, "rank": 0, "chunk_id": "string" }
  ],
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user",   "content": "..." }
  ],
  "answer": "string — filled by LLM node"
}
```

Fields are added incrementally as items flow through nodes. No node overwrites
what a previous node wrote — it only adds its own output field.

---

## Calling the Python service instead

If you prefer to keep the Python implementation canonical and call it from n8n
rather than re-implementing each node:

```
POST http://localhost:8000/query
{
  "question": "{{ $json.question }}",
  "architecture": "standard"
}
→ RAGResult JSON
```

This trades node-level observability for simplicity. Both approaches are valid
depending on whether you're learning (node-by-node) or shipping (service call).

---

## Differences from the Python implementation

| Python | n8n |
|--------|-----|
| `retrieve_step` | Nodes 2 + 3 (embed + search are separate nodes) |
| `generate_step` | Nodes 4 + 5 (prompt build + LLM are separate nodes) |
| `PipelineDeps` injection | n8n credentials (API keys in credential store) |
| `StepTrace` | n8n execution log (each node shows input/output) |
| `RAGState` | n8n item JSON flowing between nodes |

The Python implementation combines embed + search into one `retrieve_step` for
conciseness; in n8n they are separate nodes because the HTTP calls are distinct
operations. The observable behavior is identical.
