# Model specification — steps 1–3

Written to close the gap the guide warns about: *"Create the test set now. If you
wait until after training, it is easy to accidentally design tests that favor the
trained model."* Everything here is decided **before** looking at any trained
candidate.

---

## 1. Training versus real-time inference

Per the guide's golden rule, weights change **only** through reviewed releases.

| Operation | When | Changes weights? |
|---|---|---|
| Fine-tuning | Scheduled candidate builds | **Yes** |
| RAG retrieval | Every relevant request | No |
| Inference | Every request | No |
| Feedback collection | During production | No |
| Controlled retraining | When enough approved data exists | **Yes** |

**No online learning.** The service never updates weights from user traffic.
Production logs flow to a separate review pipeline; only redacted, reviewed,
labelled examples enter a versioned training set.

---

## 2. Use case and measurable targets

> **One sentence:** A news question-answering assistant for Andhra Pradesh and
> India that answers from retrieved news articles, cites the article it used, and
> **abstains when retrieval returns nothing relevant**.

| | |
|---|---|
| **Input** | A natural-language question, ≤ 500 characters, English |
| **Output** | A short answer (≤ 120 words) plus the source article id, or an explicit abstention |
| **Languages** | English only in v1 |
| **Latency target** | P95 under 3 seconds |
| **Prohibited** | Medical, legal or financial advice; claims with no retrieved evidence; personal data about private individuals; arithmetic presented as authoritative |

### Targets and how each is tested

| Metric | Target | Test |
|---|---|---|
| Correct answers | ≥ 70% | Human-reviewed held-out set (`testset.py`) |
| **Groundedness** | ≥ 95% | Every claim traceable to a retrieved article |
| **Abstention when no evidence** | ≥ 90% | Unanswerable questions in the test set |
| Hallucination | ≤ 5% | Claims checked against evidence |
| Latency P95 | < 3 s | Measured over the test set |
| Format validity | ≥ 99% | Schema validation on API responses |
| Unsafe critical output | **zero** | Adversarial safety suite (`test_safety.py`) |

Targets are deliberately lower than the guide's examples (90% / 2%) because the
serving model is small. **A target you cannot hit is not a target.**

### Example

```
Q: Where is Hyderabad?
A: Hyderabad is the capital of Telangana, in southern India. [source: art_0142]

Q: What will the Sensex close at tomorrow?
A: I don't have information about that. (no relevant article retrieved)
```

The second is as important as the first. **Abstention is a feature**, and the
failure we actually observed — inventing *"Telangana is the proposed name for
India's central government"* — is exactly what it prevents.

---

## 3. RAG, fine-tuning, or both

| Requirement | Approach | Why |
|---|---|---|
| News facts that change daily | **RAG** | Updatable, removable, citable without retraining |
| Answer shape, citation format, abstention wording | **Fine-tuning** | Repeated behaviour and structure |
| Both together | **Fine-tune + RAG** | Behaviour from weights, facts from retrieval |

### The decision, and the mistake it corrects

The guide is explicit: *"Do not fine-tune a model merely to memorize documents.
Store those documents in a retrieval system so they can be updated, removed and
cited."*

Our first two fine-tunes violated this:

1. **5,249 doc-summary pairs** — taught the model to reproduce documentation, which
   is why it echoed retrieved text back verbatim and emitted a
   `"(P.S. Visit us at [address]...)"` template.
2. **22 memorised facts** — makes those 22 questions right and nothing else, and
   the facts cannot be corrected without retraining.

**v1 decision:** facts come from RAG. Fine-tuning is used only to teach
*answer-with-citation-or-abstain* behaviour — a format, which is what fine-tuning
is actually good at.

---

## Base model (step 4)

| | |
|---|---|
| Serving candidate | `Qwen2.5-1.5B-Instruct` or `llama3.2:3b` via Ollama |
| Rejected | `SmolLM2-135M` — measured: copies retrieved context instead of reading it |
| Licence | Apache-2.0 / Llama community licence — both permit the intended use |

The 135M model is kept for pipeline development only. Measured evidence: given
retrieved context it reproduced the context verbatim rather than extracting an
answer, and scored 1.15 bits entropy on `"2 + 2 ="` while predicting a space.
