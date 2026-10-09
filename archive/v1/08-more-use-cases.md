# 8. More Useful Small-Model Specializations

## Retrieval-augmented generation (RAG)

Use RAG when facts change or must be attributable. Do not fine-tune facts that should be updated tomorrow.

## Lesson: evaluate a pipeline at each boundary

Run `python3 labs/08_retrieval_evaluation.py`. It implements a minimal lexical retriever and computes recall@k. Replace the documents and query, then see that a perfect generator cannot cite evidence that retrieval never returned.

For query $q$ and document $d$, a retriever assigns score $s(q,d)$. Retrieval recall@k is:

$$\operatorname{Recall@k}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[d_i^* \in \operatorname{topk}(q_i)]$$

Evaluate this before answer quality. If Recall@k is low, training the generator is the wrong first repair.

### Step by step: choose a specialization

1. Write the input, output, error cost, latency target, and evaluation metric.
2. Build the simplest non-generative baseline where applicable: rules for extraction, an embedding classifier for routing, or lexical retrieval for RAG.
3. Freeze a test set representing real ambiguity and class imbalance.
4. Measure each stage independently: OCR, retrieval, classification, parsing, and generation.
5. Add a small generative model only where the baseline fails and human review says the added flexibility is worth the risk.
6. Fine-tune behavior or formatting, not volatile facts.

### Method tradeoffs

| Use case | Prefer first | Why | Main limitation |
| --- | --- | --- | --- |
| RAG Q&A | retriever + constrained generator | provenance and fresh facts | retrieval errors dominate |
| Routing/classification | encoder/embedding classifier | low latency and calibration | fixed label set |
| Extraction | schema + validator + SFT | exact fields can be checked | ambiguous inputs require abstention |
| Translation | parallel data + SFT | preserves task objective | needs language/domain expertise for evaluation |
| Document VLM | compact VLM + schema | handles layout/image evidence | VRAM and OCR/layout error |

1. Chunk documents with stable IDs and metadata.
2. Retrieve top candidates using embeddings and optionally rerank.
3. Prompt the model to answer only from supplied passages, with source IDs.
4. Evaluate retrieval recall separately from answer groundedness.
5. Fine-tune only for query rewriting, answer format, abstention, or citation behavior after retrieval is measured.

## Classification and routing

For intent, triage, moderation labels, and routing, begin with an embedding model or compact encoder. A generative 1B model can work for few-label structured outputs, but a classifier is often cheaper, faster, and easier to calibrate. Report macro F1, per-class recall, calibration, and abstention behavior.

## Information extraction

Use schemas, validators, and constrained decoding. Train examples that include ambiguity and missing fields. Score field-level precision/recall/F1 and exact-record match, not just free-text similarity.

## Translation and rewriting

Build a curated parallel set for the domain and language pair. Evaluate terminology accuracy, preservation of numbers/names, and human adequacy. Do not rely solely on automatic metrics where mistranslation has consequences.

## Speech and document workflows

Small pipelines can be excellent: ASR -> diarization/segmentation -> extraction/summarization -> validation. Measure each stage separately. For scanned documents, add OCR confidence and page-region provenance to prevent the language model from concealing OCR mistakes.

## Multimodal understanding

Visual question answering, document extraction, and UI comprehension need image-text pairs plus an explicit output schema. On 12 GB, use compact VLMs and adapter tuning with low image resolution first. Measure image-grounded accuracy with distractor images and text-only controls.

## Milestone

Choose one specialization whose evaluation can run locally and make it a second independent project. This guards against overfitting your skills to one training recipe.
