### **1\. RAG for Historical Estimation Data**

**Goal:** Use existing PRDs, PERT sheets, and effort breakdowns to learn from past estimations and predict new ones accurately.

#### **✅ What to Do:**

* **Data Curation:**

  * Convert PERT sheets, estimation docs into structured JSON or table formats.

  * Label data (if not already) with: `project type`, `module`, `task`, `roles involved`, `estimated time`, `actual time`, etc.

* **Chunking Strategy:**

  * Chunk by `project -> module -> task`.

  * Each chunk should include **task name, context from PRD, estimation**, and **who executed it (Sr/Jr/PM)**.

  * Use semantic chunking (e.g., using `text-splitter` in LangChain).

* **Embeddings \+ Vector DB (RAG):**

  * Use `text-embedding-3-small` or `OpenAI Ada`, or local `InstructorXL`.

  * Store embeddings in a Vector DB like **FAISS**, **Chroma**, or **Weaviate**.

  * Attach metadata: project name, tech stack, effort, estimation deltas.

* **Retrieval Strategy:**

  * At query time, pull top-5 similar past tasks based on PRD context.

  * Inject that into the prompt for LLM to improve output accuracy.

#### **❓Do You Need to Break It Down?**

Yes.

* Raw docs are not useful as-is.

* Chunking \+ structured extraction is **essential** for both retrieval quality and downstream LLM reasoning.

---

### **2\. Code Repository Analysis**

**Goal:** Understand tech stack, structure, complexity, and deployment strategies.

#### **✅ What to Do:**

* **Language Detection & Project Type Classification:**

  * Use `linguist` (GitHub's tool) or custom logic to detect stack.

* **Dependency Graph Analysis:**

  * Use static analysis tools:

    * For Java: `jdt`, `javaparser`

    * For JS/TS: `ts-morph`, `eslint`

    * For Python: `ast`, `pylint`, `pyflakes`

  * Extract:

    * List of services/modules

    * Third-party dependencies

    * Integration points

    * Build tools (Maven, Gradle, npm, etc.)

* **Auto-Summarization of Codebase:**

  * Use LLM \+ repo parsers to create:

    * High-level architecture diagram (textual)

    * Key services and responsibilities

    * Complexity indicators (lines of code, test coverage)

#### **❓What if it's too huge?**

* **Use sampling strategy**:

  * Parse `README`, `docker-compose`, `CI/CD`, and entrypoints first.

  * For monorepos: limit scan to `src`, `apps`, `services` folders only.

  * Limit max tokens by truncating deeply nested files (LLMs won't need them in detail).

---

### **3\. PRD and Supporting Docs Analysis**

**Goal:** Extract modules, features, and scope from natural language documents.

#### **✅ What to Do:**

* **Document Ingestion:**

  * Use document loaders from LangChain (`PDF`, `Markdown`, `Confluence`, etc.)

  * ~~OCR for scanned docs using `Tesseract` or `AWS Textract`.~~

* **Text Embeddings:**

  * Break PRDs into sections by headings.

  * Use `recursive character splitter` or `markdown heading splitter` to chunk logically.

  * Embed each chunk and store for querying in RAG.

* **LLM Analysis Strategy:**

  * Prompt LLM to extract:

    * High-level features/modules

    * Functional requirements

    * Non-functional requirements

    * Risks and unknowns

  * Ask follow-up questions if unclear (`"What does XYZ mean?"`)

#### **❓Should I do anything beyond parsing?**

Yes. Parsing is the baseline.

Do:

* **Semantic chunking**

* **Embedding \+ storing**

* **Role-based extraction** (features meant for frontend/backend/infrastructure etc.)

Also:

* Consider building a **taxonomy** for "types of tasks" so outputs stay consistent.

# Flow diagram

[Estimation tool.drawio](https://drive.google.com/file/d/1QDbWHBBD0Kdg4PMCqNDfSr1y7q7fJNCD/view?usp=sharing)

Estimation tool.drawio