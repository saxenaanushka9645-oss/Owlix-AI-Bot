"""
Owlix RAG Chain — full pipeline Steps 0–13
==========================================
BACKEND: Groq Inference API (replaces Hugging Face)

Chat/Reasoning model  : llama-3.3-70b-versatile (Groq — free tier, ultra-fast)
Embedding model       : sentence-transformers/all-MiniLM-L6-v2
                        (tiny 80 MB model, 384-dim, excellent speed/accuracy balance)

Step 0/1  : Input validation & text preprocessing
Step 2    : Query rewriting with conversational memory (ambiguity detection)
Step 3    : Web retrieval via SerpAPI (retry × 3, timeout, empty-results handling)
Step 4    : Source credibility scoring & ranking (trusted-domain tiers)
Step 5    : Deduplication & context-size limiting
Step 6    : Semantic memory retrieval via ChromaDB (graceful skip on failure)
Step 7    : LLM reasoning (Groq API) — retry once, hallucination-risk flagging
Step 8    : Credibility framework — multi-factor confidence computation
Step 9    : Low-confidence handling — additional retrieval + user warning
Step 10   : Response metrics — Precision, Recall, Faithfulness (approximate)
Step 13   : Memory update (Chroma + in-process buffer, failure-safe)
"""

import os
import re
import time
import json
import asyncio
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

import requests

from langchain_chroma import Chroma
from langchain_community.utilities import SerpAPIWrapper

# ChromaDB embedding wrapper for sentence-transformers
from chromadb import EmbeddingFunction, Documents, Embeddings

load_dotenv()

logger = logging.getLogger("owlix.chain")

# ── Environment variables ─────────────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY    = os.getenv("SERPAPI_API_KEY")
#CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma_db")

GROQ_CHAT_MODEL    = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY is not set in .env")
if not SERPAPI_API_KEY:
    raise EnvironmentError("SERPAPI_API_KEY is not set in .env")

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_CONTEXT_CHARS  = 6000
LLM_MAX_RETRIES    = 2
WEB_MAX_RETRIES    = 3
LOW_CONF_THRESHOLD = 0.35
GROQ_MAX_TOKENS    = 1024
GROQ_TEMPERATURE   = 0.3

# ── Step 4: Trusted-domain credibility tiers ─────────────────────────────────
_TIER1 = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "wsj.com",
    "ft.com", "economist.com", "nature.com", "science.org", "nejm.org",
    "who.int", "un.org",
}
_TIER2 = {
    "bloomberg.com", "cnbc.com", "forbes.com", "time.com", "newsweek.com",
    "aljazeera.com", "dw.com", "france24.com", "theatlantic.com",
    "wired.com", "techcrunch.com", "arstechnica.com", "theverge.com",
    "scientificamerican.com", "mit.edu", "stanford.edu", "harvard.edu",
}
_TIER3 = {
    "wikipedia.org", "investopedia.com", "medium.com", "substack.com",
    "zdnet.com", "cnet.com", "engadget.com",
}
LOW_CREDIBILITY_SIGNALS = [
    "clickbait", "viral", "you won't believe", "shocking truth",
    "conspiracy", "fake news", "rumor",
]
HALLUCINATION_SIGNALS = [
    "i don't have", "i cannot find", "no information", "not mentioned",
    "unclear from context", "not in the provided", "i don't know",
]

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Owlix, an AI assistant. You MUST respond with ONLY a valid JSON object.
Do NOT add any text before or after the JSON. Do NOT use markdown code blocks.
Do NOT add backticks. Output raw JSON only.

Detect the language of the user query. Respond in the SAME language.

Required JSON format (use exactly these keys):
{
  "detected_lang": "BCP-47 code e.g. en-US or hi-IN",
  "summary": "Clear explanation of the topic based on provided context",
  "key_findings": "Common facts across sources",
  "key_events": "Important events or actions",
  "contradictions": "Conflicting information or None found",
  "current_status": "Latest situation based on available data",
  "conclusion": "Evidence-based conclusion ONLY from context",
  "uncertainty": "Missing information or ambiguity",
  "confidence": "High or Medium or Low",
  "followups": ["question 1", "question 2", "question 3"]
}

Rules:
- Use ONLY the provided context. Do NOT invent facts.
- If context is insufficient, set confidence to Low.
- followups must be exactly 3 short questions in the same language as the query.
- Output ONLY the JSON object. Nothing else."""

RESOLVE_PROMPT = """You are a query resolver. Given conversation history and a new user query,
output ONLY a fully self-contained search query string.
If the query is ambiguous even after resolution, prefix with [AMBIGUOUS]:

Examples:
- History: "Israel war" | Query: "what India do in this" -> "India role in Israel Gaza war"
- History: (none) | Query: "what about it" -> [AMBIGUOUS]: what about the current topic

Return ONLY the resolved query string, nothing else."""

CLARIFY_PROMPT = """The user's query is ambiguous. Ask ONE short clarifying question.
Respond ONLY with valid JSON:
{"clarification_needed": true, "clarifying_question": "Your question here"}"""


# ─────────────────────────────────────────────────────────────────────────────
# Groq Inference Client
# ─────────────────────────────────────────────────────────────────────────────
class GroqInferenceClient:
    def __init__(self, api_key: str, model: str = GROQ_CHAT_MODEL):
        self.client = Groq(api_key=api_key)
        self.model  = model
        logger.info("Groq client initialised — model: %s", self.model)

    def invoke(self, system: str, user: str) -> str:
        """
        Send a chat completion request to Groq and return the assistant text.
        Raises RuntimeError on non-recoverable errors.
        """
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=GROQ_TEMPERATURE,
                    max_tokens=GROQ_MAX_TOKENS,
                )
                return (response.choices[0].message.content or "").strip()

            except Exception as exc:
                err_str = str(exc).lower()

                # Rate-limit / quota → wait and retry
                if "429" in err_str or "rate_limit" in err_str or "too many" in err_str:
                    wait = 5 * attempt
                    logger.warning(
                        "Groq rate-limited (attempt %d/3), waiting %ds: %s", attempt, wait, exc
                    )
                    if attempt < 3:
                        time.sleep(wait)
                        continue
                    raise RuntimeError(f"Groq rate limit exceeded after retries: {exc}") from exc

                # Service unavailable → short wait and retry
                if "503" in err_str or "service unavailable" in err_str:
                    wait = 3 * attempt
                    logger.warning(
                        "Groq service unavailable (attempt %d/3), waiting %ds: %s", attempt, wait, exc
                    )
                    if attempt < 3:
                        time.sleep(wait)
                        continue
                    raise RuntimeError(f"Groq service unavailable: {exc}") from exc

                # Auth errors — non-recoverable
                if "401" in err_str or "403" in err_str or "invalid_api_key" in err_str:
                    raise RuntimeError(
                        f"Groq API key is invalid or lacks permissions. "
                        f"Check GROQ_API_KEY in your .env file. Details: {exc}"
                    ) from exc

                # All other errors — raise immediately
                raise RuntimeError(f"Groq API error: {exc}") from exc

        raise RuntimeError("Groq invoke failed after all retries.")


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-Transformers embedding function (ChromaDB compatible)
# ─────────────────────────────────────────────────────────────────────────────
class STEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = HF_EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        logger.info("Loaded embedding model: %s", model_name)

    def __call__(self, input: Documents) -> Embeddings:
        vecs = self.model.encode(list(input), show_progress_bar=False)
        return vecs.tolist()


from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

class STLangChainEmbeddings:
    def __init__(self, model_name: str = None):
        self._fn = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._fn([text])[0]

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device="cpu")
        return self.model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        return model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        return model.encode([text], show_progress_bar=False)[0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Text preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_query(raw: str) -> str:
    if not raw or not raw.strip():
        raise ValueError("Query cannot be empty. Please enter a valid question.")

    try:
        text = raw.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"([?!.,;])\1+", r"\1", text)
        for pattern in [
            r"^(hey|hi|hello|yo|hola|bonjour|namaste|hii+|heyy+)[,.\s]+",
            r"^(please|pls|plz|kindly)\s+",
            r"^(can you|could you|would you|will you)\s+",
            r"^(i want to know|tell me|i need to know|i was wondering)\s+",
            r"^(um+|uh+|hmm+|err+)[,.\s]*",
        ]:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

        noise_tokens = {"like", "basically", "literally", "actually", "just", "simply"}
        tokens = text.split()
        while tokens and tokens[0].lower() in noise_tokens:
            tokens = tokens[1:]
        text = " ".join(tokens) if tokens else text
        logger.debug("Preprocessed: %r -> %r", raw, text)
    except Exception as exc:
        logger.warning("Preprocessing failed (%s), using raw: %r", exc, raw)
        text = raw.strip()

    if not text.strip():
        raise ValueError("Query cannot be processed. Please rephrase and enter a valid question.")

    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 (additional): Noise filtering via Ranking
# ─────────────────────────────────────────────────────────────────────────────
_HIGH_NOISE_TOKENS = {
    "um", "uh", "hmm", "err", "like", "so", "okay", "ok", "yeah", "yes",
    "no", "oh", "ah", "well", "right", "alright", "sure", "fine", "good",
    "please", "thanks", "thank", "hello", "hi", "hey", "bye", "bye-bye",
    "hm", "huh", "wow", "cool", "great", "nice", "yep", "nope", "nah",
}

def rank_query_noise(query: str) -> tuple[str, float]:
    tokens = re.findall(r"\b\w{2,}\b", query.lower())
    if not tokens:
        raise ValueError("Query cannot be empty. Please enter a valid question.")

    signal   = [t for t in tokens if t not in _HIGH_NOISE_TOKENS]
    score    = len(signal) / len(tokens)
    question_words = {"what", "who", "when", "where", "why", "how", "which", "whose"}
    if any(t in question_words for t in tokens):
        score = min(score + 0.15, 1.0)
    if any(ch.isdigit() for ch in query):
        score = min(score + 0.10, 1.0)

    logger.debug("Noise rank: score=%.2f signal=%d/%d", score, len(signal), len(tokens))
    if score < 0.25:
        raise ValueError(
            "Query appears to contain mostly noise or filler words. "
            "Please enter a clear, specific question."
        )
    return query, score


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Source credibility scoring
# ─────────────────────────────────────────────────────────────────────────────
def score_source(source: dict) -> float:
    url     = (source.get("url") or "").lower()
    title   = (source.get("title") or "").lower()
    snippet = (source.get("snippet") or "").lower()

    if any(d in url for d in _TIER1) or ".gov" in url or (
        ".edu" in url and not any(d in url for d in _TIER2)
    ):
        score = 1.0
    elif any(d in url for d in _TIER2):
        score = 0.75
    elif any(d in url for d in _TIER3):
        score = 0.50
    else:
        score = 0.35

    if any(sig in f"{title} {snippet}" for sig in LOW_CREDIBILITY_SIGNALS):
        score -= 0.15
    if title and snippet:
        score = min(score + 0.05, 1.0)

    return max(score, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Deduplication
# ─────────────────────────────────────────────────────────────────────────────
def deduplicate_sources(sources: list[dict]) -> list[dict]:
    seen_urls: set[str] = set()
    seen_snippets: list[set] = []
    deduped: list[dict] = []

    for src in sources:
        url     = (src.get("url") or "").strip().rstrip("/")
        snippet = src.get("snippet") or ""
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        words   = set(snippet.lower().split())
        is_dupe = any(
            words and ew and len(words & ew) / len(words | ew) > 0.70
            for ew in seen_snippets
        )
        if not is_dupe:
            seen_snippets.append(words)
            deduped.append(src)

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Multi-factor credibility framework
# ─────────────────────────────────────────────────────────────────────────────
def compute_credibility(
    sources: list[dict],
    raw_response: str,
    query_timestamp: float,
) -> dict:
    try:
        avg_cred = (
            sum(score_source(s) for s in sources) / len(sources) if sources else 0.0
        )

        snippets = [
            set((s.get("snippet") or "").lower().split())
            for s in sources if s.get("snippet")
        ]
        if len(snippets) >= 2:
            all_w  = snippets[0].union(*snippets[1:])
            comm_w = snippets[0].intersection(*snippets[1:])
            agreement = len(comm_w) / len(all_w) if all_w else 0.0
        elif len(snippets) == 1:
            agreement = 0.5
        else:
            agreement = 0.0

        contra_words = [
            "however", "contrary", "disputes", "contradicts",
            "disagrees", "refutes", "debunked",
        ]
        raw_lower   = raw_response.lower()
        contra_cnt  = sum(raw_lower.count(w) for w in contra_words)
        consistency = max(0.0, 1.0 - min(contra_cnt * 0.1, 0.5))

        current_year = str(datetime.now(timezone.utc).year)
        has_recent   = any(
            current_year in (s.get("url", "") + s.get("snippet", ""))
            for s in sources
        )
        time_score = 1.0 if has_recent else 0.6

        bias_signals = [
            "always", "never", "everyone knows", "obviously",
            "clearly", "undeniably", "without a doubt",
        ]
        bias_cnt   = sum(raw_lower.count(b) for b in bias_signals)
        bias_flag  = bias_cnt >= 3
        bias_score = max(0.0, 1.0 - bias_cnt * 0.08)

        composite = (
            avg_cred    * 0.35 +
            agreement   * 0.20 +
            consistency * 0.20 +
            time_score  * 0.15 +
            bias_score  * 0.10
        )
        label = "High" if composite >= 0.70 else ("Medium" if composite >= 0.42 else "Low")

        return {
            "confidence":         label,
            "composite_score":    round(composite, 3),
            "source_credibility": round(avg_cred, 3),
            "source_agreement":   round(agreement, 3),
            "consistency":        round(consistency, 3),
            "time_relevance":     round(time_score, 3),
            "bias_flag":          bias_flag,
        }

    except Exception as exc:
        logger.warning("Credibility computation failed (defaulting to Low): %s", exc)
        return {
            "confidence": "Low", "composite_score": 0.0,
            "source_credibility": 0.0, "source_agreement": 0.0,
            "consistency": 0.0, "time_relevance": 0.0, "bias_flag": False,
        }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10: Response metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(query: str, response: dict, sources: list[dict]) -> dict:
    try:
        summary      = (response.get("summary") or "").lower()
        sum_words    = set(re.findall(r"\b\w{4,}\b", summary))
        source_text  = " ".join(
            (s.get("snippet") or "") + " " + (s.get("title") or "")
            for s in sources
        ).lower()
        source_words = set(re.findall(r"\b\w{4,}\b", source_text))
        query_words  = set(re.findall(r"\b\w{4,}\b", query.lower()))

        precision = (
            len(sum_words & source_words) / len(sum_words) if sum_words else 0.5
        )
        recall = (
            len(query_words & sum_words) / len(query_words) if query_words else 0.5
        )
        full_text    = json.dumps(response).lower()
        hall_hits    = sum(1 for sig in HALLUCINATION_SIGNALS if sig in full_text)
        faithfulness = max(0.0, 1.0 - hall_hits * 0.12)

        return {
            "precision":    round(min(precision, 1.0), 3),
            "recall":       round(min(recall, 1.0), 3),
            "faithfulness": round(faithfulness, 3),
            "coverage":     len(sources),
            "approximate":  True,
        }
    except Exception as exc:
        logger.warning("Metrics computation failed (neutral fallback): %s", exc)
        return {
            "precision": 0.5, "recall": 0.5, "faithfulness": 0.5,
            "coverage": len(sources), "approximate": True,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Conversation memory (Steps 2 & 13)
# ─────────────────────────────────────────────────────────────────────────────
class SimpleMemory:
    def __init__(self):
        self.history: list[dict] = []

    def add(self, human: str, ai: str):
        self.history.append({"human": human, "ai": ai})
        if len(self.history) > 6:
            self.history = self.history[-6:]

    def as_text(self) -> str:
        if not self.history:
            return "No prior conversation."
        return "\n".join(
            f"User: {t['human']}\nOwlix: {t['ai']}" for t in self.history
        )

    def clear(self):
        self.history = []


# ─────────────────────────────────────────────────────────────────────────────
# Robust JSON extractor
# ─────────────────────────────────────────────────────────────────────────────
def extract_json_from_llm_output(raw_text: str) -> dict:
    if not raw_text:
        raise ValueError("Empty LLM response")

    cleaned = raw_text
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_candidate = cleaned[start:end+1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    fixed = re.sub(r"'([^']*)'", lambda m: '"' + m.group(1).replace('"', '\\"') + '"', fixed)
    start = fixed.find("{")
    end   = fixed.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(fixed[start:end+1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM output: {raw_text[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main chain
# ─────────────────────────────────────────────────────────────────────────────
class OwlixChain:
    def __init__(self):
        logger.info("Initialising OwlixChain (Groq backend)...")
        logger.info("Chat model  : %s", GROQ_CHAT_MODEL)
        logger.info("Embed model : %s", HF_EMBEDDING_MODEL)

        self.llm        = GroqInferenceClient(GROQ_API_KEY, GROQ_CHAT_MODEL)
        self.embeddings = STLangChainEmbeddings(HF_EMBEDDING_MODEL)

        self.vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name="owlix_memory",
        )

        self.search   = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
        self.memories: dict[str, SimpleMemory] = {}
        logger.info("OwlixChain ready.")

    def _get_memory(self, session_id: str) -> SimpleMemory:
        if session_id not in self.memories:
            self.memories[session_id] = SimpleMemory()
        return self.memories[session_id]

    def clear_memory(self, session_id: str):
        if session_id in self.memories:
            self.memories[session_id].clear()

    def _resolve_query_sync(self, raw: str, history: str) -> str:
        if history == "No prior conversation.":
            return raw
        try:
            user_msg = f"Conversation History:\n{history}\n\nNew Query: {raw}"
            resolved = self.llm.invoke(RESOLVE_PROMPT, user_msg).strip().strip('"').strip("'")
            return resolved or raw
        except Exception as exc:
            logger.warning("Query resolution failed: %s", exc)
            return raw

    def _retrieve_web_sync(
        self, query: str, max_retries: int = WEB_MAX_RETRIES
    ) -> tuple[str, list[dict]]:
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                raw     = self.search.results(query)
                organic = raw.get("organic_results", [])
                if not organic:
                    return "__NO_RESULTS__", []
                sources, parts = [], []
                for r in organic[:8]:
                    title   = r.get("title", "")
                    snippet = r.get("snippet", "")
                    link    = r.get("link", "")
                    sources.append({"title": title, "url": link, "snippet": snippet})
                    parts.append(f"Source: {title}\nURL: {link}\nContent: {snippet}")
                return "\n\n".join(parts), sources
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Web retrieval attempt %d/%d failed: %s", attempt, max_retries, exc
                )
                if attempt < max_retries:
                    time.sleep(1.5 * attempt)

        err = str(last_error or "")
        if "timeout" in err.lower():
            return "__TIMEOUT__", []
        return f"__API_FAIL__:{err}", []

    def _retrieve_memory_sync(self, query: str) -> str:
        try:
            docs = self.vectorstore.similarity_search(query, k=3)
            return (
                "\n\n".join(f"[Memory] {d.page_content}" for d in docs)
                if docs else "No relevant memory found."
            )
        except Exception as exc:
            logger.warning("Memory retrieval failed (skipping): %s", exc)
            return "Memory retrieval unavailable."

    def _llm_invoke_sync(self, system: str, user: str) -> str:
        return self.llm.invoke(system, user)

    def _store_in_chroma_sync(self, query: str, summary: str):
        try:
            self.vectorstore.add_texts([f"Q: {query}\nA: {summary}"])
        except Exception as exc:
            logger.warning("Chroma store failed (non-fatal): %s", exc)

    async def run(self, query: str, session_id: str = "default") -> dict:
        start_ts = time.time()
        memory   = self._get_memory(session_id)
        history  = memory.as_text()

        try:
            clean = preprocess_query(query)
        except ValueError as ve:
            return self._error_response(str(ve), query)

        try:
            clean, noise_score = rank_query_noise(clean)
            logger.info("[%s] Noise rank score: %.2f", session_id, noise_score)
        except ValueError as ve:
            return self._error_response(str(ve), query)

        logger.info("[%s] Clean query: %r", session_id, clean)

        resolved  = await asyncio.to_thread(self._resolve_query_sync, clean, history)
        ambiguous = False

        if resolved.startswith("[AMBIGUOUS]:"):
            ambiguous = True
            resolved  = resolved[len("[AMBIGUOUS]:"):].strip()

        if ambiguous and history == "No prior conversation.":
            try:
                resp = await asyncio.to_thread(
                    self._llm_invoke_sync, CLARIFY_PROMPT, f"Query: {clean}",
                )
                data = extract_json_from_llm_output(resp)
                if data.get("clarification_needed"):
                    return {
                        "clarification_needed": True,
                        "summary": data.get("clarifying_question", "Could you provide more context?"),
                        "key_findings": "—", "key_events": "—", "contradictions": "—",
                        "current_status": "—", "conclusion": "—",
                        "uncertainty": "Query is ambiguous — awaiting clarification.",
                        "confidence": "Low",
                        "followups": ["Can you clarify?", "Tell me more.", "Any examples?"],
                        "sources": [], "resolved_query": resolved,
                        "metrics": {}, "credibility_report": {},
                    }
            except Exception as exc:
                logger.warning("Clarification call failed (proceeding): %s", exc)

        (raw_ctx, raw_sources), mem_ctx = await asyncio.gather(
            asyncio.to_thread(self._retrieve_web_sync, resolved),
            asyncio.to_thread(self._retrieve_memory_sync, resolved),
        )

        if raw_ctx == "__NO_RESULTS__":
            raw_sources, raw_ctx = [], ""
        elif raw_ctx == "__TIMEOUT__":
            return self._error_response("Request timed out. Please try again.", query, resolved)
        elif raw_ctx.startswith("__API_FAIL__:"):
            return self._error_response(
                "Unable to fetch data from sources. Please try again later.", query, resolved
            )

        if raw_sources:
            scored = sorted(
                [(s, score_source(s)) for s in raw_sources],
                key=lambda x: x[1], reverse=True,
            )
            top5   = scored[:5]
            avg_sc = sum(sc for _, sc in top5) / len(top5) if top5 else 0
            if avg_sc < 0.20:
                return self._error_response(
                    "Not enough reliable information available. "
                    "Please try rephrasing or a different topic.",
                    query, resolved,
                )
            ranked = [s for s, _ in top5]
            for i, (s, sc) in enumerate(top5):
                ranked[i]["credibility_score"] = round(sc, 2)
        else:
            ranked = []

        deduped = deduplicate_sources(ranked)
        web_ctx = "\n\n".join(
            f"Source: {s.get('title','')}\nURL: {s.get('url','')}\n"
            f"Credibility: {s.get('credibility_score',0):.2f}\n"
            f"Content: {s.get('snippet','')}"
            for s in deduped
        )
        if len(web_ctx) > MAX_CONTEXT_CHARS:
            web_ctx = web_ctx[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated]"

        if not web_ctx.strip() and mem_ctx in (
            "No relevant memory found.", "Memory retrieval unavailable."
        ):
            return self._error_response(
                "Insufficient context to generate a response. "
                "Please try a more specific query.",
                query, resolved,
            )

        user_msg = (
            f"Original User Query: {clean}\n"
            f"Resolved Query: {resolved}\n\n"
            f"Context (SerpAPI — ranked sources):\n"
            f"{web_ctx or 'No web context available.'}\n\n"
            f"Semantic Memory (Chroma):\n{mem_ctx}\n\n"
            f"Conversation History:\n{history}\n\n"
            "IMPORTANT: Respond ONLY with a JSON object. No markdown, no backticks, no extra text."
        )

        raw_text  = ""
        llm_error = None
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                raw_text  = await asyncio.to_thread(
                    self._llm_invoke_sync, SYSTEM_PROMPT, user_msg
                )
                llm_error = None
                break
            except Exception as exc:
                llm_error = exc
                logger.warning("[%s] LLM attempt %d/%d failed: %s", session_id, attempt, LLM_MAX_RETRIES, exc)
                if attempt < LLM_MAX_RETRIES:
                    await asyncio.sleep(2.0)

        if llm_error:
            return self._error_response(
                f"AI processing failed: {llm_error}. Please try again.", query, resolved
            )

        logger.info("[%s] LLM responded (%d chars): %s", session_id, len(raw_text), raw_text[:100])

        try:
            parsed = extract_json_from_llm_output(raw_text)
        except Exception as exc:
            logger.warning("[%s] JSON parse failed (%s) — using fallback. Raw: %s", session_id, exc, raw_text[:300])
            parsed = {
                "summary":        raw_text[:800] if raw_text else "Could not parse AI response.",
                "key_findings":   "—", "key_events": "—", "contradictions": "—",
                "current_status": "—", "conclusion": "—",
                "uncertainty":    "Response format error — displaying raw output.",
                "confidence":     "Low",
                "followups":      ["Can you rephrase?", "Tell me more.", "Any examples?"],
            }

        summary_lower = (parsed.get("summary") or "").lower()
        if any(sig in summary_lower for sig in HALLUCINATION_SIGNALS):
            parsed["uncertainty"] = (
                "⚠ Hallucination risk: the model indicated limited context. "
                + (parsed.get("uncertainty") or "")
            ).strip()
            parsed["confidence"] = "Low"

        cred_report   = compute_credibility(deduped, raw_text, start_ts)
        computed_conf = cred_report["confidence"]
        parsed["confidence"] = computed_conf

        if cred_report.get("bias_flag"):
            parsed["uncertainty"] = (
                "⚠ Potential bias detected in source language. "
                + (parsed.get("uncertainty") or "")
            ).strip()

        if computed_conf == "Low":
            broadened = f"{resolved} latest analysis expert opinion"
            extra_ctx, extra_sources = await asyncio.to_thread(
                self._retrieve_web_sync, broadened, 2
            )
            if extra_sources and not extra_ctx.startswith("__"):
                extra_scored = sorted(
                    [(s, score_source(s)) for s in extra_sources],
                    key=lambda x: x[1], reverse=True,
                )[:3]
                for ex_s, ex_sc in extra_scored:
                    ex_s["credibility_score"] = round(ex_sc, 2)
                    deduped.append(ex_s)
                deduped = deduplicate_sources(deduped)

            parsed["uncertainty"] = (
                "⚠ Information may not be fully reliable due to limited, "
                "conflicting, or low-credibility sources. "
                + (parsed.get("uncertainty") or "")
            ).strip()

        metrics = compute_metrics(clean, parsed, deduped)

        summary = parsed.get("summary", "")
        memory.add(clean, summary)
        await asyncio.to_thread(self._store_in_chroma_sync, resolved, summary)

        parsed["sources"]            = deduped
        parsed["resolved_query"]     = resolved
        parsed["metrics"]            = metrics
        parsed["credibility_report"] = cred_report
        return parsed

    @staticmethod
    def _error_response(msg: str, original: str, resolved: str = "") -> dict:
        return {
            "summary":            msg,
            "key_findings":       "—", "key_events": "—", "contradictions": "—",
            "current_status":     "—", "conclusion": "—",
            "uncertainty":        msg,
            "confidence":         "Low",
            "followups": [
                "Can you rephrase the query?",
                "Try a more specific question.",
                "Check your internet connection.",
            ],
            "sources":            [],
            "resolved_query":     resolved or original,
            "metrics":            {"approximate": True},
            "credibility_report": {"confidence": "Low"},
        }