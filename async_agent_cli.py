import os
import re
import sys
import json
import ast
import asyncio
import uuid
import ctypes
import subprocess
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# Optional cloud providers — imported defensively so the tool still runs fine with
# only Ollama installed. If a package isn't installed, that provider is simply
# treated as "not ready" by provider_ready() and the router skips straight past it.
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

# Textual UI components
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Input, Button, RichLog, DirectoryTree, Label, Static

# --- MULTI-MODEL REGISTRY ---
# Every model the team is allowed to use, and what it takes to actually reach it.
# "ollama" entries are always considered ready (same assumption the tool has always
# made: a local Ollama server is running). Cloud entries only light up once their
# env vars are set — until then the router silently skips them and nothing breaks.
#
# NOTE on "NVIDIA" and "NEON": NVIDIA NIM endpoints are OpenAI-API-compatible, so
# that's wired as a real, working provider below (needs NVIDIA_API_KEY). "NEON" is
# not a known LLM API as of this writing (Neon is a Postgres hosting company) — it's
# wired as a generic OpenAI-compatible custom slot instead: set NEON_API_KEY,
# NEON_BASE_URL, and NEON_MODEL to point it at whatever endpoint you actually mean,
# and it'll work exactly like any other provider. Same pattern covers "Claude Code"
# as a model string — real Claude models are called through the standard Anthropic
# API below; "Claude Code" itself is a CLI product, not a model you can call.
DEFAULT_FALLBACK_MODEL = "qwen2.5-coder:7b"  # always assumed available — the tool's original baseline

MODEL_REGISTRY = {
    # --- Local Ollama models: genuinely free — no API key, no per-token cost, run
    # entirely on your own machine. This is the only tier that's actually "free" in
    # the strict sense; everything below needs its own account with that provider.
    "qwen2.5-coder:7b": {
        "provider": "ollama", "tier": "fast", "temperature": 0.1,
        "good_for": ("trivial", "standard"),
    },
    "qwen2.5-coder:14b": {
        "provider": "ollama", "tier": "balanced", "temperature": 0.1,
        "good_for": ("standard", "foundational"),
    },
    "qwen2.5-coder:32b": {
        "provider": "ollama", "tier": "balanced", "temperature": 0.1,
        "good_for": ("foundational",),
    },
    "deepseek-r1:8b": {
        "provider": "ollama", "tier": "reasoning", "temperature": 0.2,
        "good_for": ("foundational", "planning", "review"),
    },
    "deepseek-r1:14b": {
        "provider": "ollama", "tier": "reasoning", "temperature": 0.2,
        "good_for": ("foundational", "planning", "review"),
    },
    "deepseek-r1:32b": {
        "provider": "ollama", "tier": "reasoning", "temperature": 0.2,
        "good_for": ("planning", "review"),
    },
    "llama3.1:8b": {
        "provider": "ollama", "tier": "fast", "temperature": 0.1,
        "good_for": ("trivial", "standard"),
    },
    "llama3.1:70b": {
        "provider": "ollama", "tier": "balanced", "temperature": 0.1,
        "good_for": ("foundational", "review"),
    },
    "llama3.2:3b": {
        "provider": "ollama", "tier": "fast", "temperature": 0.1,
        "good_for": ("trivial",),
    },
    "mistral:7b": {
        "provider": "ollama", "tier": "fast", "temperature": 0.1,
        "good_for": ("trivial", "standard"),
    },
    "mixtral:8x7b": {
        "provider": "ollama", "tier": "balanced", "temperature": 0.1,
        "good_for": ("standard", "foundational"),
    },
    "codellama:13b": {
        "provider": "ollama", "tier": "fast", "temperature": 0.1,
        "good_for": ("standard",),
    },
    "phi3:14b": {
        "provider": "ollama", "tier": "fast", "temperature": 0.1,
        "good_for": ("trivial", "standard"),
    },
    "gemma2:9b": {
        "provider": "ollama", "tier": "fast", "temperature": 0.1,
        "good_for": ("trivial", "standard"),
    },
    "starcoder2:15b": {
        "provider": "ollama", "tier": "balanced", "temperature": 0.1,
        "good_for": ("standard",),
    },

    # # --- OpenAI (requires OPENAI_API_KEY + billing on your account) ---
    # "gpt-4.1": {
    #     "provider": "openai", "model_name": "gpt-4.1", "tier": "premium", "temperature": 0.1,
    #     "requires_env": ["OPENAI_API_KEY"], "good_for": ("foundational", "planning", "review"),
    # },
    # "gpt-4o": {
    #     "provider": "openai", "model_name": "gpt-4o", "tier": "premium", "temperature": 0.1,
    #     "requires_env": ["OPENAI_API_KEY"], "good_for": ("foundational", "planning", "review"),
    # },
    # "gpt-4o-mini": {
    #     "provider": "openai", "model_name": "gpt-4o-mini", "tier": "balanced", "temperature": 0.1,
    #     "requires_env": ["OPENAI_API_KEY"], "good_for": ("standard",),
    # },
    # "o3-mini": {
    #     "provider": "openai", "model_name": "o3-mini", "tier": "reasoning", "temperature": 0.2,
    #     "requires_env": ["OPENAI_API_KEY"], "good_for": ("planning", "review"),
    # },

    # # --- Anthropic (requires ANTHROPIC_API_KEY + billing on your account) ---
    # "claude-sonnet-4-5": {
    #     "provider": "anthropic", "model_name": "claude-sonnet-4-5", "tier": "premium", "temperature": 0.1,
    #     "requires_env": ["ANTHROPIC_API_KEY"], "good_for": ("foundational", "planning", "review"),
    # },
    # "claude-opus-4-1": {
    #     "provider": "anthropic", "model_name": "claude-opus-4-1", "tier": "premium", "temperature": 0.1,
    #     "requires_env": ["ANTHROPIC_API_KEY"], "good_for": ("planning", "review"),
    # },
    # "claude-haiku-4-5": {
    #     "provider": "anthropic", "model_name": "claude-haiku-4-5", "tier": "balanced", "temperature": 0.1,
    #     "requires_env": ["ANTHROPIC_API_KEY"], "good_for": ("standard",),
    # },

    # # --- Google Gemini, via Google's OpenAI-compatible endpoint (needs GOOGLE_API_KEY) ---
    # "gemini-2.0-flash": {
    #     "provider": "openai_compatible", "model_name": "gemini-2.0-flash",
    #     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    #     "tier": "balanced", "temperature": 0.1,
    #     "requires_env": ["GOOGLE_API_KEY"], "good_for": ("standard", "foundational"),
    # },

    # # --- Mistral API (needs MISTRAL_API_KEY) ---
    # "mistral-large-latest": {
    #     "provider": "openai_compatible", "model_name": "mistral-large-latest",
    #     "base_url": "https://api.mistral.ai/v1", "tier": "premium", "temperature": 0.1,
    #     "requires_env": ["MISTRAL_API_KEY"], "good_for": ("foundational", "review"),
    # },

    # # --- xAI Grok (needs XAI_API_KEY) ---
    # "grok-2": {
    #     "provider": "openai_compatible", "model_name": "grok-2",
    #     "base_url": "https://api.x.ai/v1", "tier": "premium", "temperature": 0.1,
    #     "requires_env": ["XAI_API_KEY"], "good_for": ("foundational", "review"),
    # },

    # # --- DeepSeek's own hosted API, distinct from the local Ollama deepseek-r1
    # # models above (needs DEEPSEEK_API_KEY) ---
    # "deepseek-chat": {
    #     "provider": "openai_compatible", "model_name": "deepseek-chat",
    #     "base_url": "https://api.deepseek.com", "tier": "balanced", "temperature": 0.1,
    #     "requires_env": ["DEEPSEEK_API_KEY"], "good_for": ("standard", "foundational"),
    # },

    # # --- Groq: hosted inference of open models, notably fast, and Groq's free tier
    # # is real (needs GROQ_API_KEY — check their current free-tier limits) ---
    # "groq-llama-3.3-70b": {
    #     "provider": "openai_compatible", "model_name": "llama-3.3-70b-versatile",
    #     "base_url": "https://api.groq.com/openai/v1", "tier": "balanced", "temperature": 0.1,
    #     "requires_env": ["GROQ_API_KEY"], "good_for": ("standard", "foundational", "review"),
    # },

    # # --- Perplexity (needs PERPLEXITY_API_KEY) ---
    # "perplexity-sonar": {
    #     "provider": "openai_compatible", "model_name": "sonar",
    #     "base_url": "https://api.perplexity.ai", "tier": "balanced", "temperature": 0.1,
    #     "requires_env": ["PERPLEXITY_API_KEY"], "good_for": ("standard",),
    # },

    # # --- NVIDIA NIM (needs NVIDIA_API_KEY) — spread across tiers since NVIDIA's
    # # catalog (build.nvidia.com) hosts models at every size. Model IDs below are the
    # # commonly-used ones as of this writing; NVIDIA occasionally renames/adds
    # # models in their catalog, so it's worth double-checking these against
    # # build.nvidia.com if a call starts failing. ---
    # "nvidia/llama-3.1-8b-instruct": {
    #     "provider": "openai_compatible", "model_name": "meta/llama-3.1-8b-instruct",
    #     "base_url": "https://integrate.api.nvidia.com/v1", "tier": "fast", "temperature": 0.1,
    #     "requires_env": ["NVIDIA_API_KEY"], "good_for": ("trivial", "standard"),
    # },
    # "nvidia/mixtral-8x7b-instruct": {
    #     "provider": "openai_compatible", "model_name": "mistralai/mixtral-8x7b-instruct-v0.1",
    #     "base_url": "https://integrate.api.nvidia.com/v1", "tier": "balanced", "temperature": 0.1,
    #     "requires_env": ["NVIDIA_API_KEY"], "good_for": ("standard", "foundational"),
    # },
    # "nvidia/llama-3.1-70b-instruct": {
    #     "provider": "openai_compatible", "model_name": "meta/llama-3.1-70b-instruct",
    #     "base_url": "https://integrate.api.nvidia.com/v1", "tier": "premium", "temperature": 0.1,
    #     "requires_env": ["NVIDIA_API_KEY"], "good_for": ("foundational", "planning"),
    # },
    # "nvidia/llama-3.1-nemotron-70b-instruct": {
    #     "provider": "openai_compatible", "model_name": "nvidia/llama-3.1-nemotron-70b-instruct",
    #     "base_url": "https://integrate.api.nvidia.com/v1", "tier": "premium", "temperature": 0.1,
    #     "requires_env": ["NVIDIA_API_KEY"], "good_for": ("foundational", "planning", "review"),
    # },

    # # --- OpenRouter: one API key, access to a huge catalog of models from every lab,
    # # including several genuinely tagged ":free" on their platform (needs
    # # OPENROUTER_API_KEY + OPENROUTER_MODEL — set the model to whatever you want to
    # # route to, e.g. "meta-llama/llama-3.1-8b-instruct:free"; check openrouter.ai/models
    # # for the current free-tier catalog since it changes) ---
    # "openrouter-custom": {
    #     "provider": "openai_compatible", "model_name_env": "OPENROUTER_MODEL",
    #     "base_url": "https://openrouter.ai/api/v1", "tier": "custom", "temperature": 0.1,
    #     "requires_env": ["OPENROUTER_API_KEY", "OPENROUTER_MODEL"],
    #     "good_for": ("trivial", "standard", "foundational", "planning", "review"),
    # },

    # # --- Generic custom slot: point this at literally any OpenAI-compatible
    # # endpoint by setting the three env vars — this is also where "NEON" goes if
    # # you mean a specific self-hosted or third-party endpoint by that name, since
    # # it isn't a publicly known LLM API as of this writing ---
    # "neon-custom": {
    #     "provider": "openai_compatible", "model_name_env": "NEON_MODEL", "base_url_env": "NEON_BASE_URL",
    #     "tier": "custom", "temperature": 0.1,
    #     "requires_env": ["NEON_API_KEY", "NEON_BASE_URL", "NEON_MODEL"],
    #     "good_for": ("standard", "foundational"),
    # },
}

_model_client_cache = {}


def provider_ready(model_key: str) -> bool:
    """Whether a model can actually be called right now — local Ollama models are
    always considered ready; cloud/custom providers need their env vars set."""
    entry = MODEL_REGISTRY.get(model_key)
    if not entry:
        return False
    if entry["provider"] == "ollama":
        return True
    if entry["provider"] == "openai" and ChatOpenAI is None:
        return False
    if entry["provider"] == "anthropic" and ChatAnthropic is None:
        return False
    if entry["provider"] == "openai_compatible" and ChatOpenAI is None:
        return False
    return all(os.environ.get(var) for var in entry.get("requires_env", []))


def get_model_client(model_key: str, json_mode: bool = False):
    """Lazily builds and caches the right LangChain chat client for a model key,
    regardless of which provider actually serves it. Raises if the provider isn't
    configured — callers are expected to check provider_ready() first or catch this
    and fall back, never to let it surface as a raw crash."""
    cache_key = (model_key, json_mode)
    if cache_key in _model_client_cache:
        return _model_client_cache[cache_key]

    entry = MODEL_REGISTRY.get(model_key)
    if not entry:
        raise ValueError(f"Unknown model '{model_key}'")
    if not provider_ready(model_key):
        raise RuntimeError(f"Provider for '{model_key}' is not configured (needs {entry.get('requires_env', [])})")

    provider = entry["provider"]
    temperature = entry.get("temperature", 0.1)

    if provider == "ollama":
        client = ChatOllama(model=model_key, temperature=temperature, format="json" if json_mode else None)
    elif provider == "openai":
        client = ChatOpenAI(model=entry["model_name"], temperature=temperature, api_key=os.environ.get(entry["requires_env"][0]))
    elif provider == "anthropic":
        client = ChatAnthropic(model=entry["model_name"], temperature=temperature, api_key=os.environ.get(entry["requires_env"][0]))
    elif provider == "openai_compatible":
        model_name = entry.get("model_name") or os.environ.get(entry.get("model_name_env", ""))
        base_url = entry.get("base_url") or os.environ.get(entry.get("base_url_env", ""))
        api_key_var = entry["requires_env"][0]
        client = ChatOpenAI(model=model_name, temperature=temperature, base_url=base_url, api_key=os.environ.get(api_key_var))
    else:
        raise ValueError(f"Unsupported provider '{provider}' for '{model_key}'")

    _model_client_cache[cache_key] = client
    return client


def assess_project_scale(file_count: int) -> str:
    """Cheap proxy for 'how big is this project' — drives how aggressively the
    router is willing to reach for a bigger/pricier model."""
    if file_count < 20:
        return "small"
    if file_count < 150:
        return "medium"
    return "large"


def select_model(purpose: str, project_scale: str = "small", exclude: str = None) -> str:
    """This IS the CTO/Development Manager's model-assignment policy: for a given
    kind of work and project scale, walk a preference chain from strongest-fit down
    to the safe local default, skipping anything not configured (provider_ready)
    or explicitly excluded (used for cross-model review — a different model
    checking the work catches different mistakes than the one that wrote it).
    Always resolves to *something* — DEFAULT_FALLBACK_MODEL never fails, since it's
    the one provider (local Ollama) the whole tool has always assumed exists.
    Local/free models are always tried first within a tier; cloud models only get
    reached for if they're actually configured, and only escalate further as
    project_scale grows — no point paying for a premium model on a 5-file project.
    NVIDIA NIM is prioritized ahead of other cloud/local options in every tier per
    explicit preference — set NVIDIA_API_KEY and it becomes the team's default;
    without it, the router silently falls through to the next option, so nothing
    breaks if the key isn't set."""
    chains = {
        "trivial": [
            "nvidia/llama-3.1-8b-instruct",
            "llama3.2:3b", "gemma2:9b", "phi3:14b", "llama3.1:8b", "mistral:7b", "qwen2.5-coder:7b",
        ],
        "standard": [
            "nvidia/llama-3.1-8b-instruct", "nvidia/mixtral-8x7b-instruct",
            "qwen2.5-coder:14b", "mixtral:8x7b", "starcoder2:15b", "codellama:13b",
            "groq-llama-3.3-70b", "gpt-4o-mini", "claude-haiku-4-5", "gemini-2.0-flash",
            "deepseek-chat", "qwen2.5-coder:7b",
        ],
        "foundational": (
            [
                "nvidia/llama-3.1-nemotron-70b-instruct", "nvidia/llama-3.1-70b-instruct",
                "claude-opus-4-1", "claude-sonnet-4-5", "gpt-4.1", "grok-2", "mistral-large-latest",
                "qwen2.5-coder:32b", "deepseek-r1:32b",
                "llama3.1:70b", "deepseek-r1:14b", "deepseek-r1:8b", "qwen2.5-coder:14b",
            ]
            if project_scale == "large"
            else [
                "nvidia/llama-3.1-nemotron-70b-instruct", "nvidia/llama-3.1-70b-instruct",
                "deepseek-r1:14b", "deepseek-r1:8b", "qwen2.5-coder:32b", "qwen2.5-coder:14b", "llama3.1:70b",
            ]
        ),
        "planning": (
            [
                "nvidia/llama-3.1-nemotron-70b-instruct",
                "claude-opus-4-1", "claude-sonnet-4-5", "gpt-4.1", "o3-mini",
                "deepseek-r1:32b", "deepseek-r1:14b", "deepseek-r1:8b",
            ]
            if project_scale != "small"
            else ["nvidia/llama-3.1-nemotron-70b-instruct", "deepseek-r1:8b", "qwen2.5-coder:14b"]
        ),
        "review": [
            "nvidia/llama-3.1-nemotron-70b-instruct", "nvidia/mixtral-8x7b-instruct",
            "deepseek-r1:14b", "deepseek-r1:8b", "gpt-4.1", "claude-sonnet-4-5", "o3-mini",
            "grok-2", "groq-llama-3.3-70b", "qwen2.5-coder:14b",
        ],
    }
    for model_key in chains.get(purpose, [DEFAULT_FALLBACK_MODEL]):
        if model_key == exclude:
            continue
        if provider_ready(model_key):
            return model_key
    return DEFAULT_FALLBACK_MODEL


def purpose_from_role(role_label: str) -> str:
    """Maps a role label back to a router purpose, so the model chosen for a
    sub-task matches the same tier its persona already implies — an Intern and a
    Senior Sub Agent should not be reaching for the same model."""
    if role_label.startswith("Intern"):
        return "trivial"
    if role_label.startswith("Senior"):
        return "foundational"
    return "standard"


task_queue = asyncio.Queue()
app_workspace = os.getcwd()  # Dynamically defaults to your exact current directory context

# Human-in-the-loop state: at most one open question to the user at a time, so a reply
# typed into the chat box unambiguously answers whichever role is currently waiting.
clarification_lock = asyncio.Lock()
active_clarification = {"task_id": None, "future": None}


# Max number of sub-tasks the decomposition step is allowed to split a request into.
# Kept as a soft ceiling so "as many sub-agents as possible" stays practical rather than runaway.
MAX_SUBTASKS = 999999999

# Max characters of an existing file's content injected into a sub-agent's prompt.
# Keeps edits grounded in the real file without blowing up token usage.
MAX_EXISTING_FILE_CONTEXT = 300000

# How many of the most recent teammate summaries get passed to each sub-agent.
MAX_TEAM_CONTEXT_ITEMS = 100

# Every quality gate gets at least 5 passes before the pipeline accepts a best-effort
# result and moves on — code review, QA testing, and syntax repair all recheck
# repeatedly rather than giving up after one shot.
MAX_REVIEW_ROUNDS = 5
MAX_QA_ROUNDS = 10
MAX_SYNTAX_FIX_ROUNDS = 500
TEST_TIMEOUT_SECONDS = 20

# How many parallel workstreams (each with its own Development Manager) the CTO may split
# a request into. Small requests should still get exactly one workstream/manager —
# this is a ceiling, not a target.
MAX_MANAGERS = 50

# Per-task token budget is effectively unlimited (set to 111,000,000,000,000 tokens
# by explicit instruction). Nothing in this pipeline enforces this as a hard cap —
# it exists so every role's prompt can optimize purely for thoroughness, correctness,
# and quality-gate rigor instead of trimming detail to save tokens.
TOKEN_BUDGET_PER_TASK = 999_999_999_999_999_999_999_999_998

# Full org chart with each role's detailed responsibilities. Used to brief the user
# at startup and to keep every persona prompt in the pipeline consistent with the
# same job description.
TEAM_ROSTER = {
    "CTO": (
        "Owns engineering across every project. For each incoming request: scans and "
        "analyzes the existing codebase/dependencies, sets the high-level technical "
        "approach, splits large requests into workstreams, appoints a Development Manager "
        "per workstream, asks the stakeholder clarifying questions when a decision is "
        "genuinely ambiguous, collects and weighs every Development Manager's opinion "
        "before finalizing the plan, and gives the final internal go/no-go before the "
        "IT Manager presents completed work to the stakeholder."
    ),
    "Development Manager": (
        "Owns one workstream within a project, reporting to the CTO. Reviews the "
        "CTO's brief for their workstream and gives an honest opinion — approves it "
        "or raises concrete concerns/risks before work starts. Once the CTO's final "
        "plan is approved, decomposes their workstream into single-responsibility "
        "sub-tasks with clear interfaces and correct dependency order, assigns each "
        "sub-task to the right specialist (Senior/Backend/Frontend Sub Agent or "
        "Intern), and signs off on their workstream's completed deliverables before "
        "reporting back to the CTO."
    ),
    "Senior Developer": (
        "Implements foundational or higher-risk pieces (shared utilities, data "
        "models, auth, core architecture) that other sub-tasks depend on. Writes "
        "production-quality code with explicit error handling, validates all "
        "assumptions, and documents any non-obvious design choice inline."
    ),
    "Backend Sub Agent": (
        "Implements server-side logic, APIs, data access, and business rules. "
        "Validates all inputs, handles errors and edge cases explicitly, never "
        "hardcodes secrets/credentials, and designs interfaces the frontend or other "
        "services can rely on without surprises."
    ),
    "Frontend Sub Agent": (
        "Implements UI components, pages, and client-side logic. Writes accessible, "
        "responsive markup, follows the project's existing component/state-management "
        "conventions, and handles loading/empty/error states, not just the happy path."
    ),
    "Intern Developer": (
        "Handles small, well-scoped, low-risk items (renames, constants, config "
        "values, minor copy/comment edits, boilerplate). Stays strictly within the "
        "assignment and never improvises beyond it — every submission still goes "
        "through the same Staff Engineer review and QA gate as senior work."
    ),
    "Staff Engineer (Code Reviewer)": (
        "Reviews every file before it ships: checks it actually fulfills its "
        "required interface, matches project conventions, and is free of obvious "
        "bugs, missing imports, or security issues (injection, secrets, unsafe "
        "input handling). Approves solid work without nitpicking style; sends "
        "specific, actionable feedback back to the author when it isn't ready."
    ),
    "Test Engineer (QA)": (
        "Writes real tests for testable code (happy path plus at least one edge "
        "case) and actually executes them. Sends failures back to the author with "
        "the exact failure output for one fix-and-retest round; if still failing, "
        "ships flagged as an open item rather than silently, so nothing gets lost."
    ),
    "Finance Manager": (
        "Tracks token spend per task and for the whole session and reports it "
        "plainly after every task, so cost is always visible even though it is not "
        "used to limit what the team does."
    ),
    "IT Manager": (
        "Brings the CTO's sign-off summary directly to the actual human stakeholder "
        "at the end of a task, asks whether it's acceptable, and either confirms "
        "approval or turns any requested changes into a new follow-up task — this is "
        "a real conversation that blocks on the stakeholder's reply, not a rubber "
        "stamp."
    ),
}

# Live mid-task steering: user messages prefixed with "!" are corrections routed to
# whatever project is currently active, instead of being queued as a brand-new task.
# Keeps the team responsive to "no, do X instead" without waiting for the whole run
# to finish first.
live_corrections = {}  # target_dir -> list[str]


def estimate_tokens(prompt_text: str, response) -> int:
    """
    Best-effort token accounting for a single LLM call.

    Prefers the model's own usage_metadata (when the Ollama/LangChain backend reports it),
    and falls back to a ~4-chars-per-token heuristic over prompt + completion text otherwise.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage:
        total = usage.get("total_tokens")
        if total:
            return int(total)
        input_t = usage.get("input_tokens", 0) or 0
        output_t = usage.get("output_tokens", 0) or 0
        if input_t or output_t:
            return int(input_t + output_t)

    response_text = getattr(response, "content", "") or ""
    combined_len = len(prompt_text) + len(response_text)
    return max(1, combined_len // 4)


# --- INTELLIGENCE HELPERS -----------------------------------------------------
# These are what turn the pipeline from "blind JSON split + overwrite a file" into
# something closer to a real architect + engineering team: workspace awareness,
# resilient parsing, dependency ordering, shared context, and basic self-QA.

def clean_json_text(text: str) -> str:
    """Strips markdown code fences models love to wrap JSON in, even when told not to."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


def safe_json_loads(text: str, default=None):
    """Parses LLM JSON output defensively — models occasionally emit single-quoted
    dict-style syntax instead of strict JSON, so we retry with a quote fix before giving up."""
    cleaned = clean_json_text(text or "")
    try:
        return json.loads(cleaned)
    except Exception:
        try:
            fixed = re.sub(r"(?<!\\)'", '"', cleaned)
            return json.loads(fixed)
        except Exception:
            return default


def workspace_fingerprint(target_dir: str):
    """Cheap signature of a workspace (file count + newest mtime) used to detect
    whether anything changed since the last scan. Lets the team reuse its cached
    file map instead of re-walking and re-reasoning about the whole project on
    every single task — real token/time savings on large or unchanged projects."""
    ignore_dirs = {".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build", MEMORY_DIRNAME}
    count, newest = 0, 0.0
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
        for fname in files:
            count += 1
            try:
                mtime = os.path.getmtime(os.path.join(root, fname))
                newest = max(newest, mtime)
            except OSError:
                continue
    return [count, round(newest, 2)]


def get_or_refresh_workspace_summary(target_dir: str, memory: dict) -> str:
    """Reuses the cached workspace file map from project memory when nothing has
    changed on disk; only pays for a full rescan when the fingerprint moved."""
    fp = workspace_fingerprint(target_dir)
    if memory.get("workspace_fingerprint") == fp and memory.get("workspace_summary_cache"):
        return memory["workspace_summary_cache"], False
    summary = scan_workspace_context(target_dir)
    memory["workspace_fingerprint"] = fp
    memory["workspace_summary_cache"] = summary
    return summary, True


_FOLDER_REQUEST_PATTERNS = [
    re.compile(r"(?:in|inside|under|within)\s+(?:the\s+|a\s+|an\s+)?['\"]?([\w][\w\-./ ]{0,60}?)['\"]?\s+folder", re.IGNORECASE),
    re.compile(r"folder\s+(?:called|named)\s+['\"]?([\w][\w\-./ ]{0,60}?)['\"]?(?:\s|$|,|\.)", re.IGNORECASE),
    re.compile(r"(?:in|inside|under|within)\s+(?:the\s+)?['\"]?([\w][\w\-./ ]{0,60}?)['\"]?\s+directory", re.IGNORECASE),
]


def extract_requested_folder(prompt: str):
    """Detects an explicit folder the stakeholder named in their request, so the
    team never flattens new files onto the project root when the human clearly
    asked for a specific directory."""
    for pattern in _FOLDER_REQUEST_PATTERNS:
        m = pattern.search(prompt)
        if m:
            folder = m.group(1).strip().strip("/\\").replace(" ", "_")
            if folder and folder.lower() not in {"root", "project", "the project"}:
                return folder
    return None


def enforce_requested_folder(sub_task_objs, requested_folder: str):
    """Auto-corrects any sub-task file path that ignored an explicitly-requested
    folder, instead of just hoping the planner respected it."""
    if not requested_folder:
        return sub_task_objs, []
    corrections = []
    norm_folder = requested_folder.replace("\\", "/").strip("/")
    for st in sub_task_objs:
        f = (st.get("file") or "").replace("\\", "/").strip("/")
        if not f:
            continue
        if not (f == norm_folder or f.startswith(norm_folder + "/")):
            fixed = f"{norm_folder}/{f}"
            corrections.append(f"{f} -> {fixed}")
            st["file"] = fixed
    return sub_task_objs, corrections


def n8n_ready() -> bool:
    """Whether an n8n webhook is actually configured to receive triggers."""
    return bool(os.environ.get("N8N_WEBHOOK_URL"))


def trigger_n8n_workflow(sub_task_name: str, payload: dict, timeout: int = 15):
    """Fires a real n8n workflow via its webhook, for work that's automation (send
    a notification, kick off a pipeline, sync a CRM, etc.) rather than a file to
    write. n8n isn't an LLM, so this is a distinct sub-task 'type' from file/folder
    — the Development Manager routes automation-shaped requests here instead of
    asking a coding sub-agent to fake it. Uses only the stdlib (urllib) so this
    works with zero extra dependencies. Set N8N_WEBHOOK_URL (and optionally
    N8N_API_KEY, sent as a Bearer token) to enable it; without a URL configured,
    returns a clear 'not configured' result rather than raising."""
    import urllib.request
    import urllib.error

    webhook_url = os.environ.get("N8N_WEBHOOK_URL")
    if not webhook_url:
        return False, "N8N_WEBHOOK_URL is not set — nothing to trigger."

    body = json.dumps({"sub_task": sub_task_name, **payload}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("N8N_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        req = urllib.request.Request(webhook_url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            response_body = resp.read(1000).decode("utf-8", errors="replace")
            if 200 <= status < 300:
                return True, f"n8n responded {status}: {response_body[:200]}"
            return False, f"n8n responded {status}: {response_body[:200]}"
    except urllib.error.URLError as e:
        return False, f"Could not reach n8n webhook: {e}"
    except Exception as e:
        return False, f"n8n trigger failed: {e}"


def extract_code_conventions(target_dir: str, ext: str, max_samples: int = 2) -> str:
    """Samples a couple of existing files of the same type to infer house style
    (indentation, quote style) so new code matches the codebase instead of the
    model's own default habits."""
    if not ext:
        return ""
    samples = []
    ignore_dirs = {".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build", MEMORY_DIRNAME}
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
        for fname in files:
            if fname.endswith(ext):
                try:
                    with open(os.path.join(root, fname), "r", encoding="utf-8") as f:
                        samples.append(f.read(1500))
                except Exception:
                    continue
            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break
    if not samples:
        return ""
    blob = "\n".join(samples)
    indent = "tabs" if "\t" in blob.split("\n", 1)[0:1] or blob.count("\n\t") > blob.count("\n    ") else "spaces (4)"
    quote = "single quotes" if blob.count("'") > blob.count('"') else "double quotes"
    semi = "with semicolons" if ext in (".js", ".ts", ".jsx", ".tsx") and blob.count(";") > 5 else ""
    return f"House style observed in this codebase: {indent} for indentation, prefers {quote}{(', ' + semi) if semi else ''}."


def scan_workspace_context(target_dir: str, max_files: int = 40) -> str:
    """Builds a lightweight snapshot of the existing codebase so the CTO planner and
    sub-agents make decisions that fit the real project instead of working blind."""
    ignore_dirs = {".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build"}
    code_ext = {".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md", ".txt", ".yaml", ".yml", ".toml", ".html", ".css"}
    found = []
    count = 0
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
        for fname in files:
            if count >= max_files:
                break
            if os.path.splitext(fname)[1] not in code_ext:
                continue
            rel = os.path.relpath(os.path.join(root, fname), target_dir)
            found.append(f"- {rel}")
            count += 1
        if count >= max_files:
            break
    if not found:
        return "Workspace is currently empty (new project)."
    return "Existing files in workspace:\n" + "\n".join(found)


def extract_signatures(code: str, max_len: int = 300) -> str:
    """Compresses a written file down to its public surface (function/class/export lines)
    so it can be handed to *other* sub-agents as cheap context instead of the full file."""
    sigs = re.findall(
        r'^\s*(?:def |class |async def |export |function |const \w+\s*=\s*\(|module\.exports)[^\n]{0,120}',
        code,
        flags=re.MULTILINE,
    )
    if not sigs:
        return code[:max_len].replace("\n", " ").strip()
    return "; ".join(s.strip() for s in sigs[:8])


def extract_file_and_code(content: str):
    """Robustly pulls the target filepath and code body out of a sub-agent's response,
    tolerating formatting drift (extra whitespace, language tags on fences, etc.) instead
    of the previous brittle fixed-position string splitting."""
    fp_match = re.search(r"FILEPATH:\s*(.+)", content)
    if not fp_match:
        return None, None
    filepath = fp_match.group(1).strip().strip("`").strip()

    code_fence = re.search(r"```[a-zA-Z0-9]*\r?\n(.*?)```", content, flags=re.DOTALL)
    if code_fence:
        code = code_fence.group(1).strip()
    else:
        after_code = content.split("CODE:", 1)
        code = after_code[1].strip() if len(after_code) > 1 else ""

    return (filepath, code) if code else (filepath, None)


MEMORY_DIRNAME = ".agent_hub"
MEMORY_FILENAME = "memory.json"
_memory_locks = {}


def _get_memory_lock(target_dir: str) -> asyncio.Lock:
    """One lock per workspace so two tasks hitting the same project's memory file
    concurrently can't clobber each other's writes."""
    if target_dir not in _memory_locks:
        _memory_locks[target_dir] = asyncio.Lock()
    return _memory_locks[target_dir]


def _memory_path(target_dir: str) -> str:
    return os.path.join(target_dir, MEMORY_DIRNAME, MEMORY_FILENAME)


def load_project_memory(target_dir: str) -> dict:
    """Loads this project's persistent engineering memory: architecture decisions made
    in past tasks, the known interface of every file the team has touched, and a
    changelog of prior work. This is what lets the CTO reason with real institutional
    context instead of re-discovering the project from scratch on every prompt."""
    path = _memory_path(target_dir)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("architecture_notes", [])
            data.setdefault("file_registry", {})
            data.setdefault("changelog", [])
            data.setdefault("open_items", [])
            return data
        except Exception:
            pass
    return {"architecture_notes": [], "file_registry": {}, "changelog": [], "open_items": []}


def save_project_memory(target_dir: str, memory: dict) -> None:
    try:
        os.makedirs(os.path.join(target_dir, MEMORY_DIRNAME), exist_ok=True)
        with open(_memory_path(target_dir), "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
    except Exception:
        pass  # Memory persistence is best-effort; never crash a task over it


def detect_tech_stack(target_dir: str) -> str:
    """Reads real dependency manifests so the planner knows the actual stack instead
    of guessing from file extensions alone."""
    manifests = ["package.json", "requirements.txt", "pyproject.toml", "go.mod", "Cargo.toml", "Gemfile"]
    snippets = []
    for name in manifests:
        path = os.path.join(target_dir, name)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()[:1000]
                snippets.append(f"--- {name} ---\n{content}")
            except Exception:
                continue
    if not snippets:
        return "No dependency manifest found (fresh project or unrecognized stack)."
    return "\n".join(snippets)


def summarize_memory_for_planner(memory: dict, max_files: int = 25, max_changelog: int = 6) -> str:
    """Compresses project memory into planner-ready context: recent architecture
    decisions, known file interfaces, and a short engineering changelog."""
    parts = []
    if memory.get("architecture_notes"):
        parts.append(
            "Architecture decisions already on record (respect these unless the new "
            "request explicitly changes them):\n"
            + "\n".join(f"- {n}" for n in memory["architecture_notes"][-10:])
        )
    if memory.get("file_registry"):
        items = list(memory["file_registry"].items())[-max_files:]
        parts.append(
            "Known file interfaces from prior work (institutional knowledge — reuse, "
            "don't recreate):\n"
            + "\n".join(f"- {fp}: {info.get('interface', '')}" for fp, info in items)
        )
    if memory.get("changelog"):
        recent = memory["changelog"][-max_changelog:]
        parts.append(
            "Recent engineering history:\n"
            + "\n".join(
                f"- [{c.get('task_id')}] {c.get('prompt', '')[:80]} -> {c.get('approach', '')[:100]}"
                for c in recent
            )
        )
    if memory.get("open_items"):
        parts.append(
            "Outstanding items flagged by previous QA/CTO sign-off (address if this "
            "request touches them):\n"
            + "\n".join(f"- {item}" for item in memory["open_items"][-10:])
        )
    return "\n\n".join(parts) if parts else "No prior engineering history for this project yet — this is the first task."


def is_python_testable(code: str) -> bool:
    """Quick heuristic: only bother generating/running tests for files that actually
    define functions or classes — no point testing a pure config/constants file."""
    return bool(re.search(r'^\s*(?:def |class |async def )', code, flags=re.MULTILINE))


def run_pytest(test_path: str, cwd: str, timeout: int = TEST_TIMEOUT_SECONDS):
    """Actually executes the QA agent's generated tests with pytest, exactly like a
    real Test Engineer would in CI. Runs in-process (blocking) — call via
    loop.run_in_executor so it never stalls the asyncio event loop. Failures to even
    launch pytest (not installed, etc.) are treated as inconclusive, not a hard fail,
    since we can't assume the target environment has every dependency."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-q", "--no-header"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, output[-1500:]
    except subprocess.TimeoutExpired:
        return False, f"Test run timed out after {timeout}s (possible hang or infinite loop)."
    except Exception as e:
        return None, f"Could not execute pytest: {e}"  # None = inconclusive, not a failure





# --- MERMAID GRAPH GENERATION ---
# Both of these are pure/deterministic — no LLM call needed, since the org chart is
# fixed and the dependency graph is just a direct rendering of data the Development
# Manager already produced. Written to real .md files on disk (real Mermaid
# fenced blocks — renders in GitHub, VS Code, Obsidian, the Mermaid Live Editor,
# etc.), not just printed to the console log.
def generate_org_chart_mermaid() -> str:
    """Static Mermaid flowchart of the fixed team hierarchy."""
    lines = [
        "```mermaid",
        "flowchart TD",
        '    CTO["CTO"]',
        '    CTO --> DM["Development Manager"]',
        '    CTO --> ITM["IT Manager"]',
        '    CTO --> FIN["Finance Manager"]',
        '    DM --> SFE["Senior Frontend Sub Agent"]',
        '    DM --> FE["Frontend Sub Agent"]',
        '    DM --> SBE["Senior Backend Sub Agent"]',
        '    DM --> BE["Backend Sub Agent"]',
        '    DM --> SSE["Senior Software Sub Agent"]',
        '    DM --> SW["Software Sub Agent"]',
        '    DM --> INT["Intern Developer"]',
        '    DM --> N8N["n8n Workflow (automation sub-tasks)"]',
        '    FE --> SE["Staff Engineer (Code Review)"]',
        '    BE --> SE',
        '    SW --> SE',
        '    INT --> SE',
        '    SE --> QA["Test Engineer (QA)"]',
        '    QA --> ITM',
        "```",
    ]
    return "\n".join(lines)


def generate_dependency_graph_mermaid(sub_task_objs: list) -> str:
    """Renders the current task's sub-task list and their depends_on relationships
    as a Mermaid flowchart — a real, at-a-glance execution-order diagram instead of
    reading depends_on arrays out of raw JSON."""
    lines = ["```mermaid", "flowchart TD"]
    for st in sub_task_objs:
        sid = st.get("id")
        if sid is None:
            continue
        name = str(st.get("name") or f"Sub-task {sid}").replace('"', "'")[:60]
        stype = st.get("type", "file")
        if stype == "folder":
            lines.append(f'    T{sid}(["📁 {name}"])')
        elif stype == "workflow":
            lines.append(f'    T{sid}{{{{"⚙ {name}"}}}}')
        else:
            lines.append(f'    T{sid}["📄 {name}"]')
    has_edges = False
    for st in sub_task_objs:
        sid = st.get("id")
        for dep in st.get("depends_on") or []:
            lines.append(f"    T{dep} --> T{sid}")
            has_edges = True
    if not has_edges:
        lines.append("    %% no dependencies between sub-tasks — all independent")
    lines.append("```")
    return "\n".join(lines)


def write_graph_file(target_dir: str, filename: str, title: str, mermaid_block: str) -> str:
    """Writes a Mermaid graph to a real .md file under .agent_hub/graphs/ and
    returns its path relative to the project root, for logging."""
    graphs_dir = os.path.join(target_dir, MEMORY_DIRNAME, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    full_path = os.path.join(graphs_dir, filename)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{mermaid_block}\n")
    return os.path.relpath(full_path, target_dir)


LANGUAGE_BY_EXT = {
    ".py": "Python", ".ipynb": "Python",
    ".js": "JavaScript", ".jsx": "React (JavaScript)", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "React (TypeScript)",
    ".vue": "Vue.js", ".svelte": "Svelte",
    ".html": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".cs": "C# / .NET", ".csx": "C# / .NET", ".vb": "VB.NET",
    ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin",
    ".swift": "Swift", ".m": "Objective-C",
    ".rb": "Ruby", ".php": "PHP", ".go": "Go", ".rs": "Rust",
    ".sql": "SQL", ".sh": "Shell", ".ps1": "PowerShell",
    ".dart": "Dart/Flutter", ".c": "C", ".cpp": "C++", ".h": "C/C++ header",
}
FRONTEND_EXT = {".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte", ".html", ".css", ".scss", ".mjs", ".cjs"}
BACKEND_EXT = {".py", ".go", ".rb", ".java", ".rs", ".php", ".sql", ".cs", ".csx", ".vb", ".kt", ".kts", ".swift", ".dart"}
INTERN_KEYWORDS = ("typo", "rename", "simple", "minor", "small tweak", "constant", "boilerplate", "config value", "readme", "comment")


def select_engineer_persona(filepath: str, sub_task: dict = None):
    """Routes each sub-task to the right specialist based on the file's language AND
    the task's apparent complexity — the way a real team hands UI work to frontend
    devs, foundational/complex pieces to a senior engineer, and small housekeeping
    items to an intern (who still goes through the same code review and QA gates
    as everyone else). Language-agnostic by design: works across Python, .NET/C#,
    React/TypeScript, vanilla JS/jQuery, Ruby, Kotlin, Swift, React Native,
    Node/Next.js, and anything else with a recognizable extension. Returns
    (role_label, persona_sentence, language_name)."""
    ext = os.path.splitext(filepath)[1].lower()
    language = LANGUAGE_BY_EXT.get(ext, "the appropriate language for this file")

    sub_task = sub_task or {}
    name_l = (sub_task.get("name") or "").lower()
    is_trivial = any(k in name_l for k in INTERN_KEYWORDS) and not sub_task.get("depends_on")
    is_foundational = bool(sub_task.get("depends_on")) or any(
        k in name_l for k in ("core", "architecture", "schema", "model", "auth", "security", "database", "api contract")
    )

    if is_trivial:
        return "Intern Developer", (
            f"You are a supervised Intern Developer writing {language} code. Your job:\n"
            "1) SCOPE: Do exactly the assigned change and nothing else — no refactors, "
            "no 'while I'm here' improvements, no new dependencies. If you think the "
            "task needs more than what's described, do the minimal safe version and "
            "leave the rest alone.\n"
            "2) SAFETY NET: Everything you submit goes through a Staff Engineer review "
            "and automated QA before it ships — that does not excuse sloppy work, but "
            "it means you should not guess wildly on ambiguous points; make the most "
            "conservative reasonable choice and keep the diff small.\n"
            "3) STYLE: Match the existing file's formatting, naming, and structure "
            "exactly. Do not introduce a different code style than what's already there.\n"
            "4) CORRECTNESS: Handle the obvious edge cases (empty input, null/None, "
            "off-by-one) even for small changes — 'small' does not mean 'sloppy'.\n"
            "5) NEVER fabricate APIs, imports, or file paths that weren't given to you "
            "or don't exist in the codebase context you were shown."
        ), language

    if ext in FRONTEND_EXT:
        seniority = "Senior Frontend Sub Agent" if is_foundational else "Frontend Sub Agent"
        return seniority, (
            f"You are a {seniority} writing {language}. Your responsibilities:\n"
            "1) UI CORRECTNESS: Implement exactly the described component/behaviour. "
            "Structure markup semantically (correct elements/roles), not div-soup.\n"
            "2) ACCESSIBILITY: Use appropriate ARIA attributes, keyboard navigability, "
            "and labels/alt text wherever a real user would need them — don't skip "
            "this because it wasn't explicitly requested.\n"
            "3) STATE & EDGE CASES: Handle loading, empty, and error states explicitly "
            "for anything that fetches or depends on async data. Handle empty arrays, "
            "null props, and rapid repeated user interaction (e.g. double-submit).\n"
            "4) CONSISTENCY: Match the project's existing component patterns, styling "
            "approach (CSS modules/Tailwind/styled-components/plain CSS — whichever "
            "this project actually uses), and naming conventions. Never introduce a "
            "second styling system alongside an existing one.\n"
            "5) PERFORMANCE: Avoid unnecessary re-renders, unbounded state updates, or "
            "expensive computations in render paths; memoize where it genuinely matters.\n"
            "6) NO SECRETS IN CLIENT CODE: Never hardcode API keys, tokens, or secrets "
            "into frontend code — reference environment/config as the project already does.\n"
            "7) INTERFACE CONTRACT: Your exports/props/events must match the required "
            "interface exactly so sibling sub-tasks (API calls, parent components) work "
            "against it without needing to be told anything you didn't declare."
        ), language

    if ext in BACKEND_EXT:
        seniority = "Senior Backend Sub Agent" if is_foundational else "Backend Sub Agent"
        return seniority, (
            f"You are a {seniority} writing {language}. Your responsibilities:\n"
            "1) INPUT VALIDATION: Validate and sanitize every external input (request "
            "bodies, query params, file contents, env vars) — never trust caller data.\n"
            "2) ERROR HANDLING: Handle failure paths explicitly (not-found, invalid "
            "state, downstream failures, timeouts). Fail with clear, specific errors — "
            "never swallow exceptions silently or return generic 500s where a specific "
            "error would help the caller.\n"
            "3) SECURITY: Guard against injection (SQL/command/path), never hardcode "
            "secrets or credentials, use parameterized queries/prepared statements, "
            "and apply the principle of least privilege in any access logic you write.\n"
            "4) DATA INTEGRITY: Think about concurrent access, idempotency for "
            "retryable operations, and transactional boundaries where the language/"
            "framework in use provides them.\n"
            "5) INTERFACE CONTRACT: Your function signatures, return shapes, and "
            "error contracts must exactly match the required interface so other "
            "sub-tasks (frontend calls, other services) can rely on it without needing "
            "anything undocumented.\n"
            "6) CONSISTENCY: Match the project's existing architecture layer "
            "(controller/service/repository, or whatever pattern is already present), "
            "its naming conventions, and its existing error/response format.\n"
            "7) PERFORMANCE: Avoid N+1 queries, unbounded loops over external calls, "
            "and unnecessary blocking I/O where the language offers async alternatives "
            "already used elsewhere in the project."
        ), language

    seniority = "Senior Software Sub Agent" if is_foundational else "Software Sub Agent"
    return seniority, (
        f"You are a {seniority} writing {language}. Your responsibilities:\n"
        "1) CORRECTNESS FIRST: Solve exactly what the assignment describes; handle "
        "obvious edge cases (empty/null input, boundary values, malformed data) even "
        "when not explicitly listed.\n"
        "2) IDIOMATIC CODE: Write code that looks like it was written by someone fluent "
        "in this specific language/ecosystem — use its standard library and common "
        "idioms rather than translating patterns from a different language.\n"
        "3) CONSISTENCY: Match this project's existing conventions (naming, file "
        "organization, error handling style) exactly — don't introduce a new pattern "
        "when an established one already exists in this codebase.\n"
        "4) DOCUMENTATION: Add concise comments/docstrings for any non-obvious logic "
        "or public interface, matching whatever documentation convention this "
        "language/project already uses.\n"
        "5) INTERFACE CONTRACT: Whatever you export or expose must match the required "
        "interface exactly, since sibling sub-tasks will rely on it without further "
        "clarification from you.\n"
        "6) NO SPECULATION: Do not invent APIs, files, or libraries that weren't given "
        "to you or don't already exist in this project's dependencies."
    ), language


def topo_sort_subtasks(sub_tasks):
    """Reorders sub-tasks so anything listed in depends_on always executes before the
    sub-task that needs it — a safety net in case the planner's own ordering slips."""
    id_map = {st.get("id"): st for st in sub_tasks if st.get("id") is not None}
    visited, ordered = set(), []

    def visit(st, stack):
        sid = st.get("id")
        if sid in visited or sid in stack:
            return
        stack = stack | {sid}
        for dep_id in st.get("depends_on") or []:
            dep = id_map.get(dep_id)
            if dep:
                visit(dep, stack)
        visited.add(sid)
        ordered.append(st)

    for st in sub_tasks:
        visit(st, set())

    unindexed = [st for st in sub_tasks if st.get("id") is None]
    return (ordered + unindexed) if ordered else sub_tasks


class TokenTracker:
    """Process-wide running total of tokens consumed across all agents/sub-agents."""

    def __init__(self):
        self.total = 0
        self._lock = asyncio.Lock()

    async def add(self, amount: int) -> int:
        async with self._lock:
            self.total += amount
            return self.total


token_tracker = TokenTracker()


class KeepAwake:
    """
    Prevents the OS from suspending/sleeping while background sub-agents are working.

    Note: locking the screen by itself does NOT pause a running process on Windows,
    macOS, or Linux — asyncio background tasks keep executing regardless. The actual
    risk is the machine going to *sleep* (common on laptops shortly after a lock,
    depending on power settings), which does freeze the process. This class asks the
    OS to stay awake for as long as at least one agent/sub-agent is active, so work
    keeps progressing even during an extended screen lock.
    """

    def __init__(self):
        self._proc = None

    def start(self):
        try:
            if sys.platform.startswith("win"):
                ES_CONTINUOUS = 0x80000000
                ES_SYSTEM_REQUIRED = 0x00000001
                ES_AWAYMODE_REQUIRED = 0x00000040
                ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
                )
            elif sys.platform == "darwin":
                if self._proc is None:
                    self._proc = subprocess.Popen(
                        ["caffeinate", "-dims"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            elif sys.platform.startswith("linux"):
                if self._proc is None:
                    try:
                        self._proc = subprocess.Popen(
                            [
                                "systemd-inhibit",
                                "--what=idle:sleep:handle-lid-switch",
                                "--why=AgentHub background sub-agents running",
                                "--mode=block",
                                "sleep", "infinity",
                            ],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    except FileNotFoundError:
                        self._proc = None  # systemd-inhibit not available; best effort only
        except Exception:
            pass  # Never let sleep-prevention failures crash the agent pipeline

    def stop(self):
        try:
            if sys.platform.startswith("win"):
                ES_CONTINUOUS = 0x80000000
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            elif self._proc is not None:
                self._proc.terminate()
                self._proc = None
        except Exception:
            pass


keep_awake = KeepAwake()


class ActiveAgentCounter:
    """Tracks how many sub-agents are currently working so we only engage/release
    sleep-prevention at the true start/end of activity, not per sub-task."""

    def __init__(self):
        self.count = 0
        self._lock = asyncio.Lock()

    async def increment(self) -> int:
        async with self._lock:
            self.count += 1
            if self.count == 1:
                keep_awake.start()
            return self.count

    async def decrement(self) -> int:
        async with self._lock:
            self.count = max(0, self.count - 1)
            if self.count == 0:
                keep_awake.stop()
            return self.count


active_agents = ActiveAgentCounter()


class BackgroundTask:
    def __init__(self, prompt, target_dir):
        self.id = str(uuid.uuid4())[:8]
        self.prompt = prompt
        self.target_dir = target_dir
        self.status = "Queued"
        self.current_file = "None"
        self.sub_tasks = []
        # sub_task_records[i] = {"name": str, "tokens": int, "status": str}
        self.sub_task_records = []
        self.total_tokens = 0
        self.decomposition_tokens = 0
        # CTO-level planning state
        self.plan_summary = ""
        # Running log of what each completed sub-agent produced, so later sub-agents
        # in the same task are aware of their teammates' work instead of working blind.
        self.shared_context = []
        # Populated from the workspace's persistent memory file at the start of the run;
        # updated in place and flushed to disk once the task completes.
        self.project_memory = None
        # Corrections the stakeholder sent mid-run via the "!" prefix; once picked up
        # they apply to every remaining sub-task in this run, not just the next one.
        self.live_corrections = []


# --- TEXTUAL CUSTOM GRAPHICAL LAYOUT ---
class AgentHubApp(App):
    CSS = """
    Screen {
        background: #1a1a1a;
    }
    .main-container {
        layout: horizontal;
        height: 5fr;
    }
    .left-panel {
        width: 35%;
        border: solid #333333;
        background: #242424;
        padding: 1;
    }
    .right-panel {
        width: 65%;
        background: #1e1e1e;
        padding: 1;
    }
    #agents-status-container {
        height: 1fr;
        scrollbar-gutter: stable;
        overflow-y: scroll;
        margin-bottom: 1;
    }
    #system-log {
        height: 1fr;
        border: solid #333333;
        background: #141414;
    }    
    .workspace-input {
        margin-bottom: 1;
        border: solid #444444;
    }
    .input-box {
        dock: bottom;
        margin-top: 1;
        border: tall #00ff00;
    }
    .status-card {
        background: #2c2c2c;
        border: round #444444;
        margin-bottom: 1;
        padding: 1;
        height: auto;
    }
    .header-label {
        text-style: bold;
        color: #00ffff;
        margin-bottom: 1;
    }
    .dir-select-row {
        height: auto;
        margin-top: 1;
    }
    #total-tokens-label {
        text-style: bold;
        color: #00ff00;
        margin-bottom: 1;
    }
    """
    
    BINDINGS = [("q", "quit", "Quit Hub"), ("c", "clear_logs", "Clear System Monitor")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, id="app-header")
        
        with Container(classes="main-container"):
            # LEFT SIDEBAR: Interactive Workspace Selector Dashboard
            with Vertical(classes="left-panel"):
                yield Label("📁 WORKSPACE TARGET FOCUS", classes="header-label")
                yield Label(f"Active: {app_workspace}", id="path-display")
                yield Label("🔢 TOTAL TOKENS USED: 0", id="total-tokens-label")
                
                # New Input Field for typing direct paths directly
                yield Label("[gray]Type path & press Enter, or select below:[/gray]")
                yield Input(value=app_workspace, placeholder="e.g. D:\Websites or C:\Projects", id="workspace-path-input", classes="workspace-input")
                
                yield DirectoryTree(app_workspace, id="dir-tree")
                
                with Horizontal(classes="dir-select-row"):
                    yield Button("Select Tree Highlight", variant="primary", id="btn-select-dir")
                
            # RIGHT PANEL: Live Agent Matrix Monitoring & Logs
            with Vertical(classes="right-panel"):
                yield Label("🤖 LIVE BACKGROUND AGENTS ENGINE", classes="header-label")
                yield Container(id="agents-status-container")
                yield Label("📜 CONSOLE SYSTEM MONITOR", classes="header-label")
                yield RichLog(highlight=True, markup=True, id="system-log")
                
        yield Input(placeholder="Type prompt / code request here and hit Enter...", classes="input-box", id="chat-input")
        yield Footer()

    def on_mount(self) -> None:
        """Kicks off background parallel processing loops upon UI initialization."""
        self.log_widget = self.query_one("#system-log", RichLog)
        self.log_widget.write("[bold green]🤖 Agent Hub Activated Successfully.[/bold green]")
        self.log_widget.write("[info]Target folder set to current directory context. Ready for input.[/info]")
        self.log_widget.write(
            "[dim]Team on duty: CTO → Development Manager → Senior/Backend/Frontend Sub Agents, "
            "Interns, Staff Engineer (review), Test Engineer (QA), Finance Manager (cost). "
            "Works across Python, .NET/C#, React/TypeScript, vanilla JS/jQuery, Node/Next.js, "
            "Ruby, Kotlin, Swift, React Native, and more.[/dim]"
        )
        self.log_widget.write(
            "[dim]Tip: prefix a message with ! to send a live correction to a running task "
            "(e.g. \"!use snake_case for filenames\"). The Manager/CTO may also ask you "
            "questions mid-task — just reply in this box to continue.[/dim]"
        )
        configured = [m for m in MODEL_REGISTRY if provider_ready(m)]
        self.log_widget.write(
            f"[dim]Models ready: {', '.join(configured)}. Local Ollama models are always on; "
            "set OPENAI_API_KEY / ANTHROPIC_API_KEY / NVIDIA_API_KEY (or NEON_API_KEY + "
            "NEON_BASE_URL + NEON_MODEL for a custom endpoint) as env vars to light up more. "
            "The CTO/Development Manager picks a model per task by project size and role — "
            "review and QA deliberately use a different model than whoever wrote the code.[/dim]"
        )
        n8n_status = "configured" if n8n_ready() else "not configured (set N8N_WEBHOOK_URL to enable)"
        self.log_widget.write(
            f"[dim]Automation: genuinely automation-shaped sub-tasks (integrations, scheduled "
            f"jobs, data syncs) route to n8n instead of generated code — {n8n_status}. Every "
            "task also writes a real Mermaid dependency graph to .agent_hub/graphs/, plus a "
            "one-time team org chart, viewable in GitHub/VS Code/Obsidian/the Mermaid Live "
            "Editor.[/dim]"
        )
        
        # Start background queue consumers — more workers means more tasks (and their
        # full sub-agent teams) can be staffed and running concurrently.
        WORKER_POOL_SIZE = 8
        for i in range(WORKER_POOL_SIZE):
            asyncio.create_task(self.agent_worker(i))

    # --- ACTION HANDLERS ---
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Updates the agent execution target folder dynamically from the Directory Tree."""
        if event.button.id == "btn-select-dir":
            tree = self.query_one("#dir-tree", DirectoryTree)
            if tree.cursor_node and tree.cursor_node.data:
                selected_path = str(tree.cursor_node.data.path)
                self.update_active_workspace(selected_path)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handles submissions from both the workspace path entry and the core prompt console box."""
        if event.input.id == "workspace-path-input":
            target_path = event.value.strip()
            if target_path:
                self.update_active_workspace(target_path)
                
        elif event.input.id == "chat-input":
            user_input = event.value.strip()
            if not user_input:
                return
                
            event.input.value = ""  # Clear input layout immediately

            # If a role is currently waiting on a question, this input is the answer —
            # route it there instead of spawning a brand-new task.
            pending_future = active_clarification.get("future")
            if pending_future is not None and not pending_future.done():
                self.log_widget.write(f"[bold cyan][You][/bold cyan] {user_input}")
                pending_future.set_result(user_input)
                return

            # "!" prefix = a live correction for whatever's running against the current
            # workspace, not a new task. Lets you say "no, use snake_case" mid-run.
            if user_input.startswith("!"):
                correction_text = user_input[1:].strip()
                if correction_text:
                    live_corrections.setdefault(app_workspace, []).append(correction_text)
                    self.log_widget.write(
                        f"[bold magenta][You → Team][/bold magenta] Correction queued for "
                        f"'{app_workspace}': {correction_text}"
                    )
                return
            
            # Instantiate task context parameters
            new_task = BackgroundTask(prompt=user_input, target_dir=app_workspace)
            asyncio.create_task(task_queue.put(new_task))
            
            self.log_widget.write(f"[bold magenta][System][/bold magenta] Dispatched Task ID [yellow]{new_task.id}[/yellow] to background queue pipelines.")
            self.update_dashboard_ui(new_task)

    def update_active_workspace(self, path: str):
        """Validates and updates the workspace path globally, refreshing the visual state tree."""
        global app_workspace
        normalized_path = os.path.normpath(path)
        
        if os.path.isdir(normalized_path):
            app_workspace = normalized_path
            self.query_one("#path-display", Label).update(f"Active: {app_workspace}")
            self.query_one("#workspace-path-input", Input).value = app_workspace
            
            # Dynamically reload tree content pointing to the new root location
            tree = self.query_one("#dir-tree", DirectoryTree)
            tree.path = app_workspace
            
            self.log_widget.write(f"[bold yellow]📂 Workspace focus shifted to: {app_workspace}[/bold yellow]")
        else:
            self.log_widget.write(f"[bold red]❌ Path Error: '{normalized_path}' is not a valid directory folder.[/bold red]")

    def update_global_tokens_label(self, total: int) -> None:
        """Refreshes the always-visible running total of tokens used across every agent."""
        try:
            self.query_one("#total-tokens-label", Label).update(f"🔢 TOTAL TOKENS USED: {total}")
        except Exception:
            pass

    def update_dashboard_ui(self, task: BackgroundTask) -> None:
        """Injects dynamic status nodes straight into the graphical user pane interface."""
        container = self.query_one("#agents-status-container", Container)
        
        try:
            widget = self.query_one(f"#task-{task.id}", Static)
        except:
            widget = Static(id=f"task-{task.id}", classes="status-card")
            container.mount(widget)
            
        status_style = "bold green" if task.status == "Completed" else "bold yellow"

        # Per sub-agent token breakdown
        if task.sub_task_records:
            sub_lines = "\n".join(
                f"   • Sub-agent {idx + 1} [magenta]({rec['tokens']} tok, {rec['status']})[/magenta]: "
                f"[gray]{rec['name'][:45]}[/gray]"
                for idx, rec in enumerate(task.sub_task_records)
            )
        else:
            sub_lines = "   • [gray]No sub-agents dispatched yet...[/gray]"

        widget.update(
            f" [cyan]Agent ID:[/cyan] {task.id} | "
            f"[cyan]Status:[/cyan] [{status_style}]{task.status}[/{status_style}] | "
            f"[cyan]Task Tokens:[/cyan] [green]{task.total_tokens}[/green] | "
            f"[cyan]Targeting File:[/cyan] [green]{task.current_file}[/green]\n"
            f" ➔ [gray]{task.prompt[:60]}...[/gray]\n"
            f"{sub_lines}"
        )

    # --- HUMAN-IN-THE-LOOP CHECKPOINT ---
    async def ask_user(self, task: "BackgroundTask", question: str, role_label: str = "IT Manager") -> str:
        """Posts a question from the given role into the console log and genuinely
        blocks that task until the user replies in the prompt box — a real checkpoint
        with the actual human, not a simulated one. Only one question is open at a
        time; if another task needs to ask something while this one is pending, it
        will wait its turn on clarification_lock rather than talking over it."""
        async with clarification_lock:
            future = asyncio.get_event_loop().create_future()
            active_clarification["task_id"] = task.id
            active_clarification["future"] = future

            self.log_widget.write(
                f"[bold magenta][{role_label} — Task {task.id}][/bold magenta] {question}"
            )
            self.log_widget.write(
                "[dim]↳ Type your reply in the prompt box below to continue this task.[/dim]"
            )
            prior_status = task.status
            task.status = "Awaiting Your Input..."
            self.update_dashboard_ui(task)

            answer = await future

            task.status = prior_status
            self.update_dashboard_ui(task)
            active_clarification["task_id"] = None
            active_clarification["future"] = None
            return answer

    # --- UPPER-HIERARCHY APPROVAL GATE ---
    # --- MULTI-MODEL DISPATCH ---
    async def call_model(self, loop, model_key: str, messages, json_mode: bool = False, role_label: str = ""):
        """Every LLM call in the pipeline goes through here. Resolves model_key to
        whichever provider actually serves it (local Ollama, OpenAI, Anthropic,
        NVIDIA NIM, or a custom OpenAI-compatible endpoint), and — if that provider
        errors out or isn't configured — falls back to the always-available local
        default instead of surfacing a raw error or stalling the task. This is what
        lets different sub-agents genuinely use different models in the same run
        without any of that plumbing becoming visible or distracting."""
        try:
            client = get_model_client(model_key, json_mode=json_mode)
            return await loop.run_in_executor(None, lambda: client.invoke(messages))
        except Exception as e:
            if model_key == DEFAULT_FALLBACK_MODEL:
                raise
            self.log_widget.write(
                f"[dim]ℹ {role_label or 'Team'}: model '{model_key}' unavailable "
                f"({e.__class__.__name__}) — falling back to {DEFAULT_FALLBACK_MODEL}.[/dim]"
            )
            client = get_model_client(DEFAULT_FALLBACK_MODEL, json_mode=json_mode)
            return await loop.run_in_executor(None, lambda: client.invoke(messages))

    async def get_approval(self, task: "BackgroundTask", loop, system_msg: SystemMessage, prompt_text: str, role_label: str, default_approved: bool = True, model_key: str = None) -> dict:
        """Generic role opinion/approval call: any role (CTO, a Development Manager, Tech
        Lead) can be asked to approve or raise concerns on something before work
        proceeds. Always returns a dict with at least 'approved'; falls back to
        approving by default if the response isn't parseable JSON, so a malformed
        reply can never silently deadlock the pipeline."""
        res = await self.call_model(
            loop, model_key or DEFAULT_FALLBACK_MODEL, [system_msg, HumanMessage(content=prompt_text)],
            json_mode=True, role_label=role_label,
        )
        tokens = estimate_tokens(prompt_text, res)
        task.total_tokens += tokens
        running_total = await token_tracker.add(tokens)
        self.update_global_tokens_label(running_total)
        return safe_json_loads(res.content, default={"approved": default_approved, "feedback": "", "opinion": ""})

    # --- BACKGROUND WORKER QUEUE DAEMON ---
    async def agent_worker(self, worker_id):
        while True:
            task: BackgroundTask = await task_queue.get()

            active_count = await active_agents.increment()
            if active_count == 1:
                self.log_widget.write(
                    "[bold cyan]🔒 Sleep prevention engaged[/bold cyan] — sub-agents will "
                    "keep working even if the screen locks."
                )

            loop = asyncio.get_event_loop()

            try:
                # --- PHASE 0: Reconnaissance ---
                # A real architect never plans in a vacuum — scan what already exists,
                # read the real dependency manifests, and pull up the team's institutional
                # memory (past architecture decisions, known interfaces, changelog) before
                # writing a single line of the plan. Reuses the cached file map when
                # nothing on disk has changed, instead of re-walking and re-reasoning
                # about the whole project from scratch every time.
                self.log_widget.write(
                    f"[bold cyan][CTO][/bold cyan] New request for project "
                    f"'{task.target_dir}' — assigning it to the Development Manager and team."
                )
                task.status = "Loading Institutional Context..."
                self.update_dashboard_ui(task)

                async with _get_memory_lock(task.target_dir):
                    task.project_memory = load_project_memory(task.target_dir)

                memory_summary = summarize_memory_for_planner(task.project_memory)
                stack_summary = detect_tech_stack(task.target_dir)
                workspace_summary, was_rescanned = get_or_refresh_workspace_summary(task.target_dir, task.project_memory)
                requested_folder = extract_requested_folder(task.prompt)

                # Project scale drives the CTO's model-assignment policy for this whole
                # task: small/simple work stays on the fast local model, foundational or
                # large-project work escalates to a stronger (possibly cloud) model —
                # whichever is actually configured. Reuses the same fingerprint we already
                # computed for scan caching, so this costs nothing extra.
                project_scale = assess_project_scale(workspace_fingerprint(task.target_dir)[0])
                planning_model = select_model("planning", project_scale)
                self.log_widget.write(
                    f"[gray][Worker {worker_id}][/gray] Project scale: [cyan]{project_scale}[/cyan] — "
                    f"CTO/Development Manager reasoning will use [cyan]{planning_model}[/cyan]; "
                    f"sub-agents will be assigned per-task per the same policy."
                )

                known_files = len(task.project_memory.get("file_registry", {}))
                known_decisions = len(task.project_memory.get("architecture_notes", []))
                known_tasks = len(task.project_memory.get("changelog", []))
                scan_note = "full rescan" if was_rescanned else "reused cached map, no rescan needed"
                self.log_widget.write(
                    f"[gray][Worker {worker_id}][/gray] Institutional memory loaded ({scan_note}): "
                    f"[cyan]{known_files}[/cyan] known files, [cyan]{known_decisions}[/cyan] "
                    f"architecture decisions, [cyan]{known_tasks}[/cyan] prior tasks on record."
                )
                if requested_folder:
                    folder_full_path = os.path.join(task.target_dir, requested_folder)
                    os.makedirs(folder_full_path, exist_ok=True)
                    self.log_widget.write(
                        f"[gray][Worker {worker_id}][/gray] Detected explicit folder request: "
                        f"[cyan]{requested_folder}/[/cyan] — created it on disk immediately and "
                        f"will enforce it on all new files (guaranteed regardless of downstream planning)."
                    )

                # --- PHASE 0.5: CTO analyzes the request, scopes it into workstream(s), and
                # consults every Development Manager for an opinion before committing engineering
                # time — nothing proceeds on guesswork from a single voice.
                task.status = "CTO Reviewing Request..."
                self.update_dashboard_ui(task)

                cto_intake_system = SystemMessage(content=(
                    f"You are the CTO. {TEAM_ROSTER['CTO']} Analyze the incoming request "
                    "using the real codebase scan, dependency manifests, and institutional "
                    "memory you're given. Decide honestly whether this genuinely needs more "
                    "than one parallel workstream (independent pieces of work with separate "
                    "concerns) — most requests need exactly ONE; only split when there are "
                    "clearly distinct, independently-deliverable pieces. Return STRICT JSON "
                    "only: {\"analysis\": \"2-3 sentence technical read of the request\", "
                    '"workstreams": ["short description of workstream 1", "..."], '
                    '"clarifying_questions": ["at most 3 SHORT questions, ONLY for genuine '
                    'work-blocking ambiguity, else omit/empty"]}'
                ))
                cto_intake_prompt = (
                    f"=== Dependency manifests ===\n{stack_summary}\n\n"
                    f"=== Workspace file listing ===\n{workspace_summary}\n\n"
                    f"=== Institutional memory ===\n{memory_summary}\n\n"
                    f"=== Stakeholder request ===\n{task.prompt}\n\n"
                    f"Use at most {MAX_MANAGERS} workstreams. Return JSON only."
                )
                cto_plan = await self.get_approval(task, loop, cto_intake_system, cto_intake_prompt, "CTO", model_key=planning_model)

                cto_questions = [q for q in (cto_plan.get("clarifying_questions") or []) if q]
                if cto_questions:
                    numbered = "\n".join(f"{idx + 1}. {q}" for idx, q in enumerate(cto_questions[:3]))
                    cto_answer = await self.ask_user(
                        task,
                        f"Before scoping \"{task.prompt[:80]}\", I need a bit more detail:\n{numbered}",
                        role_label="CTO",
                    )
                    cto_intake_prompt += (
                        f"\n\n=== Stakeholder's answer ===\n{cto_answer}\nIncorporate this. "
                        "Do not ask further questions this time."
                    )
                    cto_plan = await self.get_approval(task, loop, cto_intake_system, cto_intake_prompt, "CTO", model_key=planning_model)

                workstreams = [w for w in (cto_plan.get("workstreams") or []) if w][:MAX_MANAGERS] or [task.prompt]
                if cto_plan.get("analysis"):
                    self.log_widget.write(f"[bold cyan][CTO][/bold cyan] {cto_plan['analysis']}")
                self.log_widget.write(
                    f"[gray][Worker {worker_id}][/gray] Scoped into [cyan]{len(workstreams)}[/cyan] "
                    f"workstream(s); consulting the Development Manager(s) for their opinion..."
                )

                # --- Manager Discussion: every Development Manager reviews their workstream and
                # gives an honest opinion — approval or concrete, named concerns — before the
                # CTO commits the team to it.
                manager_notes = []
                for w_idx, ws_desc in enumerate(workstreams, start=1):
                    manager_label = f"Development Manager {w_idx}" if len(workstreams) > 1 else "Development Manager"
                    opinion_system = SystemMessage(content=(
                        f"You are {manager_label}. {TEAM_ROSTER['Development Manager']} You are "
                        "reviewing a workstream brief from the CTO before your team commits to "
                        "it. Give an honest opinion — approve it, or raise concrete risks/"
                        "concerns (missing info, conflicts with existing code, unrealistic "
                        "scope). Return STRICT JSON only: {\"approved\": true or false, "
                        "\"opinion\": \"1-2 sentences, specific\"}"
                    ))
                    opinion_prompt = (
                        f"Your workstream brief from the CTO: {ws_desc}\n"
                        f"Full original stakeholder request for context: {task.prompt}\n"
                        f"Relevant institutional memory: {memory_summary}\n\nReturn JSON only."
                    )
                    verdict = await self.get_approval(task, loop, opinion_system, opinion_prompt, manager_label, model_key=planning_model)
                    opinion_text = verdict.get("opinion") or verdict.get("feedback") or ""
                    approved = verdict.get("approved", True)
                    manager_notes.append(
                        f"{manager_label} on '{ws_desc[:70]}': "
                        f"{'APPROVED' if approved else 'CONCERN'} — {opinion_text}"
                    )
                    style = "green" if approved else "yellow"
                    mark = "✓ Approved" if approved else "⚠ Raised a concern"
                    self.log_widget.write(f"[{style}][{manager_label}][/{style}] {mark}: {opinion_text}")

                # --- CTO synthesizes every Manager's opinion into a final, approved approach.
                cto_final_system = SystemMessage(content=(
                    "You are the CTO giving final go-ahead after hearing every Project "
                    "Manager's opinion. Weigh any concerns raised — address them in the final "
                    "approach rather than ignoring them — and commit the team to a clear "
                    "direction. Return STRICT JSON only: {\"final_approach\": \"2-4 sentences, "
                    "incorporating any manager feedback that mattered\"}"
                ))
                cto_final_prompt = (
                    f"Original request: {task.prompt}\n"
                    f"Workstreams considered: {workstreams}\n"
                    "Manager opinions:\n" + "\n".join(manager_notes) + "\n\nReturn JSON only."
                )
                cto_final = await self.get_approval(task, loop, cto_final_system, cto_final_prompt, "CTO", model_key=planning_model)
                cto_final_approach = cto_final.get("final_approach") or cto_plan.get("analysis", "")
                self.log_widget.write(f"[bold green][CTO][/bold green] ✅ Final approach approved: {cto_final_approach}")
                self.log_widget.write(
                    "[bold cyan][CTO][/bold cyan] Handing off to the Development Manager to build "
                    "the detailed execution plan."
                )

                # --- PHASE 1: Development Manager architecture + decomposition (reporting to CTO) ---
                task.status = "Architecting Solution..."
                self.update_dashboard_ui(task)

                planner_system = SystemMessage(content=(
                    "You are the Development Manager for this specific project, reporting to the "
                    "CTO. You act as the technical architect for your project. Your full "
                    "responsibilities:\n"
                    "1) SCOPE: Solve exactly what was asked — no speculative extra features, "
                    "no unrelated refactors, no gold-plating. If something adjacent is broken "
                    "and blocks the request, fix only what's necessary and note it as a "
                    "separate architecture_note rather than silently expanding scope.\n"
                    "2) RESPECT THE EXISTING PROJECT: If the workspace already contains a real "
                    "project, you extend/modify the specific relevant files in place. You never "
                    "restructure, rename, or recreate an existing project from scratch. You "
                    "reuse existing files/interfaces/patterns from institutional memory and the "
                    "workspace listing instead of duplicating or reinventing them.\n"
                    "3) DECOMPOSITION: Break work into the smallest sensible, single-"
                    "responsibility sub-tasks so your team (senior/backend/frontend developers, "
                    "interns, QA) can implement each independently with minimal context. Each "
                    "sub-task needs exactly one target file and one clear interface — never "
                    "bundle unrelated changes into one sub-task.\n"
                    "4) DEPENDENCY ORDER: Sequence sub-tasks so shared utilities, config, data "
                    "models, and type/interface definitions are built before anything that "
                    "consumes them. Populate depends_on accurately — this is used to enforce "
                    "correct build order, not just for documentation.\n"
                    "5) STRUCTURE DISCIPLINE: Respect any explicitly requested folder/directory "
                    "exactly. Never place new files at the project root when a folder, "
                    "subdirectory, or module name was named in the request or implied by the "
                    "existing project layout (e.g. an existing 'src/', 'lib/', 'components/' "
                    "convention must be followed for new files of that kind).\n"
                    "6) LANGUAGE-AGNOSTIC: The project may be Python, .NET/C#, Java, Kotlin, "
                    "Swift, Ruby, Go, Rust, PHP, React/TypeScript/JavaScript, vanilla JS/jQuery, "
                    "Node/Next.js, React Native, or anything else. Infer the real stack from the "
                    "dependency manifests and workspace listing provided, and plan file paths, "
                    "naming conventions, and interfaces appropriate to that exact stack — never "
                    "default to Python/JS assumptions if the manifests show otherwise.\n"
                    "7) NON-CODE WORK: The request may not be about writing code at all — it "
                    "could be reading/summarizing files, extracting data, scanning the codebase "
                    "for an answer, or a design/documentation task. Plan sub-tasks appropriately "
                    "for the actual nature of the request; not everything needs a 'file' output.\n"
                    "8) INSTITUTIONAL MEMORY: When a request introduces a durable design "
                    "decision (a new pattern, library choice, naming convention, or constraint), "
                    "record it in architecture_notes so the whole team remembers it on every "
                    "future task for this project — this is permanent, cross-session memory, "
                    "treat it accordingly.\n"
                    "9) WHEN TO ASK: If the request is genuinely too ambiguous to plan "
                    "responsibly — a critical business rule, a missing target platform, an "
                    "undefined data shape, or a decision that would be expensive to reverse — "
                    "ask via clarifying_questions rather than guessing. Do NOT ask about things "
                    "you can reasonably infer or that don't materially change the implementation; "
                    "over-asking wastes the stakeholder's time as much as under-asking risks "
                    "building the wrong thing.\n"
                    "10) OUTPUT DISCIPLINE: Return only the exact JSON schema you're given — no "
                    "prose, no markdown fences, no trailing commentary."
                ))
                folder_instruction = (
                    f"\n=== MANDATORY: the stakeholder explicitly requested the folder "
                    f"'{requested_folder}' — every new file's path MUST start with "
                    f"'{requested_folder}/'. Never place new files at the project root when a "
                    f"folder was explicitly requested. ===\n"
                    if requested_folder else ""
                )
                decomp_prompt = (
                    f"=== CTO-approved approach (already discussed with your Manager peers — "
                    f"build to this, don't relitigate it) ===\n{cto_final_approach}\n\n"
                    f"=== Manager discussion notes from scoping ===\n" + "\n".join(manager_notes) + "\n\n"
                    f"=== Dependency manifests (actual project stack) ===\n{stack_summary}\n\n"
                    f"=== Workspace file listing ===\n{workspace_summary}\n\n"
                    f"=== Institutional memory (prior decisions, interfaces, history) ===\n{memory_summary}\n\n"
                    f"{folder_instruction}"
                    f"=== New feature request from stakeholder ===\n{task.prompt}\n\n"
                    "Design the solution and return STRICT JSON only — no prose, no markdown "
                    "fences — matching exactly this schema:\n"
                    '{"approach": "1-3 sentence high-level technical approach", '
                    '"architecture_notes": ["any NEW durable decision worth remembering for '
                    'future tasks, omit if none"], '
                    '"clarifying_questions": ["at most 3 SHORT questions, ONLY if there is '
                    'genuine, work-blocking ambiguity — omit entirely (empty list) for any '
                    'request you can reasonably plan as-is"], '
                    '"sub_tasks": [{"id": 1, "name": "short task title", '
                    '"type": "\'file\' (default, produces code/content), \'folder\' (creates an '
                    'empty/standalone directory with no file content), or \'workflow\' (delegates '
                    'to an n8n automation workflow instead of writing code — use this ONLY when '
                    'the work is genuinely an integration/automation/scheduled-job/data-pipeline '
                    'task that an n8n workflow would actually handle, not for regular application '
                    'code)", '
                    '"file": "relative/path/to/file.ext for type=file, relative/path/to/dir for '
                    'type=folder, or a short workflow name for type=workflow", '
                    '"interface": "for type=file: one-line description of what this file must '
                    'expose/do so other sub-tasks can rely on it. For type=workflow: describe '
                    'exactly what the automation should do (trigger, steps, data involved) since '
                    'this is sent directly to n8n as the workflow brief. Empty string for '
                    'type=folder.", '
                    '"depends_on": [ids of sub-tasks this needs completed first]}]}\n'
                    "If clarifying_questions is non-empty, sub_tasks may be an empty list for "
                    "this pass. "
                    f"Otherwise use at most {MAX_SUBTASKS} sub-tasks, ordered so dependencies "
                    "always appear before the sub-tasks that depend on them. Reuse existing "
                    "files/interfaces from institutional memory instead of recreating them. "
                    "If the workspace already contains a real project, modify/extend the "
                    "specific relevant files — do not restructure or recreate the whole project. "
                    "IMPORTANT: do not skip an explicit 'create a folder' request just because "
                    "a file elsewhere in the plan happens to live inside it — if the stakeholder "
                    "asked for a folder to exist (with or without files in it), include an "
                    "explicit type='folder' sub-task for it."
                )

                response = await self.call_model(
                    loop, planning_model, [planner_system, HumanMessage(content=decomp_prompt)],
                    json_mode=True, role_label="Development Manager",
                )

                decomp_tokens = estimate_tokens(decomp_prompt, response)
                task.decomposition_tokens = decomp_tokens
                task.total_tokens += decomp_tokens
                running_total = await token_tracker.add(decomp_tokens)
                self.update_global_tokens_label(running_total)
                self.log_widget.write(
                    f"[gray][Worker {worker_id}][/gray] Architecture planning used "
                    f"[magenta]{decomp_tokens}[/magenta] tokens (running total: {running_total})."
                )

                plan = safe_json_loads(response.content, default=None)

                # --- Human-in-the-loop: the CTO asks the actual stakeholder before
                # committing to a plan built on guesses, exactly like a real architect would.
                clarifying_questions = [q for q in ((plan or {}).get("clarifying_questions") or []) if q]
                if clarifying_questions:
                    numbered = "\n".join(f"{idx + 1}. {q}" for idx, q in enumerate(clarifying_questions[:3]))
                    answer = await self.ask_user(
                        task,
                        f"Before I start building \"{task.prompt[:80]}\", I need a bit more detail:\n{numbered}",
                        role_label="CTO",
                    )
                    decomp_prompt += (
                        f"\n\n=== Stakeholder's answer to your questions ===\n{answer}\n"
                        "Incorporate this directly. Do not ask further questions this time — "
                        "return a final plan with a non-empty sub_tasks array."
                    )
                    response = await self.call_model(
                        loop, planning_model, [planner_system, HumanMessage(content=decomp_prompt)],
                        json_mode=True, role_label="Development Manager",
                    )
                    followup_tokens = estimate_tokens(decomp_prompt, response)
                    task.total_tokens += followup_tokens
                    running_total = await token_tracker.add(followup_tokens)
                    self.update_global_tokens_label(running_total)
                    plan = safe_json_loads(response.content, default=None)

                if plan and isinstance(plan.get("sub_tasks"), list) and plan["sub_tasks"]:
                    sub_task_objs = topo_sort_subtasks(plan["sub_tasks"][:MAX_SUBTASKS])
                    task.plan_summary = plan.get("approach", "")
                    if task.plan_summary:
                        self.log_widget.write(f"[bold cyan][Development Manager][/bold cyan] {task.plan_summary}")
                    new_notes = [n for n in (plan.get("architecture_notes") or []) if n]
                    if new_notes:
                        task.project_memory.setdefault("architecture_notes", []).extend(new_notes)
                        for note in new_notes:
                            self.log_widget.write(f"[bold cyan][Architecture Decision][/bold cyan] {note}")
                else:
                    # Fallback: treat the whole request as one sub-task rather than failing the run
                    sub_task_objs = [{"id": 1, "name": task.prompt, "file": "", "interface": "", "depends_on": []}]
                    self.log_widget.write(
                        "[yellow]⚠ Planner response wasn't valid JSON — falling back to a "
                        "single sub-task for this request.[/yellow]"
                    )

                sub_task_objs, folder_fixes = enforce_requested_folder(sub_task_objs, requested_folder)
                for fix in folder_fixes:
                    self.log_widget.write(f"[yellow]⚠ [Development Manager] Corrected path to respect requested folder: {fix}[/yellow]")

                # --- CTO clears the concrete breakdown for execution — the last approval
                # gate before any agent touches a file. Non-blocking: a concern is logged
                # and tracked as an open item rather than stalling the whole pipeline, since
                # the CTO already approved the overall approach upstream.
                breakdown_system = SystemMessage(content=(
                    "You are the CTO doing a final check on the concrete sub-task breakdown "
                    "before your team starts implementing. You already approved the overall "
                    "approach — just confirm this breakdown actually matches it and nothing "
                    "important is missing. Return STRICT JSON only: {\"approved\": true or "
                    "false, \"feedback\": \"specific concern if any, else empty string\"}"
                ))
                breakdown_prompt = (
                    f"Approved approach: {cto_final_approach}\n"
                    f"Concrete sub-tasks: {json.dumps([{'name': s.get('name'), 'file': s.get('file')} for s in sub_task_objs])}\n\n"
                    "Return JSON only."
                )
                breakdown_verdict = await self.get_approval(task, loop, breakdown_system, breakdown_prompt, "CTO", model_key=planning_model)
                if breakdown_verdict.get("approved", True):
                    self.log_widget.write("[bold green][CTO][/bold green] ✅ Breakdown cleared — team may begin.")
                else:
                    concern = breakdown_verdict.get("feedback", "")
                    self.log_widget.write(f"[bold yellow][CTO][/bold yellow] ⚠ Proceeding with a flagged concern: {concern}")
                    task.project_memory.setdefault("open_items", []).append(f"[{task.id}] CTO concern at kickoff: {concern}")

                task.sub_tasks = [st.get("name", task.prompt) for st in sub_task_objs]
                task.sub_task_records = [
                    {"name": st.get("name", task.prompt), "tokens": 0, "status": "pending"}
                    for st in sub_task_objs
                ]
                self.update_dashboard_ui(task)

                # --- Graphs: real files on disk, not just log text. Org chart is fixed and
                # only needs writing once per project; the dependency graph reflects this
                # specific task's actual breakdown and is regenerated every time.
                if not task.project_memory.get("org_chart_written"):
                    org_chart_rel = write_graph_file(
                        task.target_dir, "org_hierarchy.md", "Team Org Hierarchy", generate_org_chart_mermaid()
                    )
                    task.project_memory["org_chart_written"] = True
                    self.log_widget.write(f"[cyan][Development Manager][/cyan] Generated org chart: {org_chart_rel}")

                dep_graph_rel = write_graph_file(
                    task.target_dir, f"task_{task.id}_dependencies.md",
                    f"Task {task.id} Dependency Graph — {task.prompt[:60]}",
                    generate_dependency_graph_mermaid(sub_task_objs),
                )
                self.log_widget.write(f"[cyan][Development Manager][/cyan] Generated dependency graph: {dep_graph_rel}")

                # --- PHASE 2: Sub-agent execution, each aware of what teammates already built ---
                task.status = "Executing Sub-agents..."
                self.update_dashboard_ui(task)

                for i, sub in enumerate(sub_task_objs):
                    task.sub_task_records[i]["status"] = "running"
                    self.update_dashboard_ui(task)

                    sub_name = sub.get("name", f"Sub-task {i + 1}")
                    target_file = (sub.get("file") or "").strip()
                    interface_note = sub.get("interface", "")

                    # A folder-only sub-task creates an actual directory — no code, no LLM
                    # call, no review/QA needed for an empty/standalone directory. This is
                    # the fix for "create a folder" requests silently turning into a file:
                    # every other sub-task type used to always go through FILEPATH/CODE
                    # generation, which has no way to represent a bare directory.
                    if sub.get("type") == "folder" and target_file:
                        folder_path = os.path.join(task.target_dir, target_file)
                        os.makedirs(folder_path, exist_ok=True)
                        self.log_widget.write(
                            f"[cyan][Development Manager][/cyan] Creating folder directly "
                            f"(no sub-agent needed): {target_file}/"
                        )
                        self.log_widget.write(f"✓ Created folder: [green]{target_file}/[/green]")
                        task.shared_context.append(f"{target_file}/: (folder)")
                        task.sub_task_records[i]["status"] = "done"
                        self.update_dashboard_ui(task)
                        continue

                    # A workflow-type sub-task is automation, not code: send a
                    # notification, kick off a pipeline, sync external systems. n8n
                    # isn't an LLM, so this bypasses the whole code-writing/review/QA
                    # pipeline entirely and just fires the webhook directly.
                    if sub.get("type") == "workflow":
                        self.log_widget.write(
                            f"[cyan][Development Manager][/cyan] Routing to n8n (automation, "
                            f"not code): {sub_name[:60]}"
                        )
                        if not n8n_ready():
                            self.log_widget.write(
                                "[yellow]⚠ n8n not configured (set N8N_WEBHOOK_URL) — skipping "
                                f"this automation step, flagged for follow-up: {sub_name}[/yellow]"
                            )
                            task.project_memory.setdefault("open_items", []).append(
                                f"[{task.id}] n8n workflow step skipped (not configured): {sub_name}"
                            )
                        else:
                            n8n_payload = {
                                "task_id": task.id,
                                "project": task.target_dir,
                                "interface": interface_note,
                                "original_request": task.prompt,
                            }
                            ok, message = await loop.run_in_executor(
                                None, lambda: trigger_n8n_workflow(sub_name, n8n_payload)
                            )
                            if ok:
                                self.log_widget.write(f"✓ n8n workflow triggered: [green]{sub_name}[/green] — {message}")
                                task.shared_context.append(f"n8n:{sub_name}: (workflow triggered)")
                            else:
                                self.log_widget.write(f"[red]✗ n8n trigger failed for {sub_name}: {message}[/red]")
                                task.project_memory.setdefault("open_items", []).append(
                                    f"[{task.id}] n8n workflow failed: {sub_name} — {message}"
                                )
                        task.sub_task_records[i]["status"] = "done"
                        self.update_dashboard_ui(task)
                        continue

                    role_label, persona_sentence, language = select_engineer_persona(target_file, sub)
                    implementer_model = select_model(purpose_from_role(role_label), project_scale)
                    self.log_widget.write(
                        f"[cyan][Development Manager][/cyan] Assigning to [{role_label}]: {sub_name[:60]} "
                        f"(model: {implementer_model})"
                    )
                    self.log_widget.write(
                        f"[gray][Worker {worker_id}][/gray] [{role_label} · {language}] Processing segment: {sub_name[:40]}..."
                    )

                    # Pick up any live "!" corrections the stakeholder sent for this project
                    # since the last sub-task — a real mid-flight steering channel, not just
                    # a post-hoc review.
                    new_corrections = live_corrections.pop(task.target_dir, [])
                    if new_corrections:
                        task.live_corrections.extend(new_corrections)
                        for c in new_corrections:
                            self.log_widget.write(
                                f"[bold magenta][Stakeholder Correction][/bold magenta] Applying to task {task.id}: {c}"
                            )

                    existing_snippet = ""
                    if target_file:
                        full_existing_path = os.path.join(task.target_dir, target_file)
                        if os.path.isfile(full_existing_path):
                            try:
                                with open(full_existing_path, "r", encoding="utf-8") as ef:
                                    existing_snippet = ef.read()[:MAX_EXISTING_FILE_CONTEXT]
                            except Exception:
                                existing_snippet = ""

                    style_note = extract_code_conventions(task.target_dir, os.path.splitext(target_file)[1]) if target_file else ""

                    team_context = (
                        "\n".join(f"- {c}" for c in task.shared_context[-MAX_TEAM_CONTEXT_ITEMS:])
                        if task.shared_context else "No sibling sub-tasks have completed yet."
                    )
                    corrections_note = (
                        "\n".join(f"- {c}" for c in task.live_corrections)
                        if task.live_corrections else ""
                    )

                    exec_system = SystemMessage(content=(
                        f"{persona_sentence}\n\n"
                        "ADDITIONALLY, as a member of this specific team: you are executing "
                        "one precise piece of a larger architecture designed by your Project "
                        "Manager. Stay strictly within scope — do not touch files outside your "
                        "assignment. Your file's exports/behaviour must match the required "
                        "interface exactly so the rest of the team can rely on it without "
                        "asking you anything further. Respect any stakeholder corrections and "
                        "house-style notes given to you below — they override your own default "
                        "habits. Output ONLY the FILEPATH/CODE format requested — no "
                        "explanation, no markdown outside the code fence, no partial files: "
                        "always submit the file's complete final contents, not a diff or a "
                        "snippet to merge by hand."
                    ))
                    exec_prompt = (
                        f"Workspace Folder: '{task.target_dir}'\n"
                        f"Overall approach: {task.plan_summary or 'N/A'}\n"
                        f"Work already completed by teammates:\n{team_context}\n\n"
                        + (f"Stakeholder corrections to respect (override anything conflicting "
                           f"above):\n{corrections_note}\n\n" if corrections_note else "")
                        + (f"{style_note}\n\n" if style_note else "")
                        + f"Your assignment: {sub_name}\n"
                        f"Target file: {target_file or '(choose an appropriate relative path)'}\n"
                        f"Required interface/behaviour: {interface_note or 'N/A'}\n"
                        + (
                            f"\nCurrent contents of that file (edit/extend it, don't discard "
                            f"unrelated code):\n```\n{existing_snippet}\n```\n"
                            if existing_snippet else "\n(This is a new file.)\n"
                        )
                        + "\nOutput your answer strictly in this format:\n"
                          "FILEPATH: <relative path to target file>\nCODE:\n```\n<your complete, final file contents>\n```"
                    )

                    res = await self.call_model(
                        loop, implementer_model, [exec_system, HumanMessage(content=exec_prompt)],
                        role_label=role_label,
                    )
                    content = res.content

                    sub_tokens = estimate_tokens(exec_prompt, res)
                    task.sub_task_records[i]["tokens"] = sub_tokens
                    task.total_tokens += sub_tokens
                    running_total = await token_tracker.add(sub_tokens)
                    self.update_global_tokens_label(running_total)

                    filepath, code_block = extract_file_and_code(content)

                    if filepath and code_block:
                        task.current_file = filepath
                        self.update_dashboard_ui(task)

                        # --- Code review gate: a Staff Engineer checks the submission before
                        # it's allowed to land, exactly like a PR review in a real team.
                        reviewer_system = SystemMessage(content=(
                            "You are a pragmatic Staff Engineer performing code review, the "
                            "quality gate every submission must pass before it ships. Check, "
                            "in order:\n"
                            "1) INTERFACE COMPLIANCE: Does the code actually expose/do what "
                            "the required interface says? Missing or mismatched exports are "
                            "an automatic rejection.\n"
                            "2) CORRECTNESS: Obvious logic errors, off-by-ones, unhandled "
                            "null/empty/edge cases, or code that clearly won't do what the "
                            "assignment describes.\n"
                            "3) COMPLETENESS: Missing imports, undefined references, "
                            "truncated code, or placeholder/TODO stubs where real "
                            "implementation was expected.\n"
                            "4) SECURITY: Injection risks, hardcoded secrets, unsafe "
                            "deserialization, unvalidated external input.\n"
                            "5) CONSISTENCY: Does it fit how the rest of the team is building "
                            "this feature (naming, structure, error handling)?\n"
                            "Be pragmatic — approve solid code that meets the above; don't "
                            "reject over subjective style preferences or minor nitpicks that "
                            "don't affect correctness. When you do reject, your feedback must "
                            "be specific and actionable (what's wrong and what to do about "
                            "it), not vague ('improve quality'). Return STRICT JSON only, no "
                            "prose: {\"approved\": true or false, \"feedback\": \"specific, "
                            "actionable feedback if not approved, else empty string\"}"
                        ))
                        reviewer_model = select_model("review", project_scale, exclude=implementer_model)
                        self.log_widget.write(
                            f"[gray][Staff Engineer][/gray] Reviewing with [cyan]{reviewer_model}[/cyan] "
                            f"(independent of the implementer's {implementer_model})."
                        )
                        for review_round in range(MAX_REVIEW_ROUNDS + 1):
                            review_prompt = (
                                f"File: {filepath}\n"
                                f"Required interface: {interface_note or 'N/A'}\n"
                                f"Overall approach: {task.plan_summary or 'N/A'}\n\n"
                                f"Submitted code:\n```\n{code_block}\n```\n\nReturn JSON only."
                            )
                            review_res = await self.call_model(
                                loop, reviewer_model, [reviewer_system, HumanMessage(content=review_prompt)],
                                json_mode=True, role_label="Staff Engineer",
                            )
                            review_tokens = estimate_tokens(review_prompt, review_res)
                            task.total_tokens += review_tokens
                            running_total = await token_tracker.add(review_tokens)
                            self.update_global_tokens_label(running_total)

                            verdict = safe_json_loads(review_res.content, default={"approved": True, "feedback": ""})
                            if verdict.get("approved", True):
                                self.log_widget.write(f"[green]✓ [Staff Engineer] Review passed for {filepath}.[/green]")
                                break

                            feedback = verdict.get("feedback", "Please address the review feedback.")
                            self.log_widget.write(
                                f"[yellow]⚠ [Staff Engineer] Requested changes to {filepath}: {feedback[:120]}[/yellow]"
                            )
                            if review_round >= MAX_REVIEW_ROUNDS:
                                self.log_widget.write(
                                    f"[yellow]⚠ Accepting best-effort version of {filepath} after "
                                    f"max review rounds.[/yellow]"
                                )
                                break

                            revision_prompt = (
                                exec_prompt
                                + f"\n\nA senior reviewer looked at your first submission and "
                                  f"requested changes: {feedback}\nSubmit the corrected, complete file."
                            )
                            rev_res = await self.call_model(
                                loop, implementer_model, [exec_system, HumanMessage(content=revision_prompt)],
                                role_label=role_label,
                            )
                            rev_tokens = estimate_tokens(revision_prompt, rev_res)
                            task.total_tokens += rev_tokens
                            running_total = await token_tracker.add(rev_tokens)
                            self.update_global_tokens_label(running_total)

                            _, revised_code = extract_file_and_code(rev_res.content)
                            if revised_code:
                                code_block = revised_code
                                self.log_widget.write(f"[green]✓ [{role_label}] Revision applied to {filepath} after review.[/green]")
                            else:
                                break  # Couldn't parse a revision — keep the original rather than lose the file

                        full_path = os.path.join(task.target_dir, filepath)
                        os.makedirs(os.path.dirname(full_path) or task.target_dir, exist_ok=True)

                        # Self-QA: catch obviously broken Python before it lands on disk.
                        # Rechecked repeatedly (not just once) — keeps fixing and re-parsing
                        # until it's clean or the retry budget is exhausted.
                        if filepath.endswith(".py"):
                            for fix_round in range(MAX_SYNTAX_FIX_ROUNDS + 1):
                                try:
                                    ast.parse(code_block)
                                    if fix_round > 0:
                                        self.log_widget.write(
                                            f"[green]✓ [{role_label}] Syntax clean for {filepath} "
                                            f"after {fix_round} fix round(s).[/green]"
                                        )
                                    break
                                except SyntaxError as syn_err:
                                    if fix_round >= MAX_SYNTAX_FIX_ROUNDS:
                                        self.log_widget.write(
                                            f"[red]✗ {filepath} still has syntax issues after "
                                            f"{MAX_SYNTAX_FIX_ROUNDS} fix attempts — writing "
                                            f"best-effort version, flagged for follow-up.[/red]"
                                        )
                                        task.project_memory.setdefault("open_items", []).append(
                                            f"{filepath}: syntax errors persisted after "
                                            f"{MAX_SYNTAX_FIX_ROUNDS} automated fix attempts in task {task.id}."
                                        )
                                        break
                                    self.log_widget.write(
                                        f"[yellow]⚠ Syntax issue in {filepath} (attempt "
                                        f"{fix_round + 1}/{MAX_SYNTAX_FIX_ROUNDS}) — requesting a fix...[/yellow]"
                                    )
                                    fix_prompt = (
                                        f"The following Python code has a syntax error: {syn_err}\n\n"
                                        f"```\n{code_block}\n```\n\n"
                                        "Return only the corrected, complete file in this format:\n"
                                        "FILEPATH: <relative path>\nCODE:\n```\n<corrected code>\n```"
                                    )
                                    fix_res = await self.call_model(
                                        loop, implementer_model, [exec_system, HumanMessage(content=fix_prompt)],
                                        role_label=role_label,
                                    )
                                    fix_tokens = estimate_tokens(fix_prompt, fix_res)
                                    task.total_tokens += fix_tokens
                                    running_total = await token_tracker.add(fix_tokens)
                                    self.update_global_tokens_label(running_total)

                                    _, fixed_code = extract_file_and_code(fix_res.content)
                                    if fixed_code:
                                        code_block = fixed_code
                                    else:
                                        break  # No parseable revision came back — stop retrying blindly

                        with open(full_path, "w", encoding="utf-8") as f:
                            f.write(code_block)

                        # --- QA gate: a Test Engineer writes real tests and actually runs
                        # them, exactly like CI would before a PR is considered done.
                        test_summary = ""
                        is_test_file_itself = "test" in os.path.basename(filepath).lower() or "/tests/" in filepath.replace("\\", "/")
                        if filepath.endswith(".py") and not is_test_file_itself and is_python_testable(code_block):
                            task.sub_task_records[i]["status"] = "testing"
                            self.update_dashboard_ui(task)

                            base_name = os.path.splitext(os.path.basename(filepath))[0]
                            test_rel_path = os.path.join("tests", f"test_{base_name}.py").replace("\\", "/")

                            qa_system = SystemMessage(content=(
                                "You are a Test Engineer (SDET) writing pytest tests for a "
                                "teammate's code — the last automated check before this file "
                                "ships. Requirements:\n"
                                "1) COVERAGE: Test the main happy-path behaviour AND at least "
                                "one meaningful edge case (empty input, invalid input, "
                                "boundary value, or error path) — not just a smoke test that "
                                "the function runs.\n"
                                "2) REAL ASSERTIONS: Assert actual expected values/behaviour, "
                                "never assertions that trivially always pass (e.g. "
                                "assert True, or asserting a value equals itself).\n"
                                "3) ISOLATION: Keep tests self-contained and deterministic — "
                                "no network calls, no external services, no reliance on system "
                                "time/locale, no writes outside temp directories (use tmp_path "
                                "fixtures where file I/O is needed).\n"
                                "4) IMPORTS: Import the module under test using its real "
                                "relative module path from the project root, matching how the "
                                "rest of this project imports things.\n"
                                "5) READABILITY: Name test functions descriptively "
                                "(test_<behaviour>_<condition>) so a failing test tells you "
                                "what broke without reading the assertion."
                            ))
                            qa_prompt = (
                                f"File under test: {filepath}\n"
                                f"Required interface: {interface_note or 'N/A'}\n\n"
                                f"Code:\n```\n{code_block}\n```\n\n"
                                "Output strictly in this format:\n"
                                f"FILEPATH: {test_rel_path}\nCODE:\n```\n<pytest test file contents>\n```"
                            )
                            qa_model = select_model("review", project_scale, exclude=implementer_model)
                            self.log_widget.write(
                                f"[gray][Test Engineer][/gray] Writing tests with [cyan]{qa_model}[/cyan]."
                            )
                            qa_res = await self.call_model(
                                loop, qa_model, [qa_system, HumanMessage(content=qa_prompt)],
                                role_label="Test Engineer",
                            )
                            qa_tokens = estimate_tokens(qa_prompt, qa_res)
                            task.total_tokens += qa_tokens
                            running_total = await token_tracker.add(qa_tokens)
                            self.update_global_tokens_label(running_total)

                            test_filepath, test_code = extract_file_and_code(qa_res.content)
                            if test_filepath and test_code:
                                full_test_path = os.path.join(task.target_dir, test_filepath)
                                os.makedirs(os.path.dirname(full_test_path) or task.target_dir, exist_ok=True)
                                with open(full_test_path, "w", encoding="utf-8") as tf:
                                    tf.write(test_code)

                                for qa_round in range(MAX_QA_ROUNDS + 1):
                                    passed, test_output = await loop.run_in_executor(
                                        None, lambda: run_pytest(full_test_path, task.target_dir)
                                    )
                                    if passed is None:
                                        test_summary = "inconclusive (couldn't run pytest in this environment)"
                                        self.log_widget.write(
                                            f"[dim]ℹ [Test Engineer] Inconclusive for {filepath} — {test_output}[/dim]"
                                        )
                                        break
                                    if passed:
                                        test_summary = "passed"
                                        self.log_widget.write(f"[green]✓ [Test Engineer] Tests passed for {filepath}.[/green]")
                                        break

                                    test_summary = "failed"
                                    self.log_widget.write(
                                        f"[yellow]⚠ [Test Engineer] Tests failed for {filepath}, sending back to {role_label}...[/yellow]"
                                    )
                                    if qa_round >= MAX_QA_ROUNDS:
                                        self.log_widget.write(
                                            f"[red]✗ [Test Engineer] {filepath} still failing tests after fix attempt — "
                                            f"shipped as best-effort, flagged for follow-up.[/red]"
                                        )
                                        task.project_memory.setdefault("open_items", []).append(
                                            f"{filepath}: failing QA tests as of task {task.id} — needs follow-up."
                                        )
                                        break

                                    qa_fix_prompt = (
                                        exec_prompt
                                        + f"\n\nQA ran the following pytest tests against your code and they "
                                          f"FAILED:\n```\n{test_output}\n```\nFix the implementation so the "
                                          "tests pass. Submit the corrected, complete file."
                                    )
                                    qa_fix_res = await self.call_model(
                                        loop, implementer_model, [exec_system, HumanMessage(content=qa_fix_prompt)],
                                        role_label=role_label,
                                    )
                                    qa_fix_tokens = estimate_tokens(qa_fix_prompt, qa_fix_res)
                                    task.total_tokens += qa_fix_tokens
                                    running_total = await token_tracker.add(qa_fix_tokens)
                                    self.update_global_tokens_label(running_total)

                                    _, qa_fixed_code = extract_file_and_code(qa_fix_res.content)
                                    if qa_fixed_code:
                                        code_block = qa_fixed_code
                                        with open(full_path, "w", encoding="utf-8") as f:
                                            f.write(code_block)
                                    else:
                                        break
                            else:
                                test_summary = "no tests generated"

                        signature_note = extract_signatures(code_block)
                        task.shared_context.append(f"{filepath}: {signature_note}")
                        task.project_memory.setdefault("file_registry", {})[filepath] = {
                            "interface": interface_note or signature_note,
                            "signature": signature_note,
                            "last_task": task.id,
                            "tests": test_summary or "not applicable",
                            "built_by": role_label,
                            "model": implementer_model,
                            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        }

                        self.log_widget.write(
                            f"✓ [{role_label}] Written adjustments to file: [green]{filepath}[/green] "
                            f"([magenta]{sub_tokens}[/magenta] tokens, running total: {running_total})"
                        )
                    else:
                        self.log_widget.write(
                            f"[yellow]⚠ Sub-agent {i + 1} returned no usable file/code block "
                            f"({sub_tokens} tokens, running total: {running_total}).[/yellow]"
                        )

                    task.sub_task_records[i]["status"] = "done"
                    self.update_dashboard_ui(task)

                # --- PHASE 3: Manager sign-off — the CTO reviews the team's completed work
                # as a whole against the original request before it's considered "shipped",
                # exactly like a release manager approving a PR before merge.
                task.status = "Manager Sign-off..."
                self.update_dashboard_ui(task)

                # --- PHASE 3a: The CTO drafts an internal sign-off summary first — the same
                # honesty check as before, used to brief the Manager rather than to auto-ship.
                task.status = "Preparing Manager Review..."
                self.update_dashboard_ui(task)

                delivered_files = "\n".join(f"- {c}" for c in task.shared_context) or "(no files were written)"
                signoff_system = SystemMessage(content=(
                    "You are the CTO preparing a sign-off summary for the Engineering "
                    "Manager to present to the stakeholder. Your job:\n"
                    "1) COMPARE DELIVERED VS REQUESTED: Go through the original request "
                    "point by point and check whether the delivered files actually cover "
                    "each part of it.\n"
                    "2) BE HONEST ABOUT GAPS: List anything missing, partially done, or "
                    "risky (untested, flagged by QA, outside the original interface) as a "
                    "specific open_item — vague reassurance or approving incomplete work "
                    "erodes trust in the team and in you.\n"
                    "3) SUMMARY QUALITY: Write the summary the way you'd actually brief a "
                    "non-technical stakeholder — plain language, what changed and why it "
                    "matters, not a list of filenames.\n"
                    "4) READY_TO_SHIP: Only true if there are no open_items that would "
                    "materially matter to the stakeholder. Return STRICT JSON only: "
                    "{\"ready_to_ship\": true or false, "
                    "\"summary\": \"1-2 sentence summary of what was delivered\", "
                    "\"open_items\": [\"specific gap or follow-up\", ...] or []}"
                ))
                signoff_prompt = (
                    f"Original request: {task.prompt}\n"
                    f"Planned approach: {task.plan_summary or 'N/A'}\n\n"
                    f"Files delivered by the team:\n{delivered_files}\n\n"
                    "Does this fulfill the original request? Return JSON only."
                )
                signoff_res = await self.call_model(
                    loop, planning_model, [signoff_system, HumanMessage(content=signoff_prompt)],
                    json_mode=True, role_label="CTO",
                )
                signoff_tokens = estimate_tokens(signoff_prompt, signoff_res)
                task.total_tokens += signoff_tokens
                running_total = await token_tracker.add(signoff_tokens)
                self.update_global_tokens_label(running_total)

                signoff = safe_json_loads(signoff_res.content, default={"ready_to_ship": True, "summary": "", "open_items": []})
                for item in signoff.get("open_items") or []:
                    task.project_memory.setdefault("open_items", []).append(f"[{task.id}] {item}")

                # --- PHASE 3b: Real human review. The Manager brings the CTO's summary to
                # the actual stakeholder and asks — this is not simulated, it genuinely
                # blocks on your reply in the prompt box.
                task.status = "Manager Sign-off..."
                self.update_dashboard_ui(task)

                files_touched = ", ".join(c.split(":", 1)[0] for c in task.shared_context) or "no files"
                caveat = (
                    f" The CTO flagged some open items: {'; '.join(signoff.get('open_items') or [])}."
                    if not signoff.get("ready_to_ship", True) and signoff.get("open_items")
                    else ""
                )
                review_question = (
                    f"Here's what the team delivered for \"{task.prompt[:100]}\": "
                    f"{signoff.get('summary', 'Work completed.')} Files touched: {files_touched}."
                    f"{caveat} Does this look good, or is there anything you'd like changed "
                    "or added? Reply 'approve' if it's good, or just describe what's missing."
                )
                user_reply = await self.ask_user(task, review_question, role_label="IT Manager")

                approval_phrases = ("approve", "approved", "looks good", "lgtm", "ship it", "yes", "good", "ok", "okay", "sounds good")
                is_approved = any(phrase in user_reply.strip().lower() for phrase in approval_phrases)

                if is_approved:
                    self.log_widget.write(
                        f"[bold green][IT Manager][/bold green] ✅ User approved — task {task.id} signed off and shipped."
                    )
                else:
                    self.log_widget.write(
                        f"[bold yellow][IT Manager][/bold yellow] Feedback received — dispatching a follow-up task to address it."
                    )
                    task.project_memory.setdefault("open_items", []).append(
                        f"[{task.id}] User feedback at review: {user_reply}"
                    )
                    follow_up = BackgroundTask(
                        prompt=f"Follow-up on \"{task.prompt}\" (task {task.id}): {user_reply}",
                        target_dir=task.target_dir,
                    )
                    self.update_dashboard_ui(follow_up)
                    await task_queue.put(follow_up)
                    self.log_widget.write(
                        f"[bold magenta][IT Manager][/bold magenta] Dispatched follow-up Task ID [yellow]{follow_up.id}[/yellow]."
                    )

                # --- PHASE 4: Close the loop — write this task into the team's permanent
                # institutional memory so every future task on this project starts smarter.
                task.project_memory.setdefault("changelog", []).append({
                    "task_id": task.id,
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "prompt": task.prompt,
                    "approach": task.plan_summary,
                    "files": [c.split(":", 1)[0] for c in task.shared_context],
                })
                async with _get_memory_lock(task.target_dir):
                    save_project_memory(task.target_dir, task.project_memory)

                task.status = "Completed"
                task.current_file = "None"
                self.update_dashboard_ui(task)
                self.log_widget.write(
                    f"[bold green]✓ Task {task.id} completed.[/bold green] "
                    f"Total tokens for this task: [magenta]{task.total_tokens}[/magenta] "
                    f"across {len(task.sub_task_records)} sub-agent(s). "
                    f"[dim]Institutional memory updated for '{task.target_dir}'.[/dim]"
                )
                self.log_widget.write(
                    f"[bold cyan][Finance Manager][/bold cyan] Task {task.id} cost "
                    f"[magenta]{task.total_tokens}[/magenta] tokens · session total so far: "
                    f"[magenta]{token_tracker.total}[/magenta] tokens."
                )

            except Exception as e:
                task.status = f"Failed: {str(e)}"
                self.update_dashboard_ui(task)
                self.log_widget.write(f"[bold red]✗ Task {task.id} failed: {e}[/bold red]")
            finally:
                task_queue.task_done()
                remaining = await active_agents.decrement()
                if remaining == 0:
                    self.log_widget.write(
                        "[dim]💤 All agents idle — sleep prevention released.[/dim]"
                    )

    def action_clear_logs(self) -> None:
        self.query_one("#system-log", RichLog).clear()

    def on_unmount(self) -> None:
        """Guarantees the OS sleep-inhibitor process is cleaned up if the app quits
        while sub-agents are still mid-task, so it never gets orphaned."""
        keep_awake.stop()

if __name__ == "__main__":
    AgentHubApp().run()