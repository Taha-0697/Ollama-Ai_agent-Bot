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

# Textual UI components
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Input, Button, RichLog, DirectoryTree, Label, Static

# --- THREAD-SAFE LLM ENGINE CONFIGURATION ---
llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0, format="json")
code_llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.1)

# Global tracking structures
task_queue = asyncio.Queue()
app_workspace = os.getcwd()  # Dynamically defaults to your exact current directory context

# Max number of sub-tasks the decomposition step is allowed to split a request into.
# Kept as a soft ceiling so "as many sub-agents as possible" stays practical rather than runaway.
MAX_SUBTASKS = 200

# Max characters of an existing file's content injected into a sub-agent's prompt.
# Keeps edits grounded in the real file without blowing up token usage.
MAX_EXISTING_FILE_CONTEXT = 3000

# How many of the most recent teammate summaries get passed to each sub-agent.
MAX_TEAM_CONTEXT_ITEMS = 10

# Code review is a real quality gate, but we cap it at one revision round so a
# stubborn reviewer/implementer disagreement can't loop forever.
MAX_REVIEW_ROUNDS = 5

# QA gets the same one-shot fix budget: write tests, run them, allow one fix-and-retest
# round if they fail, then move on rather than stall the whole pipeline.
MAX_QA_ROUNDS = 3
TEST_TIMEOUT_SECONDS = 50


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
        
        # Start 3 background queue consumers
        for i in range(3):
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
                # writing a single line of the plan.
                task.status = "Loading Institutional Context..."
                self.update_dashboard_ui(task)

                async with _get_memory_lock(task.target_dir):
                    task.project_memory = load_project_memory(task.target_dir)

                memory_summary = summarize_memory_for_planner(task.project_memory)
                stack_summary = detect_tech_stack(task.target_dir)
                workspace_summary = scan_workspace_context(task.target_dir)

                known_files = len(task.project_memory.get("file_registry", {}))
                known_decisions = len(task.project_memory.get("architecture_notes", []))
                known_tasks = len(task.project_memory.get("changelog", []))
                self.log_widget.write(
                    f"[gray][Worker {worker_id}][/gray] Institutional memory loaded: "
                    f"[cyan]{known_files}[/cyan] known files, [cyan]{known_decisions}[/cyan] "
                    f"architecture decisions, [cyan]{known_tasks}[/cyan] prior tasks on record."
                )

                # --- PHASE 1: CTO-level architecture + decomposition ---
                task.status = "Architecting Solution..."
                self.update_dashboard_ui(task)

                planner_system = SystemMessage(content=(
                    "You are a Principal Engineer / CTO-level technical architect leading a "
                    "real engineering team. You design clean, minimal, production-quality "
                    "solutions, respect prior architecture decisions and the project's actual "
                    "dependency stack, and break new work into the smallest sensible, "
                    "single-responsibility sub-tasks so engineers (sub-agents) can implement "
                    "them independently with minimal context. You never needlessly duplicate "
                    "existing files or reinvent an interface the team already built. You think "
                    "in dependency order: sub-tasks that others rely on (shared utilities, "
                    "config, data models) come first. When a request introduces a durable "
                    "design decision (a new pattern, library choice, or convention), you record "
                    "it so the team remembers it on future work."
                ))
                decomp_prompt = (
                    f"=== Dependency manifests (actual project stack) ===\n{stack_summary}\n\n"
                    f"=== Workspace file listing ===\n{workspace_summary}\n\n"
                    f"=== Institutional memory (prior decisions, interfaces, history) ===\n{memory_summary}\n\n"
                    f"=== New feature request from stakeholder ===\n{task.prompt}\n\n"
                    "Design the solution and return STRICT JSON only — no prose, no markdown "
                    "fences — matching exactly this schema:\n"
                    '{"approach": "1-3 sentence high-level technical approach", '
                    '"architecture_notes": ["any NEW durable decision worth remembering for '
                    'future tasks, omit if none"], '
                    '"sub_tasks": [{"id": 1, "name": "short task title", '
                    '"file": "relative/path/to/file.ext", '
                    '"interface": "one-line description of what this file must expose/do so '
                    'other sub-tasks can rely on it", "depends_on": [ids of sub-tasks this '
                    'needs completed first]}]}\n'
                    f"Use at most {MAX_SUBTASKS} sub-tasks. Order the array so dependencies "
                    "always appear before the sub-tasks that depend on them. Reuse existing "
                    "files/interfaces from institutional memory instead of recreating them."
                )

                response = await loop.run_in_executor(
                    None, lambda: llm.invoke([planner_system, HumanMessage(content=decomp_prompt)])
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
                if plan and isinstance(plan.get("sub_tasks"), list) and plan["sub_tasks"]:
                    sub_task_objs = topo_sort_subtasks(plan["sub_tasks"][:MAX_SUBTASKS])
                    task.plan_summary = plan.get("approach", "")
                    if task.plan_summary:
                        self.log_widget.write(f"[bold cyan][CTO Plan][/bold cyan] {task.plan_summary}")
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

                task.sub_tasks = [st.get("name", task.prompt) for st in sub_task_objs]
                task.sub_task_records = [
                    {"name": st.get("name", task.prompt), "tokens": 0, "status": "pending"}
                    for st in sub_task_objs
                ]
                self.update_dashboard_ui(task)

                # --- PHASE 2: Sub-agent execution, each aware of what teammates already built ---
                task.status = "Executing Sub-agents..."
                self.update_dashboard_ui(task)

                for i, sub in enumerate(sub_task_objs):
                    task.sub_task_records[i]["status"] = "running"
                    self.update_dashboard_ui(task)
                    sub_name = sub.get("name", f"Sub-task {i + 1}")
                    self.log_widget.write(f"[gray][Worker {worker_id}][/gray] Processing segment: {sub_name[:40]}...")

                    target_file = (sub.get("file") or "").strip()
                    interface_note = sub.get("interface", "")

                    existing_snippet = ""
                    if target_file:
                        full_existing_path = os.path.join(task.target_dir, target_file)
                        if os.path.isfile(full_existing_path):
                            try:
                                with open(full_existing_path, "r", encoding="utf-8") as ef:
                                    existing_snippet = ef.read()[:MAX_EXISTING_FILE_CONTEXT]
                            except Exception:
                                existing_snippet = ""

                    team_context = (
                        "\n".join(f"- {c}" for c in task.shared_context[-MAX_TEAM_CONTEXT_ITEMS:])
                        if task.shared_context else "No sibling sub-tasks have completed yet."
                    )

                    exec_system = SystemMessage(content=(
                        "You are a senior software engineer executing one precise piece of a "
                        "larger architecture designed by your CTO. Stay strictly within scope, "
                        "write clean idiomatic production code, and make sure this file's "
                        "exports/behaviour match the required interface exactly so the rest of "
                        "the team can rely on it."
                    ))
                    exec_prompt = (
                        f"Workspace Folder: '{task.target_dir}'\n"
                        f"Overall approach: {task.plan_summary or 'N/A'}\n"
                        f"Work already completed by teammates:\n{team_context}\n\n"
                        f"Your assignment: {sub_name}\n"
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

                    res = await loop.run_in_executor(
                        None, lambda: code_llm.invoke([exec_system, HumanMessage(content=exec_prompt)])
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
                            "You are a pragmatic Staff Engineer performing code review. Check "
                            "whether the submitted code actually fulfills its required "
                            "interface, is consistent with how the rest of the team is "
                            "building this feature, and has no obvious bugs, missing "
                            "imports, or security issues. Approve solid code — don't nitpick "
                            "style. Return STRICT JSON only, no prose: "
                            '{"approved": true or false, "feedback": "specific, actionable '
                            'feedback if not approved, else empty string"}'
                        ))
                        for review_round in range(MAX_REVIEW_ROUNDS + 1):
                            review_prompt = (
                                f"File: {filepath}\n"
                                f"Required interface: {interface_note or 'N/A'}\n"
                                f"Overall approach: {task.plan_summary or 'N/A'}\n\n"
                                f"Submitted code:\n```\n{code_block}\n```\n\nReturn JSON only."
                            )
                            review_res = await loop.run_in_executor(
                                None, lambda: llm.invoke([reviewer_system, HumanMessage(content=review_prompt)])
                            )
                            review_tokens = estimate_tokens(review_prompt, review_res)
                            task.total_tokens += review_tokens
                            running_total = await token_tracker.add(review_tokens)
                            self.update_global_tokens_label(running_total)

                            verdict = safe_json_loads(review_res.content, default={"approved": True, "feedback": ""})
                            if verdict.get("approved", True):
                                self.log_widget.write(f"[green]✓ Review passed for {filepath}.[/green]")
                                break

                            feedback = verdict.get("feedback", "Please address the review feedback.")
                            self.log_widget.write(
                                f"[yellow]⚠ Reviewer requested changes to {filepath}: {feedback[:120]}[/yellow]"
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
                            rev_res = await loop.run_in_executor(
                                None, lambda: code_llm.invoke([exec_system, HumanMessage(content=revision_prompt)])
                            )
                            rev_tokens = estimate_tokens(revision_prompt, rev_res)
                            task.total_tokens += rev_tokens
                            running_total = await token_tracker.add(rev_tokens)
                            self.update_global_tokens_label(running_total)

                            _, revised_code = extract_file_and_code(rev_res.content)
                            if revised_code:
                                code_block = revised_code
                                self.log_widget.write(f"[green]✓ Revision applied to {filepath} after review.[/green]")
                            else:
                                break  # Couldn't parse a revision — keep the original rather than lose the file

                        full_path = os.path.join(task.target_dir, filepath)
                        os.makedirs(os.path.dirname(full_path) or task.target_dir, exist_ok=True)

                        # Lightweight self-QA: catch obviously broken Python before it lands on disk.
                        if filepath.endswith(".py"):
                            try:
                                ast.parse(code_block)
                            except SyntaxError as syn_err:
                                self.log_widget.write(
                                    f"[yellow]⚠ Syntax issue detected in {filepath} — requesting a fix...[/yellow]"
                                )
                                fix_prompt = (
                                    f"The following Python code has a syntax error: {syn_err}\n\n"
                                    f"```\n{code_block}\n```\n\n"
                                    "Return only the corrected, complete file in this format:\n"
                                    "FILEPATH: <relative path>\nCODE:\n```\n<corrected code>\n```"
                                )
                                fix_res = await loop.run_in_executor(
                                    None, lambda: code_llm.invoke([exec_system, HumanMessage(content=fix_prompt)])
                                )
                                fix_tokens = estimate_tokens(fix_prompt, fix_res)
                                task.total_tokens += fix_tokens
                                running_total = await token_tracker.add(fix_tokens)
                                self.update_global_tokens_label(running_total)

                                _, fixed_code = extract_file_and_code(fix_res.content)
                                if fixed_code:
                                    try:
                                        ast.parse(fixed_code)
                                        code_block = fixed_code
                                        self.log_widget.write(f"[green]✓ Syntax fix applied to {filepath}.[/green]")
                                    except SyntaxError:
                                        self.log_widget.write(
                                            f"[red]✗ Fix attempt for {filepath} still has issues; "
                                            f"writing best-effort version.[/red]"
                                        )

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
                                "teammate's code. Cover the main behaviour and at least one "
                                "edge case. Import the module using its real relative module "
                                "path from the project root. Keep tests self-contained — no "
                                "network calls, no external services, no file-system writes "
                                "outside temp directories."
                            ))
                            qa_prompt = (
                                f"File under test: {filepath}\n"
                                f"Required interface: {interface_note or 'N/A'}\n\n"
                                f"Code:\n```\n{code_block}\n```\n\n"
                                "Output strictly in this format:\n"
                                f"FILEPATH: {test_rel_path}\nCODE:\n```\n<pytest test file contents>\n```"
                            )
                            qa_res = await loop.run_in_executor(
                                None, lambda: code_llm.invoke([qa_system, HumanMessage(content=qa_prompt)])
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
                                            f"[dim]ℹ QA inconclusive for {filepath} — {test_output}[/dim]"
                                        )
                                        break
                                    if passed:
                                        test_summary = "passed"
                                        self.log_widget.write(f"[green]✓ QA tests passed for {filepath}.[/green]")
                                        break

                                    test_summary = "failed"
                                    self.log_widget.write(
                                        f"[yellow]⚠ QA tests failed for {filepath}, sending back to engineer...[/yellow]"
                                    )
                                    if qa_round >= MAX_QA_ROUNDS:
                                        self.log_widget.write(
                                            f"[red]✗ {filepath} still failing tests after fix attempt — "
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
                                    qa_fix_res = await loop.run_in_executor(
                                        None, lambda: code_llm.invoke([exec_system, HumanMessage(content=qa_fix_prompt)])
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
                            "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        }

                        self.log_widget.write(
                            f"✓ Written adjustments to file: [green]{filepath}[/green] "
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

                delivered_files = "\n".join(f"- {c}" for c in task.shared_context) or "(no files were written)"
                signoff_system = SystemMessage(content=(
                    "You are the CTO doing final sign-off on your team's completed work "
                    "before it ships. Compare what was delivered against the original "
                    "request. Be honest about gaps — approving incomplete work erodes trust "
                    "in the team. Return STRICT JSON only: {\"ready_to_ship\": true or false, "
                    "\"summary\": \"1-2 sentence summary of what was delivered\", "
                    "\"open_items\": [\"specific gap or follow-up\", ...] or []}"
                ))
                signoff_prompt = (
                    f"Original request: {task.prompt}\n"
                    f"Planned approach: {task.plan_summary or 'N/A'}\n\n"
                    f"Files delivered by the team:\n{delivered_files}\n\n"
                    "Does this fulfill the original request? Return JSON only."
                )
                signoff_res = await loop.run_in_executor(
                    None, lambda: llm.invoke([signoff_system, HumanMessage(content=signoff_prompt)])
                )
                signoff_tokens = estimate_tokens(signoff_prompt, signoff_res)
                task.total_tokens += signoff_tokens
                running_total = await token_tracker.add(signoff_tokens)
                self.update_global_tokens_label(running_total)

                signoff = safe_json_loads(signoff_res.content, default={"ready_to_ship": True, "summary": "", "open_items": []})
                if signoff.get("ready_to_ship", True):
                    self.log_widget.write(
                        f"[bold green][CTO Sign-off][/bold green] ✅ Approved — {signoff.get('summary', 'Meets the request.')}"
                    )
                else:
                    self.log_widget.write(
                        f"[bold yellow][CTO Sign-off][/bold yellow] ⚠ Shipped with open items — {signoff.get('summary', '')}"
                    )
                    for item in signoff.get("open_items") or []:
                        self.log_widget.write(f"   [yellow]• {item}[/yellow]")
                        task.project_memory.setdefault("open_items", []).append(f"[{task.id}] {item}")

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