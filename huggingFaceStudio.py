"""
HF AI Studio - 本機執行 Hugging Face 語言模型的桌面介面
pip install PyQt6 transformers torch markdown pygments psutil
"""
import os
import sys
import gc
import html
import time
import threading
from string import Template

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer,
    StoppingCriteria, StoppingCriteriaList,
)
from PyQt6.QtCore import Qt, QThread, QTimer, QSettings, pyqtSignal
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QLineEdit, QPlainTextEdit, QTextBrowser, QPushButton, QProgressBar,
    QSlider, QSpinBox, QDoubleSpinBox, QAbstractSpinBox, QGroupBox, QScrollArea,
    QSplitter, QStatusBar, QFileDialog, QMessageBox,
)

# ---------- 選用套件 ----------
try:
    import markdown as mdlib
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


PRESET_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]
DEFAULT_SYSTEM = "You are a helpful assistant."

# ---------- 主題配色 ----------
THEMES = {
    "dark": {
        "bg": "#0f1115", "panel": "#161a21", "border": "#2a303b", "text": "#e6e9ef",
        "subtext": "#8b93a3", "accent": "#5b8cff", "accent_hover": "#7aa2ff",
        "accent_disabled": "#34425f", "input_bg": "#1c212a", "hover": "#222833",
        "user_bubble": "#1f2a44", "ai_bubble": "#1a1f28", "chat_bg": "#12151b",
        "code_bg": "#0b0d11", "danger": "#e5534b", "danger_hover": "#f06a63",
        "success": "#3fb950",
    },
    "light": {
        "bg": "#f5f6f8", "panel": "#ffffff", "border": "#dfe3ea", "text": "#1f2328",
        "subtext": "#6b7280", "accent": "#3b6ef5", "accent_hover": "#5a86ff",
        "accent_disabled": "#a9bdf5", "input_bg": "#ffffff", "hover": "#f0f2f5",
        "user_bubble": "#e8efff", "ai_bubble": "#f7f8fa", "chat_bg": "#ffffff",
        "code_bg": "#f3f4f6", "danger": "#d93f36", "danger_hover": "#e8574f",
        "success": "#1a7f37",
    },
}

QSS_TEMPLATE = Template("""
* { font-family: "Segoe UI", "Microsoft JhengHei UI", "PingFang TC", "Noto Sans CJK TC", sans-serif;
    font-size: 13px; color: $text; }
QMainWindow, QDialog, QMessageBox, #central { background: $bg; }
#sidebar, #sidebarContent { background: $panel; }
#sidebar { border-right: 1px solid $border; }
#appTitle { font-size: 19px; font-weight: 700; }
#appSubtitle, #hint { color: $subtext; font-size: 11px; }
#chatTitle { font-size: 15px; font-weight: 600; }
QGroupBox { border: 1px solid $border; border-radius: 10px; margin-top: 16px;
            padding: 14px 10px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                   left: 12px; padding: 0 4px; color: $subtext; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background: $input_bg; border: 1px solid $border; border-radius: 6px; padding: 5px 8px;
    selection-background-color: $accent; selection-color: #ffffff; }
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus {
    border: 1px solid $accent; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView { background: $input_bg; border: 1px solid $border;
    selection-background-color: $accent; outline: none; }
QPushButton { background: $accent; color: #ffffff; border: none; border-radius: 6px;
              padding: 7px 14px; font-weight: 600; }
QPushButton:hover { background: $accent_hover; }
QPushButton:disabled { background: $accent_disabled; color: rgba(255,255,255,0.6); }
QPushButton#secondary { background: transparent; color: $text; border: 1px solid $border; }
QPushButton#secondary:hover { background: $hover; }
QPushButton#secondary:disabled { color: $subtext; }
QPushButton#danger { background: $danger; }
QPushButton#danger:hover { background: $danger_hover; }
QPushButton#danger:disabled { background: $accent_disabled; }
QTextBrowser#chat { background: $chat_bg; border: 1px solid $border; border-radius: 10px;
                    padding: 8px; font-size: ${chat_font}px; }
QSlider::groove:horizontal { height: 4px; background: $border; border-radius: 2px; }
QSlider::sub-page:horizontal { background: $accent; border-radius: 2px; }
QSlider::handle:horizontal { background: $accent; width: 14px; height: 14px;
                             margin: -5px 0; border-radius: 7px; }
QProgressBar { background: $border; border: none; border-radius: 3px; }
QProgressBar::chunk { background: $accent; border-radius: 3px; }
QProgressBar#modelProgress { background: $hover; border: 1px solid $border; border-radius: 4px; height: 8px; }
QProgressBar#modelProgress::chunk { background: $accent; border-radius: 3px; }
#progressDesc { color: $subtext; font-size: 11px; }
#progressPct { color: $accent; font-size: 12px; font-weight: 700; }
QStatusBar { background: $panel; border-top: 1px solid $border; }
QStatusBar QLabel { color: $subtext; font-size: 12px; padding: 0 6px; }
QScrollArea { border: none; background: $panel; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: $border; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSplitter::handle { background: $border; width: 1px; }
QToolTip { background: $panel; color: $text; border: 1px solid $border; padding: 4px; }
""")


def doc_css(p):
    """對話框內 HTML 文件的樣式（程式碼區塊、連結、表格）"""
    mono = '"Cascadia Code", Consolas, "Courier New", monospace'
    return f"""
        pre {{ background-color: {p['code_bg']}; font-family: {mono}; }}
        code {{ background-color: {p['code_bg']}; font-family: {mono}; }}
        a {{ color: {p['accent']}; }}
        th {{ background-color: {p['hover']}; }}
        blockquote {{ color: {p['subtext']}; }}
    """


def md_to_html(text, p, dark):
    """Markdown → HTML（含程式碼高亮與表格框線）"""
    if HAS_MARKDOWN:
        try:
            out = mdlib.markdown(
                text,
                extensions=["fenced_code", "tables", "codehilite", "nl2br", "sane_lists"],
                extension_configs={"codehilite": {
                    "noclasses": True, "guess_lang": False,
                    "pygments_style": "monokai" if dark else "friendly",
                }},
            )
            return out.replace(
                "<table>",
                f'<table border="1" cellspacing="0" cellpadding="6" '
                f'style="border-collapse:collapse; border-color:{p["border"]};">',
            )
        except Exception:
            pass
    return html.escape(text).replace("\n", "<br>")


def format_bytes(num_bytes):
    if not num_bytes or num_bytes <= 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


# ---------- 裝置 / 精度 ----------
def mps_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(choice):
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_available():
            return "mps"
        return "cpu"
    return choice


def resolve_dtype(choice, device):
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if choice in mapping:
        if device == "cpu" and choice == "float16":
            return torch.float32
        return mapping[choice]
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


# ---------- Prompt 組裝（Chat Template） ----------
def merge_consecutive(messages):
    """合併連續相同角色的訊息（部分模板要求 user/assistant 交替）"""
    merged = []
    for m in messages:
        if merged and merged[-1]["role"] == m["role"] and m["role"] != "system":
            merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n\n" + m["content"]}
        else:
            merged.append(dict(m))
    return merged


def build_prompt(tokenizer, messages):
    """回傳 (prompt 文字, 是否使用 chat template)"""
    messages = merge_consecutive(messages)
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), True
        except Exception:
            # 有些模型（如 Gemma）不支援 system 角色 → 合併進第一則 user 訊息
            sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
            msgs = [m for m in messages if m["role"] != "system"]
            if sys_txt and msgs:
                msgs[0] = {"role": msgs[0]["role"], "content": f"{sys_txt}\n\n{msgs[0]['content']}"}
            try:
                return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), True
            except Exception:
                pass
    # 沒有模板的 Base 模型 → 純文字格式
    names = {"system": "System", "user": "User", "assistant": "Assistant"}
    lines = [f"{names[m['role']]}: {m['content']}" for m in messages]
    lines.append("Assistant:")
    return "\n".join(lines), False


# ---------- 背景執行緒：載入模型 ----------
_progress_hooks = threading.local()

try:
    import tqdm.std
    _orig_tqdm_init = tqdm.std.tqdm.__init__
    _orig_tqdm_update = tqdm.std.tqdm.update
    _orig_tqdm_close = tqdm.std.tqdm.close

    def _hooked_tqdm_init(self, *args, **kwargs):
        kwargs["disable"] = False
        kwargs["mininterval"] = 0.05
        _orig_tqdm_init(self, *args, **kwargs)
        self.disable = False
        cb = getattr(_progress_hooks, "callback", None)
        if cb:
            cb("init", self)

    def _hooked_tqdm_update(self, n=1):
        res = _orig_tqdm_update(self, n)
        cb = getattr(_progress_hooks, "callback", None)
        if cb:
            cb("update", self)
        return res

    def _hooked_tqdm_close(self):
        cb = getattr(_progress_hooks, "callback", None)
        if cb:
            cb("close", self)
        return _orig_tqdm_close(self)

    tqdm.std.tqdm.__init__ = _hooked_tqdm_init
    tqdm.std.tqdm.update = _hooked_tqdm_update
    tqdm.std.tqdm.close = _hooked_tqdm_close
    os.environ["TQDM_POSITION"] = "-1"
except Exception:
    pass


class ModelLoaderThread(QThread):
    progress = pyqtSignal(int, str)
    loaded = pyqtSignal(object, object, dict)
    failed = pyqtSignal(str)

    def __init__(self, model_id, device, dtype_choice, token):
        super().__init__()
        self.model_id, self.device, self.dtype_choice, self.token = model_id, device, dtype_choice, token

    def run(self):
        last_pct = 0

        def on_tqdm(event, bar):
            nonlocal last_pct
            desc = (getattr(bar, "desc", "") or "").strip()
            total = getattr(bar, "total", None)
            n = getattr(bar, "n", 0)
            unit = getattr(bar, "unit", "")

            if not total or total <= 0:
                pct = last_pct
            else:
                pct = int(min(100, max(0, (n / total) * 100)))

            if unit == "B" or "b" in unit.lower():
                size_info = f"({format_bytes(n)} / {format_bytes(total)})" if total and total > 0 else f"({format_bytes(n)})"
                msg = f"下載 {desc} {size_info}".strip() if desc else f"下載權重檔案 {size_info}"
            elif desc in ("Loading weights", "Loading checkpoint shards"):
                msg = f"載入模型權重 ({n} / {total})"
            elif desc:
                msg = f"{desc} ({n} / {total})"
            else:
                msg = f"載入中 ({pct}%)"

            last_pct = pct
            self.progress.emit(pct, msg)

        _progress_hooks.callback = on_tqdm
        try:
            self.progress.emit(0, "正在連接與檢查模型…")
            kw = {"token": self.token} if self.token else {}

            self.progress.emit(5, "正在載入 Tokenizer…")
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kw)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            self.progress.emit(10, "正在載入模型結構與權重…")
            dtype = resolve_dtype(self.dtype_choice, self.device)
            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype, **kw)
            except (TypeError, ValueError):
                model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype, **kw)

            if self.device != "cpu":
                self.progress.emit(95, f"正在配置模型至 {self.device.upper()}…")
            model.to(self.device)
            model.eval()
            self.progress.emit(100, "載入完成！")

            info = {
                "model_id": self.model_id,
                "device": self.device,
                "dtype": str(dtype).replace("torch.", ""),
                "params": sum(p.numel() for p in model.parameters()) / 1e9,
                "chat_template": bool(getattr(tokenizer, "chat_template", None)),
            }
            self.loaded.emit(model, tokenizer, info)
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            _progress_hooks.callback = None


# ---------- 背景執行緒：串流生成 ----------
class StopFlagCriteria(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(),
                          dtype=torch.bool, device=input_ids.device)


class GenerationThread(QThread):
    token = pyqtSignal(str)
    done = pyqtSignal(str, dict)
    failed = pyqtSignal(str)

    def __init__(self, model, tokenizer, messages, params):
        super().__init__()
        self.model, self.tokenizer = model, tokenizer
        self.messages, self.params = messages, params
        self.stop_event = threading.Event()
        self.user_stopped = False

    def stop(self):
        self.user_stopped = True
        self.stop_event.set()

    def run(self):
        try:
            tok, model, p = self.tokenizer, self.model, self.params
            device = next(model.parameters()).device
            ctx = (
                getattr(model.config, "max_position_embeddings", None)
                or getattr(model.config, "n_positions", None)
                or getattr(model.config, "n_ctx", None)
                or getattr(model.config, "seq_length", None)
                or 4096
            )
            limit = max(256, ctx - p["max_new_tokens"])

            # 超出上下文長度時，自動刪除最舊的對話
            messages, trimmed = list(self.messages), 0
            while True:
                prompt, is_chat = build_prompt(tok, messages)
                enc = tok(prompt, return_tensors="pt", add_special_tokens=not is_chat)
                n_input = enc["input_ids"].shape[1]
                non_sys = [i for i, m in enumerate(messages) if m["role"] != "system"]
                if n_input <= limit or len(non_sys) <= 1:
                    break
                messages.pop(non_sys[0]); trimmed += 1
                non_sys = [i for i, m in enumerate(messages) if m["role"] != "system"]
                if non_sys and messages[non_sys[0]]["role"] == "assistant":
                    messages.pop(non_sys[0]); trimmed += 1

            inputs = {k: v.to(device) for k, v in enc.items()}
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            max_new = max(1, min(int(p["max_new_tokens"]), max(16, ctx - n_input)))
            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new,
                repetition_penalty=p["repetition_penalty"],
                pad_token_id=tok.pad_token_id,
                stopping_criteria=StoppingCriteriaList([StopFlagCriteria(self.stop_event)]),
            )
            if p["temperature"] > 0:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=p["temperature"],
                    top_p=p["top_p"],
                )
                if int(p.get("top_k", 0)) > 0:
                    gen_kwargs["top_k"] = int(p["top_k"])
            else:
                gen_kwargs.update(do_sample=False)

            err = {}

            def _generate():
                try:
                    model.generate(**gen_kwargs)
                except Exception as e:
                    err["e"] = e
                finally:
                    streamer.end()  # 確保發生錯誤時迴圈也能結束

            start = time.time()
            first_token_at = None
            worker = threading.Thread(target=_generate, daemon=True)
            worker.start()

            text = ""
            for chunk in streamer:
                if not chunk:
                    continue
                if first_token_at is None:
                    first_token_at = time.time()
                text += chunk
                self.token.emit(chunk)
                # Base 模型會自己接著寫 "User:"，偵測到就停止
                if not is_chat and "\nUser:" in text:
                    text = text.split("\nUser:")[0]
                    self.stop_event.set()
                    break
            worker.join()

            if "e" in err:
                raise err["e"]

            elapsed = time.time() - start
            n_tokens = len(tok(text, add_special_tokens=False)["input_ids"])
            self.done.emit(text.strip(), {
                "tokens": n_tokens,
                "seconds": elapsed,
                "tps": n_tokens / elapsed if elapsed > 0 else 0.0,
                "ttft": (first_token_at - start) if first_token_at else None,
                "user_stopped": self.user_stopped,
                "trimmed": trimmed,
            })
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# ---------- 自訂元件 ----------
class PromptEdit(QPlainTextEdit):
    """Enter 送出、Shift+Enter 換行"""
    submitted = pyqtSignal()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
                not (e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.submitted.emit()
            return
        super().keyPressEvent(e)


class ParamSlider(QWidget):
    """滑桿 + 數值框 連動元件"""

    def __init__(self, label, minimum, maximum, default, step, decimals, tooltip=""):
        super().__init__()
        self.decimals, self.default = decimals, default
        self.scale = 10 ** decimals
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setVerticalSpacing(4)

        lbl = QLabel(label)
        lbl.setToolTip(tooltip)
        self.setToolTip(tooltip)

        if decimals:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(decimals)
            self.spin.setRange(minimum, maximum)
            self.spin.setSingleStep(step)
        else:
            self.spin = QSpinBox()
            self.spin.setRange(int(minimum), int(maximum))
            self.spin.setSingleStep(int(step))
        self.spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.spin.setFixedWidth(72)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(minimum * self.scale), int(maximum * self.scale))

        self.slider.valueChanged.connect(
            lambda v: self.spin.setValue(v / self.scale if decimals else v))
        self.spin.valueChanged.connect(
            lambda v: self.slider.setValue(int(round(v * self.scale))))

        lay.addWidget(lbl, 0, 0)
        lay.addWidget(self.spin, 0, 1, alignment=Qt.AlignmentFlag.AlignRight)
        lay.addWidget(self.slider, 1, 0, 1, 2)
        self.setValue(default)

    def value(self):
        return self.spin.value()

    def setValue(self, v):
        try:
            self.spin.setValue(float(v) if self.decimals else int(float(v)))
        except (ValueError, TypeError):
            self.spin.setValue(self.default)

    def reset(self):
        self.setValue(self.default)


# ---------- 主視窗 ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("HFAIStudio", "App")
        self.model = self.tokenizer = None
        self.model_info = {}
        self.loader_thread = self.gen_thread = None
        self.loading = self.generating = False
        self.load_progress_pct = 0
        self.load_progress_text = ""
        self.chat_items = []           # {"role": user/assistant/note, "content": ...}
        self.theme = "dark"
        self._render_dirty = False

        self._build_ui()
        self._connect()
        self.load_settings()
        self.apply_theme()
        self.update_controls()

        self.render_timer = QTimer(self, interval=80, timeout=self._flush_render)
        self.render_timer.start()
        self.res_timer = QTimer(self, interval=1500, timeout=self.update_resources)
        self.res_timer.start()
        self.update_resources()

    # ===== UI 建構 =====
    def _build_ui(self):
        self.setWindowTitle("Hugging Face AI Studio")
        self.resize(1260, 820)
        self.setMinimumSize(980, 640)

        central = QWidget(objectName="central")
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_chat_area())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 920])
        root.addWidget(splitter)

        sb = QStatusBar()
        self.setStatusBar(sb)
        self.status_label = QLabel()
        sb.addWidget(self.status_label, 1)
        self.res_label = QLabel()
        sb.addPermanentWidget(self.res_label)
        self.vram_bar = QProgressBar(objectName="vramBar")
        self.vram_bar.setRange(0, 100)
        self.vram_bar.setTextVisible(False)
        self.vram_bar.setFixedSize(110, 6)
        self.vram_bar.setVisible(torch.cuda.is_available())
        sb.addPermanentWidget(self.vram_bar)

    def _build_sidebar(self):
        sidebar = QFrame(objectName="sidebar")
        sidebar.setMinimumWidth(300)
        sidebar.setMaximumWidth(440)
        outer = QVBoxLayout(sidebar)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget(objectName="sidebarContent")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(6)

        lay.addWidget(QLabel("HF AI Studio", objectName="appTitle"))
        lay.addWidget(QLabel("在本機執行 Hugging Face 語言模型", objectName="appSubtitle"))

        # --- 模型 ---
        g = QGroupBox("模型")
        gl = QVBoxLayout(g)
        gl.setSpacing(8)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(PRESET_MODELS)
        self.model_combo.lineEdit().setPlaceholderText("可直接輸入任意模型 ID，例如 gpt2")
        self.model_combo.setToolTip("從清單選擇，或直接輸入任何 Hugging Face 模型 ID")
        gl.addWidget(QLabel("模型 ID"))
        gl.addWidget(self.model_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItem("自動偵測", "auto")
        if torch.cuda.is_available():
            self.device_combo.addItem(f"CUDA · {torch.cuda.get_device_name(0)}", "cuda")
        if mps_available():
            self.device_combo.addItem("MPS · Apple Silicon", "mps")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.device_combo.setMinimumContentsLength(6)

        self.dtype_combo = QComboBox()
        for label, val in [("自動", "auto"), ("float32", "float32"),
                           ("float16", "float16"), ("bfloat16", "bfloat16")]:
            self.dtype_combo.addItem(label, val)
        self.dtype_combo.setToolTip("自動：CUDA 用 bf16/fp16、MPS 用 fp16、CPU 用 fp32")

        grid = QGridLayout()
        grid.addWidget(QLabel("裝置"), 0, 0)
        grid.addWidget(QLabel("精度"), 0, 1)
        grid.addWidget(self.device_combo, 1, 0)
        grid.addWidget(self.dtype_combo, 1, 1)
        gl.addLayout(grid)

        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("hf_xxx（Llama / Gemma 等受限模型需要）")
        gl.addWidget(QLabel("HF Token（選填）"))
        gl.addWidget(self.token_edit)

        btns = QHBoxLayout()
        self.load_btn = QPushButton("載入模型")
        self.unload_btn = QPushButton("卸載", objectName="secondary")
        btns.addWidget(self.load_btn, 2)
        btns.addWidget(self.unload_btn, 1)
        gl.addLayout(btns)

        self.prog_box = QWidget()
        prog_lay = QVBoxLayout(self.prog_box)
        prog_lay.setContentsMargins(0, 4, 0, 0)
        prog_lay.setSpacing(4)

        header_lay = QHBoxLayout()
        self.prog_desc_lbl = QLabel("準備中…", objectName="progressDesc")
        self.prog_pct_lbl = QLabel("0%", objectName="progressPct")
        header_lay.addWidget(self.prog_desc_lbl, 1)
        header_lay.addWidget(self.prog_pct_lbl, 0)
        prog_lay.addLayout(header_lay)

        self.progress = QProgressBar(objectName="modelProgress")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(8)
        self.progress.setTextVisible(False)
        prog_lay.addWidget(self.progress)

        self.prog_box.hide()
        gl.addWidget(self.prog_box)
        lay.addWidget(g)

        # --- 生成參數 ---
        g2 = QGroupBox("生成參數")
        l2 = QVBoxLayout(g2)
        l2.setSpacing(10)
        self.p_temp = ParamSlider("Temperature", 0.0, 2.0, 0.7, 0.05, 2,
                                  "隨機性：越高越有創意；0 = 貪婪解碼（輸出固定）")
        self.p_top_p = ParamSlider("Top-p", 0.05, 1.0, 0.9, 0.05, 2,
                                   "核採樣：只從累積機率前 p 的詞中挑選")
        self.p_top_k = ParamSlider("Top-k", 0, 200, 50, 1, 0,
                                   "只從機率最高的 k 個詞中挑選（0 = 不限制）")
        self.p_max = ParamSlider("Max new tokens", 16, 4096, 512, 16, 0,
                                 "單次回覆的最大長度")
        self.p_rep = ParamSlider("Repetition penalty", 1.0, 2.0, 1.1, 0.05, 2,
                                 "懲罰重複內容；1.0 = 不懲罰")
        self.param_widgets = {
            "temperature": self.p_temp, "top_p": self.p_top_p, "top_k": self.p_top_k,
            "max_new_tokens": self.p_max, "repetition_penalty": self.p_rep,
        }
        for w in self.param_widgets.values():
            l2.addWidget(w)
        self.reset_btn = QPushButton("恢復預設值", objectName="secondary")
        l2.addWidget(self.reset_btn)
        lay.addWidget(g2)

        # --- 系統提示詞 ---
        g3 = QGroupBox("系統提示詞（System Prompt）")
        l3 = QVBoxLayout(g3)
        self.system_edit = QPlainTextEdit()
        self.system_edit.setPlaceholderText("設定 AI 的角色與行為，例如：你是一位資深 Python 工程師…")
        self.system_edit.setFixedHeight(96)
        l3.addWidget(self.system_edit)
        lay.addWidget(g3)

        # --- 外觀 ---
        g4 = QGroupBox("外觀")
        l4 = QGridLayout(g4)
        self.theme_btn = QPushButton(objectName="secondary")
        self.font_spin = QSpinBox()
        self.font_spin.setRange(11, 22)
        self.font_spin.setValue(14)
        self.font_spin.setSuffix(" px")
        l4.addWidget(self.theme_btn, 0, 0, 1, 2)
        l4.addWidget(QLabel("對話字體大小"), 1, 0)
        l4.addWidget(self.font_spin, 1, 1)
        lay.addWidget(g4)

        lay.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        return sidebar

    def _build_chat_area(self):
        w = QWidget(objectName="central")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(20, 16, 20, 12)
        lay.setSpacing(10)

        header = QHBoxLayout()
        header.addWidget(QLabel("對話", objectName="chatTitle"))
        header.addSpacing(8)
        self.model_badge = QLabel("尚未載入模型", objectName="hint")
        header.addWidget(self.model_badge)
        header.addStretch()
        self.clear_btn = QPushButton("清除對話", objectName="secondary")
        self.export_btn = QPushButton("匯出 Markdown", objectName="secondary")
        header.addWidget(self.clear_btn)
        header.addWidget(self.export_btn)
        lay.addLayout(header)

        self.chat_view = QTextBrowser(objectName="chat")
        self.chat_view.setOpenExternalLinks(True)
        lay.addWidget(self.chat_view, 1)

        row = QHBoxLayout()
        self.prompt_edit = PromptEdit()
        self.prompt_edit.setPlaceholderText("輸入訊息…（Enter 送出，Shift+Enter 換行）")
        self.prompt_edit.setFixedHeight(84)
        col = QVBoxLayout()
        self.send_btn = QPushButton("發送")
        self.stop_btn = QPushButton("停止", objectName="danger")
        for b in (self.send_btn, self.stop_btn):
            b.setFixedWidth(92)
            col.addWidget(b)
        row.addWidget(self.prompt_edit, 1)
        row.addLayout(col)
        lay.addLayout(row)

        lay.addWidget(QLabel("參數調整會在下一則訊息立即生效 · 多輪對話會自動套用模型的 Chat Template",
                             objectName="hint"))
        return w

    def _connect(self):
        self.load_btn.clicked.connect(self.load_model)
        self.unload_btn.clicked.connect(lambda: self.unload_model(silent=False))
        self.reset_btn.clicked.connect(lambda: [w.reset() for w in self.param_widgets.values()])
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.font_spin.valueChanged.connect(self.apply_theme)
        self.clear_btn.clicked.connect(self.clear_chat)
        self.export_btn.clicked.connect(self.export_chat)
        self.send_btn.clicked.connect(self.send_prompt)
        self.prompt_edit.submitted.connect(self.send_prompt)
        self.stop_btn.clicked.connect(self.stop_generation)

    # ===== 主題 =====
    def apply_theme(self):
        p = THEMES[self.theme]
        app = QApplication.instance()
        app.setStyleSheet(QSS_TEMPLATE.substitute(p, chat_font=self.font_spin.value()))

        pal = QPalette()
        for role, key in [(QPalette.ColorRole.Window, "bg"), (QPalette.ColorRole.WindowText, "text"),
                          (QPalette.ColorRole.Base, "input_bg"), (QPalette.ColorRole.Text, "text"),
                          (QPalette.ColorRole.Button, "panel"), (QPalette.ColorRole.ButtonText, "text"),
                          (QPalette.ColorRole.Highlight, "accent"), (QPalette.ColorRole.ToolTipBase, "panel"),
                          (QPalette.ColorRole.ToolTipText, "text"), (QPalette.ColorRole.PlaceholderText, "subtext")]:
            pal.setColor(role, QColor(p[key]))
        app.setPalette(pal)

        self.chat_view.document().setDefaultStyleSheet(doc_css(p))
        self.theme_btn.setText("切換為淺色模式" if self.theme == "dark" else "切換為深色模式")
        self.update_status()
        self.render_chat()

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self.apply_theme()

    # ===== 狀態控制 =====
    def update_controls(self):
        has_model = self.model is not None
        busy = self.loading or self.generating
        for w in (self.model_combo, self.device_combo, self.dtype_combo, self.token_edit, self.load_btn):
            w.setEnabled(not busy)
        self.unload_btn.setEnabled(has_model and not busy)
        self.send_btn.setEnabled(has_model and not busy)
        self.stop_btn.setEnabled(self.generating)
        self.prompt_edit.setEnabled(has_model and not self.loading)
        self.clear_btn.setEnabled(not self.generating)
        self.prog_box.setVisible(self.loading)
        self.load_btn.setText("重新載入" if has_model else "載入模型")
        self.update_status()

    def update_status(self):
        p = THEMES[self.theme]
        if self.loading:
            dot = p["accent"]
            pct_str = f" ({self.load_progress_pct}%)" if self.load_progress_pct > 0 else ""
            desc_str = f" · {self.load_progress_text}" if self.load_progress_text else ""
            txt = f"模型載入中{pct_str}{desc_str}"
        elif self.model is not None:
            i = self.model_info
            dot = p["success"]
            txt = f"{i['model_id']} · {i['params']:.2f}B 參數 · {i['device'].upper()} · {i['dtype']}"
            if self.generating:
                dot, txt = p["accent"], txt + " · 生成中…"
        else:
            dot, txt = p["subtext"], "未載入模型"
        self.status_label.setText(f'<span style="color:{dot}">●</span>&nbsp; {html.escape(txt)}')
        self.model_badge.setText(self.model_info.get("model_id", "尚未載入模型") if self.model else "尚未載入模型")

    def update_resources(self):
        parts = []
        if HAS_PSUTIL:
            rss = psutil.Process().memory_info().rss / 1024 ** 3
            parts.append(f"RAM {rss:.2f} GB（系統 {psutil.virtual_memory().percent:.0f}%）")
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            parts.append(f"VRAM {alloc:.2f} / {total:.1f} GB（保留 {reserved:.2f}）")
            self.vram_bar.setValue(int(reserved / total * 100) if total else 0)
        elif mps_available() and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            parts.append(f"MPS {torch.mps.current_allocated_memory() / 1024 ** 3:.2f} GB")
        self.res_label.setText("   |   ".join(parts))

    # ===== 模型載入 =====
    def load_model(self):
        model_id = self.model_combo.currentText().strip()
        if not model_id:
            QMessageBox.warning(self, "提醒", "請輸入或選擇一個模型 ID")
            return
        if self.model is not None:
            self.unload_model(silent=True)

        device = resolve_device(self.device_combo.currentData())
        self.loading = True
        self.load_progress_pct = 0
        self.load_progress_text = "準備中…"
        self.progress.setValue(0)
        self.prog_pct_lbl.setText("0%")
        self.prog_desc_lbl.setText("準備中…")
        self.prog_desc_lbl.setToolTip("")
        self.update_controls()
        self.add_note(f"正在載入 {model_id} 到 {device.upper()}…（首次執行需下載權重，請稍候）")

        self.loader_thread = ModelLoaderThread(
            model_id, device, self.dtype_combo.currentData(), self.token_edit.text().strip())
        self.loader_thread.progress.connect(self.on_model_progress)
        self.loader_thread.loaded.connect(self.on_model_loaded)
        self.loader_thread.failed.connect(self.on_load_failed)
        self.loader_thread.start()

    def on_model_progress(self, pct, msg):
        self.load_progress_pct = pct
        self.load_progress_text = msg
        self.progress.setValue(pct)
        self.prog_pct_lbl.setText(f"{pct}%")
        disp = msg if len(msg) <= 32 else (msg[:29] + "…")
        self.prog_desc_lbl.setText(disp)
        self.prog_desc_lbl.setToolTip(msg)
        self.update_status()

    def on_model_loaded(self, model, tokenizer, info):
        self.model, self.tokenizer, self.model_info = model, tokenizer, info
        self.loading = False
        self.load_progress_pct = 100
        self.load_progress_text = "載入完成"
        self.update_controls()
        self.add_note(f"✓ 載入完成 · {info['params']:.2f}B 參數 · {info['device'].upper()} · {info['dtype']}")
        if not info["chat_template"]:
            self.add_note("此模型沒有 Chat Template，將使用純文字對話格式（建議改用 Instruct / Chat 版本）")
        self.prompt_edit.setFocus()

    def on_load_failed(self, msg):
        self.loading = False
        self.load_progress_pct = 0
        self.load_progress_text = ""
        self.update_controls()
        self.add_note(f"載入失敗：{msg}", error=True)
        QMessageBox.critical(self, "載入失敗",
                             f"{msg}\n\n提示：Llama、Gemma 等受限模型需要先在 Hugging Face 網站同意授權，"
                             f"並填入 HF Token。")

    def unload_model(self, silent=False):
        self.model = self.tokenizer = None
        self.model_info = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if mps_available() and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if not silent:
            self.add_note("已卸載模型並釋放記憶體")
        self.update_controls()
        self.update_resources()

    # ===== 對話 =====
    def current_params(self):
        return {k: w.value() for k, w in self.param_widgets.items()}

    def send_prompt(self):
        if self.generating or self.model is None:
            return
        text = self.prompt_edit.toPlainText().strip()
        if not text:
            return
        self.prompt_edit.clear()
        self.chat_items.append({"role": "user", "content": text})

        messages = []
        sys_prompt = self.system_edit.toPlainText().strip()
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages += [{"role": i["role"], "content": i["content"]} for i in self.chat_items
                     if i["role"] in ("user", "assistant") and i["content"] and not i.get("empty")]

        short_name = self.model_info["model_id"].split("/")[-1]
        self.chat_items.append({"role": "assistant", "content": "", "streaming": True, "model": short_name})
        self.generating = True
        self.update_controls()
        self.render_chat(force_scroll=True)

        self.gen_thread = GenerationThread(self.model, self.tokenizer, messages, self.current_params())
        self.gen_thread.token.connect(self.on_token)
        self.gen_thread.done.connect(self.on_gen_done)
        self.gen_thread.failed.connect(self.on_gen_failed)
        self.gen_thread.start()

    def stop_generation(self):
        if self.gen_thread and self.gen_thread.isRunning():
            self.gen_thread.stop()
            self.stop_btn.setEnabled(False)

    def on_token(self, chunk):
        if self.chat_items and self.chat_items[-1].get("streaming"):
            self.chat_items[-1]["content"] += chunk
            self._render_dirty = True

    def on_gen_done(self, text, stats):
        if not self.chat_items:
            self.generating = False
            self.update_controls()
            return
        item = self.chat_items[-1]
        item["streaming"] = False
        if text:
            item["content"] = text
        else:
            item["content"], item["empty"] = "_（沒有產生任何內容）_", True

        s = f"{stats['tokens']} tokens · {stats['seconds']:.1f} 秒 · {stats['tps']:.1f} tok/s"
        if stats["ttft"] is not None:
            s += f" · 首字延遲 {stats['ttft']:.2f} 秒"
        if stats["trimmed"]:
            s += f" · 已省略 {stats['trimmed']} 則較舊訊息（超出上下文長度）"
        if stats["user_stopped"]:
            s += " · 已手動停止"
        item["stats"] = s

        self.generating = False
        self.update_controls()
        self.render_chat(force_scroll=True)
        self.prompt_edit.setFocus()

    def on_gen_failed(self, msg):
        if self.chat_items and self.chat_items[-1].get("streaming"):
            item = self.chat_items[-1]
            if item["content"]:
                item["streaming"] = False
            else:
                self.chat_items.pop()
        self.add_note(f"生成錯誤：{msg}", error=True)
        self.generating = False
        self.update_controls()

    def add_note(self, text, error=False):
        self.chat_items.append({"role": "note", "content": text, "error": error})
        self.render_chat(force_scroll=True)

    def clear_chat(self):
        self.chat_items.clear()
        self.render_chat()

    def export_chat(self):
        if not any(i["role"] in ("user", "assistant") for i in self.chat_items):
            QMessageBox.information(self, "匯出", "目前沒有可匯出的對話內容")
            return
        path, _ = QFileDialog.getSaveFileName(self, "匯出對話", "chat.md", "Markdown (*.md);;Text (*.txt)")
        if not path:
            return
        lines = [f"# 對話紀錄 — {self.model_info.get('model_id', '')}\n"]
        sys_prompt = self.system_edit.toPlainText().strip()
        if sys_prompt:
            lines.append(f"> **System:** {sys_prompt}\n")
        for i in self.chat_items:
            if i["role"] == "user":
                lines.append(f"### 你\n\n{i['content']}\n")
            elif i["role"] == "assistant":
                lines.append(f"### AI\n\n{i['content']}\n")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self.statusBar().showMessage(f"已匯出至 {path}", 4000)

    # ===== 渲染 =====
    def _flush_render(self):
        if self._render_dirty:
            self.render_chat()

    def render_chat(self, force_scroll=False):
        self._render_dirty = False
        p = THEMES[self.theme]
        dark = self.theme == "dark"
        sb = self.chat_view.verticalScrollBar()
        old_value = sb.value()
        stick = force_scroll or old_value >= sb.maximum() - 40

        parts = []
        if not self.chat_items:
            parts.append(
                f'<p align="center" style="color:{p["subtext"]};"><br><br><br>'
                f'<span style="font-size:20px; font-weight:600;">HF AI Studio</span><br><br>'
                f'從左側選擇模型並按「載入模型」，即可開始對話<br>'
                f'<span style="font-size:11px;">Enter 送出 · Shift+Enter 換行</span></p>')

        for item in self.chat_items:
            role = item["role"]
            if role == "note":
                color = p["danger"] if item.get("error") else p["subtext"]
                parts.append(f'<p align="center" style="color:{color}; font-size:11px;">'
                             f'{html.escape(item["content"])}</p>')
                continue

            is_user = role == "user"
            bg = p["user_bubble"] if is_user else p["ai_bubble"]
            name = "你" if is_user else f"AI · {item.get('model', '')}"
            name_color = p["subtext"] if is_user else p["accent"]
            content = item["content"]

            if is_user:
                body = html.escape(content).replace("\n", "<br>")
            elif item.get("streaming") and not content:
                body = f'<span style="color:{p["subtext"]};">思考中…</span>'
            elif item.get("streaming"):
                body = md_to_html(content + " ▌", p, dark)
            else:
                body = md_to_html(content, p, dark)

            stats_html = ""
            if item.get("stats"):
                stats_html = (f'<br><span style="color:{p["subtext"]}; font-size:11px;">'
                              f'{html.escape(item["stats"])}</span>')

            parts.append(
                f'<table width="100%" border="0" cellspacing="0" cellpadding="12" bgcolor="{bg}">'
                f'<tr><td><span style="color:{name_color}; font-weight:600;">{html.escape(name)}</span>'
                f'<br>{body}{stats_html}</td></tr></table>'
                f'<p style="font-size:4px;">&nbsp;</p>')

        self.chat_view.setHtml("".join(parts))
        QTimer.singleShot(0, lambda: sb.setValue(sb.maximum() if stick else old_value))

    # ===== 設定保存 =====
    def load_settings(self):
        s = self.settings
        self.theme = s.value("theme", "dark")
        if self.theme not in THEMES:
            self.theme = "dark"
        model_id = s.value("model_id")
        if model_id:
            self.model_combo.setCurrentText(model_id)
        self.system_edit.setPlainText(str(s.value("system_prompt", DEFAULT_SYSTEM)))
        self.font_spin.blockSignals(True)
        try:
            self.font_spin.setValue(int(s.value("chat_font", 14)))
        except (ValueError, TypeError):
            self.font_spin.setValue(14)
        self.font_spin.blockSignals(False)
        for key, w in self.param_widgets.items():
            v = s.value(f"params/{key}")
            if v is not None:
                try:
                    w.setValue(v)
                except (ValueError, TypeError):
                    pass

    def save_settings(self):
        s = self.settings
        s.setValue("theme", self.theme)
        s.setValue("model_id", self.model_combo.currentText().strip())
        s.setValue("system_prompt", self.system_edit.toPlainText())
        s.setValue("chat_font", self.font_spin.value())
        for key, w in self.param_widgets.items():
            s.setValue(f"params/{key}", w.value())

    def closeEvent(self, e):
        self.save_settings()
        if self.gen_thread and self.gen_thread.isRunning():
            self.gen_thread.stop()
            self.gen_thread.wait(5000)
        if self.loader_thread and self.loader_thread.isRunning():
            ans = QMessageBox.question(self, "確認離開", "模型仍在下載/載入中，確定要強制離開嗎？")
            if ans != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            os._exit(0)  # 下載無法中途安全中斷，直接結束程序
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("HF AI Studio")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()