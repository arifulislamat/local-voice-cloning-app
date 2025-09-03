import random
import numpy as np
import torch
import gradio as gr
import warnings
import os
import logging
import sys
import argparse
from contextlib import contextmanager
from functools import lru_cache
import gc
import threading
import time
import re

# Environment setup - must be done before imports
os.environ.update({
    "TRANSFORMERS_ATTN_IMPLEMENTATION": "eager",
    "TOKENIZERS_PARALLELISM": "false", 
    "TRANSFORMERS_VERBOSITY": "error",
    "DIFFUSERS_VERBOSITY": "error"
})

def setup_warning_filters():
    """Filter out annoying deprecation warnings"""
    warning_patterns = [
        ".*LoRACompatibleLinear.*",
        ".*torch.backends.cuda.sdp_kernel.*", 
        ".*past_key_values.*tuple of tuples.*",
        ".*scaled_dot_product_attention.*output_attentions=True.*"
    ]
    for pattern in warning_patterns:
        warnings.filterwarnings("ignore", message=pattern)

class WarningFilter:
    """Custom stderr filter to hide specific warnings"""
    
    WARNING_KEYWORDS = [
        'LlamaSdpaAttention', 'scaled_dot_product_attention', 'output_attentions=True',
        'manual attention implementation', 'attn_implementation="eager"', 'past_key_values',
        'tuple of tuples', 'deprecated and will be removed', 'FutureWarning', 'LoRACompatibleLinear'
    ]
    
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, message):
        if not any(keyword in message for keyword in self.WARNING_KEYWORDS):
            self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()
    
    def isatty(self):
        return self.original_stderr.isatty()

class GradioOutputCapture:
    """Capture Gradio output while maintaining compatibility with logging systems"""
    
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.public_url = None
        self.buffer = []
        
    def write(self, message):
        # Store message in buffer for processing
        self.buffer.append(message)
        
        # Look for public URL in the message
        if "Running on public URL:" in message:
            url_match = re.search(r'https://[^\s\n]+', message)
            if url_match:
                self.public_url = url_match.group().strip()
                return  # Don't print this message
        
        # Filter out messages we don't want to show
        hide_patterns = [
            "Running on local URL:",
            "This share link expires",
            "For free permanent hosting",
            "gradio deploy"
        ]
        
        if not any(pattern in message for pattern in hide_patterns):
            self.original_stdout.write(message)
    
    def flush(self):
        self.original_stdout.flush()
    
    def isatty(self):
        return self.original_stdout.isatty()
    
    def fileno(self):
        return self.original_stdout.fileno()

@contextmanager
def suppress_warnings():
    """Context manager to suppress warnings and stderr noise"""
    original_stderr = sys.stderr
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sys.stderr = WarningFilter(original_stderr)
            yield
    finally:
        sys.stderr = original_stderr

# Apply warning filters early
setup_warning_filters()

class ColoredFormatter(logging.Formatter):
    """Colored log formatter for better readability"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Color log level
        level_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{level_color}{record.levelname}{self.RESET}"
        
        # Color logger names
        name_colors = {
            '__main__': '\033[1;34m',  # Bold blue
            'root': '\033[1;36m',      # Bold cyan  
            'httpx': '\033[1;35m'      # Bold magenta
        }
        
        for name, color in name_colors.items():
            if name in record.name:
                record.name = f"{color}{record.name}\033[0m"
                break
                
        return super().format(record)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:\033[1m%(name)s\033[0m:%(message)s',
    handlers=[logging.StreamHandler()]
)

for handler in logging.getLogger().handlers:
    handler.setFormatter(ColoredFormatter('%(levelname)s:\033[1m%(name)s\033[0m:%(message)s'))

logger = logging.getLogger(__name__)

# Fix for perth watermarker
try:
    import perth
    if perth.PerthImplicitWatermarker is None:
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
except ImportError:
    logger.warning("Perth watermarker not available")

from chatterbox.tts import ChatterboxTTS

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TEXT_LENGTH = 450
DEFAULT_TEXT = """Now let's explore the beauty of Bangladesh. First, we admire the vibrant streets of Dhaka,
the capital, where the buzz of life never stops. Then, we immerse ourselves in the calmness of the Sundarbans,
the largest mangrove forest in the world, home to the majestic Bengal tiger. A sprinkle of the rich cultural heritage,
with centuries-old temples and bustling markets. Now feel that energy. Oh, the magic of Bangladesh is truly extraordinary."""

logger.info(f"Using device: {DEVICE}")

def set_seed(seed: int) -> int:
    """Set random seeds for reproducible generation"""
    if seed <= 0:
        seed = random.randint(1, 2**32 - 1)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    return seed

@lru_cache(maxsize=1)
def get_model():
    """Load and cache the TTS model"""
    try:
        logger.info("Loading ChatterboxTTS model...")
        with suppress_warnings():
            model = ChatterboxTTS.from_pretrained(DEVICE)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def validate_inputs(text, exaggeration, temperature, cfgw, min_p, top_p, repetition_penalty):
    """Validate user inputs"""
    if not text or len(text.strip()) == 0:
        raise gr.Error("Please provide text to synthesize")
    
    if len(text) > MAX_TEXT_LENGTH:
        raise gr.Error(f"❌ Text is too long! You entered {len(text)} characters, but the maximum allowed is {MAX_TEXT_LENGTH} characters. Please shorten your text by {len(text) - MAX_TEXT_LENGTH} characters.")
    
    # Parameter validation with ranges
    validations = [
        (exaggeration, 0.25, 2, "Exaggeration"),
        (temperature, 0.05, 5, "Temperature"), 
        (cfgw, 0, 1, "CFG/Pace"),
        (min_p, 0, 1, "min_p"),
        (top_p, 0, 1, "top_p"),
        (repetition_penalty, 1, 2, "Repetition penalty")
    ]
    
    for value, min_val, max_val, name in validations:
        if not min_val <= value <= max_val:
            raise gr.Error(f"{name} must be between {min_val} and {max_val}")

def generate_audio(text, audio_prompt_path, exaggeration, temperature, seed_num, 
                  cfgw, min_p, top_p, repetition_penalty, progress=gr.Progress()):
    """Generate audio from text with specified parameters"""
    try:
        progress(0.1, desc="Validating inputs...")
        validate_inputs(text, exaggeration, temperature, cfgw, min_p, top_p, repetition_penalty)
        
        text = text.strip()
        
        # Warn about long generation times
        if len(text) > 1000:
            logger.info(f"Generating long audio: {len(text)} chars (~{len(text)//10}s estimated)")
        
        progress(0.2, desc="Loading model...")
        model = get_model()
        
        progress(0.3, desc="Setting seed...")
        actual_seed = set_seed(int(seed_num) if seed_num != 0 else 0)
        logger.info(f"Using seed: {actual_seed}")
        
        progress_msg = "Generating long audio (this may take a while)..." if len(text) > 1500 else "Generating audio..."
        progress(0.4, desc=progress_msg)
        
        logger.info(f"Generating with params: exaggeration={exaggeration}, temp={temperature}, "
                   f"cfg={cfgw}, min_p={min_p}, top_p={top_p}, rep_penalty={repetition_penalty}")
        
        # Generate audio
        with suppress_warnings():
            wav = model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                temperature=temperature,
                cfg_weight=cfgw,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        
        progress(0.9, desc="Processing output...")
        
        # Convert to numpy for Gradio
        audio_array = wav.squeeze(0).cpu().numpy()
        
        # Cleanup GPU memory
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        progress(1.0, desc="Complete!")
        
        duration_estimate = len(text) / 10
        logger.info(f"Generated audio for {len(text)} chars (~{duration_estimate:.1f}s estimated)")
        
        return (model.sr, audio_array)
        
    except gr.Error:
        raise
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise gr.Error(f"Audio generation failed: {str(e)}")

def create_interface():
    """Build the Gradio web interface"""
    
    custom_css = """
    .container {
        max-width: 1200px;
        margin: auto;
    }
    .tips-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        width: 100%;
        box-sizing: border-box;
    }
    .dark .tips-container {
        background-color: #2b2b2b;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Local Voice Cloning App") as app:
        # Page header
        gr.Markdown("# 🎙️ Local Voice Cloning App\nGenerate high-quality speech from text with custom voice references.")
        
        with gr.Row():
            # Left column - inputs
            with gr.Column(scale=1):
                # Input settings header with character count
                input_header = gr.Markdown(f"### Input Settings")
                
                text = gr.Textbox(
                    value=DEFAULT_TEXT,
                    label=f"Characters: {len(DEFAULT_TEXT)}/{MAX_TEXT_LENGTH} | Estimated duration: ~{len(DEFAULT_TEXT)//10} seconds",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=6,
                    max_lines=20,
                    max_length=MAX_TEXT_LENGTH
                )
                
                def update_char_count(text_input):
                    count = len(text_input) if text_input else 0
                    duration = count // 10
                    
                    # Color coding based on character count
                    if count > MAX_TEXT_LENGTH:
                        color = "🔴"
                        status = f"({count - MAX_TEXT_LENGTH} over limit!)"
                    elif count > MAX_TEXT_LENGTH * 0.9:  # 90% of limit
                        color = "🟡"
                        status = f"({MAX_TEXT_LENGTH - count} remaining)"
                    else:
                        color = "🟢"
                        status = ""
                    
                    label = f"{color} Characters: {count}/{MAX_TEXT_LENGTH} | Estimated duration: ~{duration} seconds {status}"
                    return gr.update(label=label)
                
                text.change(update_char_count, inputs=[text], outputs=[ text])
                
                # Audio upload
                ref_wav = gr.Audio(
                    sources=["upload"], 
                    type="filepath", 
                    label="Reference Audio (Optional)",
                    value=None
                )
            
            # Right column - output and controls
            with gr.Column(scale=1):
                gr.Markdown("### Generated Audio")
                audio_output = gr.Audio(label="Output Audio", type="numpy", autoplay=False)
                
                gr.Markdown("### Generation Parameters")
                
                with gr.Row():
                    exaggeration = gr.Slider(0.25, 2, step=0.05, label="Exaggeration", value=0.5,
                                           info="0.5 = Neutral, extreme values may be unstable")
                    cfg_weight = gr.Slider(0.0, 1, step=0.05, label="CFG/Pace", value=0.5,
                                         info="Controls guidance strength")
                
                with gr.Accordion("🔧 Advanced Options", open=False):
                    with gr.Row():
                        seed_num = gr.Number(value=0, label="Random Seed", info="0 for random seed", precision=0)
                        temp = gr.Slider(0.05, 5, step=0.05, label="Temperature", value=0.8, info="Controls randomness")
                    
                    with gr.Row():
                        min_p = gr.Slider(0.00, 1.00, step=0.01, label="min_p", value=0.05,
                                        info="Newer sampler, 0.02-0.1 recommended, 0 disables")
                        top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p", value=1.00,
                                        info="Original sampler, 1.0 disables (recommended)")
                    
                    repetition_penalty = gr.Slider(1.00, 2.00, step=0.1, label="Repetition Penalty", 
                                                 value=1.2, info="Reduces repetition in output")
                
                with gr.Row():
                    run_btn = gr.Button("🎵 Generate Audio", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)
        
        # Tips section
        gr.HTML("""
        <div class="tips-container">
            <strong>Tips:</strong>
            <ul style="margin-top: 10px;">
                <li>📝 <strong>Text limit: 450 characters maximum</strong> for optimal quality</li>
                <li>Upload a reference audio for voice cloning</li>
                <li>Adjust exaggeration for emotional expression</li>
                <li>Lower temperature for more consistent output</li>
                <li>Estimated: ~10 characters per second of audio</li>
            </ul>
        </div>
        """)
        
        # Examples
        gr.Examples(
            examples=[
                ["Hello, this is a test of the ChatterboxTTS system.", None, 0.5, 0.8, 0, 0.5, 0.05, 1.0, 1.2],
                ["The quick brown fox jumps over the lazy dog.", None, 0.7, 0.6, 42, 0.4, 0.08, 1.0, 1.1],
            ],
            inputs=[text, ref_wav, exaggeration, temp, seed_num, cfg_weight, min_p, top_p, repetition_penalty],
            outputs=audio_output,
            fn=generate_audio,
            cache_examples=False,
        )
        
        # Event handlers
        run_btn.click(
            fn=generate_audio,
            inputs=[text, ref_wav, exaggeration, temp, seed_num, cfg_weight, min_p, top_p, repetition_penalty],
            outputs=audio_output,
            show_progress="full",
        )
        
        def clear_all():
            return "", None, 0.5, 0.8, 0, 0.5, 0.05, 1.0, 1.2, None
        
        clear_btn.click(
            fn=clear_all,
            outputs=[text, ref_wav, exaggeration, temp, seed_num, cfg_weight, min_p, top_p, repetition_penalty, audio_output],
        )
        
        # Footer
        gr.Markdown("""
        ---
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span>Powered by ChatterboxTTS | Transformer | Llama | Gradio</span>
            <span><a href="https://arifulislamat.com" target="_blank" style="color: #ff6b35; text-decoration: none;">Ariful Islam</a></span>
        </div>
        """)
    
    return app

def print_banner(args, local_url=None, public_url=None):
    """Print startup banner with URLs"""
    print("\n" + "="*70)
    print("🎙️  \033[1;36mLocal Voice Cloning App\033[0m")
    print("="*70)
    
    if args.public:
        print("🌐 \033[1;32mStarting with PUBLIC sharing enabled...\033[0m")
        local_url = local_url or f"http://localhost:{args.port}"
        print(f"🏠 \033[1;33mLocal URL:\033[0m \033[1;34m{local_url}\033[0m")
        
        if public_url:
            print(f"🌍 \033[1;32mPublic URL:\033[0m \033[1;35m{public_url}\033[0m")
            print("📝 \033[1;36mShare link expires in 1 week\033[0m")
        else:
            print("🔄 \033[1;33mGenerating public URL...\033[0m")
    else:
        local_url = local_url or f"http://localhost:{args.port}"
        print(f"🏠 \033[1;33mStarting locally:\033[0m \033[1;34m{local_url}\033[0m")
    
    print("👤 \033[1;35mAriful Islam\033[0m")
    print("🔗 \033[4;36mhttps://arifulislamat.com\033[0m")
    print("="*70)
    print("💡 \033[1;37mTip: Use --public flag for public sharing\033[0m")
    print("="*70 + "\n")

def monitor_gradio_startup(capture_stdout):
    """Monitor gradio's startup output for public URL"""
    max_wait = 30  # seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        if capture_stdout.public_url:
            # Found public URL, print it nicely
            print(f"\r🌍 \033[1;32mPublic URL:\033[0m \033[1;35m{capture_stdout.public_url}\033[0m")
            print("📝 \033[1;36mShare link expires in 1 week\033[0m")
            print("\n🚀 \033[1;32mApp is ready!\033[0m Press Ctrl+C to stop.\n")
            return capture_stdout.public_url
        time.sleep(0.5)
    
    print("\r⚠️  \033[1;33mPublic URL generation timed out\033[0m")
    return None

def main():
    """Run the application"""
    parser = argparse.ArgumentParser(description="Local Voice Cloning App")
    parser.add_argument("--public", action="store_true", help="Enable public Gradio sharing")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on (default: 7860)")
    args = parser.parse_args()
    
    try:
        # Pre-load model
        logger.info("Pre-loading model...")
        with suppress_warnings():
            get_model()
        
        # Build interface
        app = create_interface()
        
        # Configure server
        server_name = "0.0.0.0" if args.public else "localhost"
        local_url = f"http://localhost:{args.port}"
        
        # Print initial banner
        print_banner(args, local_url)
        
        # Configure queue
        app.queue(
            max_size=20,
            default_concurrency_limit=1,
            api_open=False,
        )
        
        # Handle different launch modes
        if args.public:
            logger.info("Launching with public sharing...")
            
            # Capture stdout to monitor for public URL
            original_stdout = sys.stdout
            capture_stdout = GradioOutputCapture(original_stdout)
            
            try:
                sys.stdout = capture_stdout
                
                # Start monitoring in background thread
                monitor_thread = threading.Thread(target=monitor_gradio_startup, args=(capture_stdout,), daemon=True)
                monitor_thread.start()
                
                # Launch gradio (this will block)
                app.launch(
                    share=True,
                    server_name=server_name,
                    server_port=args.port,
                    show_error=True,
                    quiet=False,  # Let some output through for debugging
                )
                
            finally:
                sys.stdout = original_stdout
        else:
            # Local only - simpler launch
            print(f"🚀 \033[1;32mStarting local server...\033[0m")
            
            # Use threading to print ready message after server starts
            def print_ready_message():
                time.sleep(2)  # Wait for server to fully start
                print("\n🚀 \033[1;32mApp is ready!\033[0m Press Ctrl+C to stop.\n")
            
            ready_thread = threading.Thread(target=print_ready_message, daemon=True)
            ready_thread.start()
            
            app.launch(
                share=False,
                server_name=server_name,
                server_port=args.port,
                show_error=True,
                quiet=True,
            )
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    main()