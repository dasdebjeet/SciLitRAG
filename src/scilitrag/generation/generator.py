
"""
Local Hugging Face text generation for SciLitRAG.

Loads Qwen3-14B using 4-bit NF4 quantization and generates responses
directly through PyTorch and Transformers.

GPU memory management:
    - Force all model layers onto NVIDIA GPU 0 (RTX 5080).
    - Do not permit automatic CPU or disk layer offloading.
    - Keep the model loaded when reusing QwenGenerator in a persistent backend.
    - Call close() when the generator is no longer needed.
    - Avoid keeping embedding and reranking models resident on the GPU
      simultaneously unless sufficient VRAM is available.

Performance monitoring:
    - Measure tokenizer initialization time.
    - Measure model loading time.
    - Log actual model device placement.
    - Measure generation latency and tokens per second.
    - Report GPU memory allocation and reservation.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResult:
    """
    Store generated text and measured language-model inference statistics.

    Attributes:
        text: Decoded response produced by the language model.
        generation_duration_seconds: Time spent inside model.generate(),
            measured after CUDA synchronization.
        generation_token_count: Number of newly generated token IDs,
            excluding the input prompt.
        tokens_per_second: Generated-token throughput calculated from
            generation duration.
    """

    text: str
    generation_duration_seconds: float
    generation_token_count: int
    tokens_per_second: float    


class QwenGenerator:
    """Local Qwen3-14B generator with GPU-only 4-bit quantization."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-14B",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.tokenizer = None
        self.model = None

        self._validate_environment()
        self._load_model()

    @staticmethod
    def _validate_environment() -> None:
        """Fail early if CUDA or the required GPU is unavailable."""

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Check your PyTorch and NVIDIA setup."
            )

        gpu_name = torch.cuda.get_device_name(0)
        total_memory = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )

        logger.info("GPU detected: %s", gpu_name)
        logger.info("Total GPU memory: %.2f GB", total_memory)

    @staticmethod
    def _log_gpu_memory(stage: str) -> None:
        """Log current GPU memory allocation and reservation."""

        if not torch.cuda.is_available():
            return

        torch.cuda.synchronize(0)

        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free, total = torch.cuda.mem_get_info(0)

        logger.info(
            "[%s] GPU allocated: %.2f GB | reserved: %.2f GB "
            "| free: %.2f GB | total: %.2f GB",
            stage,
            allocated,
            reserved,
            free / 1024**3,
            total / 1024**3,
        )

    def _log_device_map(self) -> None:
        """
        Report model placement and detect unexpected CPU/disk offloading.

        With device_map={"": 0}, every model component should be assigned
        to GPU 0. This check makes the actual placement visible in logs.
        """

        device_map = getattr(self.model, "hf_device_map", None)

        if not device_map:
            logger.warning("Model device map is unavailable.")
            return

        devices = set(device_map.values())

        logger.info("Model device map: %s", device_map)

        cpu_components = [
            name
            for name, device in device_map.items()
            if str(device).lower() in {"cpu", "disk"}
        ]

        if cpu_components:
            logger.error(
                "Unexpected CPU/disk offloading detected: %s",
                cpu_components,
            )
        else:
            logger.info(
                "GPU placement verified. Assigned device(s): %s",
                devices,
            )

    def _load_model(self) -> None:
        """Load the tokenizer and quantized model directly onto GPU 0."""

        # -------------------------------------------------------------
        # 1. Load tokenizer
        # -------------------------------------------------------------
        logger.info("Loading tokenizer: %s", self.model_name)

        tokenizer_start = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )

        tokenizer_duration = time.perf_counter() - tokenizer_start

        logger.info(
            "Tokenizer loaded in %.2f seconds.",
            tokenizer_duration,
        )

        # -------------------------------------------------------------
        # 2. Configure 4-bit NF4 quantization
        # -------------------------------------------------------------
        # NF4 is designed for quantization of normally distributed
        # model weights. Double quantization reduces metadata overhead.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        logger.info(
            "[MODEL] Loading generator | model=%s "
            "| component=QwenGenerator | device=cuda:0 "
            "| quantization=4-bit-NF4",
            self.model_name,
        )

        self._log_gpu_memory("Before model loading")

        # -------------------------------------------------------------
        # 3. Load the model exclusively onto the RTX 5080
        # -------------------------------------------------------------
        model_start = time.perf_counter()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,

            quantization_config=quantization_config,

            # Force the entire model onto NVIDIA GPU 0.
            # Unlike "auto", this does not automatically offload
            # model layers to system RAM or disk.
            device_map={"": 0},

            # Use BF16 for computations where supported.
            dtype=torch.bfloat16,

            # Reduce CPU memory overhead during checkpoint loading.
            low_cpu_mem_usage=True,

            # Prefer safetensors when available.
            use_safetensors=True,
        )

        # Wait for asynchronous CUDA operations before measuring time.
        torch.cuda.synchronize(0)

        model_duration = time.perf_counter() - model_start

        logger.info(
            "[MODEL] Generator loaded | model=%s | device=cuda:0 "
            "| quantization=4-bit-NF4 | duration=%.2fs",
            self.model_name,
            model_duration,
        )

        model_duration = time.perf_counter() - model_start

        self.model.eval()

        logger.info(
            "Qwen3 model loaded successfully in %.2f seconds.",
            model_duration,
        )

        # -------------------------------------------------------------
        # 4. Verify placement and GPU memory consumption
        # -------------------------------------------------------------
        self._log_device_map()
        self._log_gpu_memory("After model loading")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate a response to a user prompt.

        Args:
            prompt: User's input text.
            system_prompt: Optional system-level instructions.
            max_new_tokens: Maximum number of generated tokens.

        Returns:
            GenerationResult containing the generated text, measured
            generation duration, generated-token count, and throughput.
        """

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Generator is not loaded.")

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if system_prompt is None:
            system_prompt = (
                "You are a scientific research assistant. "
                "Provide clear, accurate and technically grounded answers."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Use Qwen3's native chat template and disable thinking mode.
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        # Since the complete model is assigned to GPU 0, place the
        # input tensors on that same device.
        model_inputs = model_inputs.to(torch.device("cuda:0"))

        prompt_length = model_inputs["input_ids"].shape[1]

        logger.info("Input prompt tokens: %d", prompt_length)

        generation_limit = (
            max_new_tokens
            if max_new_tokens is not None
            else self.max_new_tokens
        )

        # Synchronize before timing to exclude previous asynchronous work.
        torch.cuda.synchronize(0)
        generation_start = time.perf_counter()

        output_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=generation_limit,
            do_sample=False,
            repetition_penalty=1.05,
            use_cache=True,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # CUDA operations are asynchronous, so synchronize before
        # recording the actual generation duration.
        torch.cuda.synchronize(0)

        generation_duration = time.perf_counter() - generation_start

        # Remove the input tokens; decode only newly generated tokens.
        generated_ids = output_ids[0][prompt_length:]

        generated_token_count = int(generated_ids.numel())

        tokens_per_second = (
            generated_token_count / generation_duration
            if generation_duration > 0
            else 0.0
        )

        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        logger.info(
            "Generation completed in %.2f seconds.",
            generation_duration,
        )
        logger.info(
            "Generated tokens: %d | Speed: %.2f tokens/second",
            generated_token_count,
            tokens_per_second,
        )

        self._log_gpu_memory("After generation")

        return GenerationResult(
            text=response.strip(),
            generation_duration_seconds=generation_duration,
            generation_token_count=int(generated_token_count),
            tokens_per_second=float(tokens_per_second),
        )

    def close(self) -> None:
        """Release model references and clear unused CUDA memory."""

        logger.info("Releasing Qwen model from memory...")

        if torch.cuda.is_available():
            torch.cuda.synchronize(0)

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Generator resources released.")


def main() -> None:
    """Standalone smoke test for local Qwen generation."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    generator = None

    try:
        # Load the model once for this process.
        generator = QwenGenerator(max_new_tokens=256)

        generation_result = generator.generate(
            prompt=(
                "Explain what Retrieval-Augmented Generation (RAG) is "
                "in approximately 150 words. Include its main components."
            )
        )

        print("\n" + "=" * 70)
        print("QWEN3 GENERATION RESULT")
        print("=" * 70)
        print(generation_result.text)
        print("=" * 70)

        logger.info(
            "Generation statistics | duration=%.2fs | tokens=%d | speed=%.2f tokens/s",
            generation_result.generation_duration_seconds,
            generation_result.generation_token_count,
            generation_result.tokens_per_second,
        )

    finally:
        # Release GPU resources when the standalone test exits.
        if generator is not None:
            generator.close()


if __name__ == "__main__":
    main()
