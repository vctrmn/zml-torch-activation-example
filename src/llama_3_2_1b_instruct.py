import torch
from transformers import pipeline

from .utils.logging import get_logger
from .utils.zml import ActivationCollector


logger = get_logger(__name__)


MODEL_NAME: str = "meta-llama/Llama-3.2-1B-Instruct"

# CPU Instruction Set and dtype Support:
#
# 1. SSE2 (Early 2000s, Intel Pentium 4, AMD Athlon 64):
#    - Supports: float32, float64
#
# 2. AVX (2011, Intel Sandy Bridge, AMD Bulldozer):
#    - Supports: float32, float64
#
# 3. AVX2 (2013, Intel Haswell, AMD Excavator):
#    - Supports: float32, float64, int8
#    - AVX2 is optimized for both integer and floating-point operations,
#      but it does NOT support bfloat16.
#
# 4. AVX512 (2017, Intel Skylake-X, Xeon processors):
#    - Supports: float32, float64, int8, bfloat16 (with AVX512 BF16 instructions)
#    - Provides significant parallel computing performance gains.
#
# 5. AVX512 BF16 (2020, Intel Cooper Lake, Ice Lake):
#    - Supports: bfloat16 (for mixed-precision AI workloads)
#    - Enables faster training and inference for AI models.


def main() -> None:
    try:
        logger.info("Start running main()")
        logger.info(f"Is CUDA available ? `{torch.cuda.is_available()}`")
        logger.info(f"CPU capability : `{torch.backends.cpu.get_cpu_capability()}`")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model : `{MODEL_NAME}`")
        text_generation_pipeline = pipeline(
            "text-generation",
            model=MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        model, tokenizer = text_generation_pipeline.model, text_generation_pipeline.tokenizer
        logger.info(
            f"Model loaded successfully {model.config.architectures} - `{model.config.torch_dtype}` - {tokenizer.model_max_length} max tokens"  # noqa: E501
        )

        # Wrap the pipeline, and extract activations.
        # Activations files can be huge for big models,
        # so let's stop collecting after 1000 layers.
        zml_pipeline = ActivationCollector(text_generation_pipeline, max_layers=1000, stop_after_first_step=True)

        messages = [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": "Who are you?"},
        ]
        outputs, activations = zml_pipeline(messages, max_new_tokens=256)
        logger.info(f"ouputs : {outputs}")

        filename = MODEL_NAME.split("/")[-1] + ".activations.pt"
        torch.save(activations, filename)
        logger.info(f"Saved {len(activations)} activations to {filename}")

        logger.info("End running main()")
    except Exception as exception:
        logger.error(exception)
        raise


if __name__ == "__main__":
    main()
