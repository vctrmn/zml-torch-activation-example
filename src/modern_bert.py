import torch
from transformers import pipeline

from .utils.logging import get_logger
from .utils.zml import ActivationCollector


logger = get_logger(__name__)


MODEL_NAME: str = "answerdotai/ModernBERT-base"


def main() -> None:
    try:
        logger.info("Start running main()")
        logger.info(f"CPU capability : `{torch.backends.cpu.get_cpu_capability()}`")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model : `{MODEL_NAME}`")

        fill_mask_pipeline = pipeline(
            "fill-mask",
            model=MODEL_NAME,
            device_map=device,
        )
        model, tokenizer = fill_mask_pipeline.model, fill_mask_pipeline.tokenizer
        logger.info(
            f"Model loaded successfully {model.config.architectures} - `{model.config.torch_dtype}` - {tokenizer.model_max_length} max tokens"  # noqa: E501
        )

        # input_text = "Paris is the [MASK] of France."
        # outputs = fill_mask_pipeline(input_text)
        # logger.info(f"ouputs : {outputs}")

        # Wrap the pipeline, and extract activations.
        # Activations files can be huge for big models,
        # so let's stop collecting after 1000 layers.
        zml_pipeline = ActivationCollector(fill_mask_pipeline, max_layers=1000, stop_after_first_step=True)

        input_text = "Paris is the [MASK] of France."
        outputs, activations = zml_pipeline(input_text)
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
