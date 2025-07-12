from PIL import Image
from mflux import Flux1, Config

class MLXFlux:
    """Base class for Flux models with common functionality."""
    
    def __init__(self, model_name: str, quantize: int = 8):
        self.flux = Flux1.from_name(model_name=model_name, quantize=quantize)

    def __call__(self, prompt: str, seed: int = 42, **kwargs) -> Image.Image:
        """Generate an image from a text prompt."""
        config = Config(**kwargs)
        
        result = self.flux.generate_image(
            config=config,
            prompt=prompt,
            seed=seed,
        )
        
        return result.image