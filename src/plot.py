import matplotlib.pyplot as plt
from PIL import Image
import io


def figure_to_PIL(figure: plt.Figure) -> Image.Image:
    """Converts a matplotlib figure to a PIL image

    Args:
        figure (plt.Figure): matplotlib Figure

    Returns:
        Image.Image: PNG version of the image
    """
    # Convert plot to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    # need to load because PIL is lazy
    image.load()
    buf.close()
    return image
