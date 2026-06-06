"""Vendored non-linear diverging colormaps (white plateau near zero).
Copied from the user's myCmap.py (the self-contained cdict variants only)."""
import matplotlib.colors as mc
from matplotlib.colors import LinearSegmentedColormap

cdictDivergeNL8 = {
    'red': ((0.0, 0.192, 0.192), (0.2, 0.270, 0.270), (0.3, 0.455, 0.455), (0.4, .67, .67),
            (0.45, .77, .77), (0.5, 1., 1.), (0.525, 1., 1.), (0.55, 1., 1.), (0.6, .992, .992),
            (0.65, .95, .95), (0.7, 0.9, 0.9), (0.8, 0.843, 0.843), (1.0, 0.647, 0.647)),
    'green': ((0.0, 0.211, 0.211), (0.2, 0.459, 0.459), (0.3, 0.678, 0.678), (0.4, .751, .751),
              (0.45, .851, .851), (0.5, 1., 1.), (0.525, 1., 1.), (0.55, 1., 1.), (0.6, .7, .7),
              (0.65, .682, .682), (0.7, 0.427, 0.427), (0.8, 0.188, 0.188), (1.0, 0, 0)),
    'blue': ((0.0, 0.584, 0.584), (0.2, 0.706, 0.706), (0.3, 0.80, 0.80), (0.4, .85, .85),
             (0.45, .914, .914), (0.5, 1., 1.), (0.525, 1., 1.), (0.55, 1., 1.), (0.6, .480, .480),
             (0.65, .35, .35), (0.7, 0.3, 0.3), (0.8, 0.253, 0.253), (1.0, 0.2, 0.2)),
}
joy_nl8 = LinearSegmentedColormap("JoyNL8", cdictDivergeNL8)

def bokeh_palette(cmap, n=256):
    """Sample an mpl Colormap to an n-color hex list (preserves the NL white plateau)."""
    return [mc.rgb2hex(cmap(i/(n-1))) for i in range(n)]
