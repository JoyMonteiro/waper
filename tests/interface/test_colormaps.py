from waper.interface.colormaps import joy_nl8, bokeh_palette

def test_nl_palette_white_plateau():
    pal = bokeh_palette(joy_nl8, n=256)
    assert len(pal) == 256
    white = [i for i, c in enumerate(pal) if c.lower() in ("#ffffff", "#fefefe")]
    # NL8 white plateau sits just above centre (verified: data-fraction ~0.50–0.55)
    assert white, "expected a white plateau"
    assert 0.45 <= white[0] / 255 <= 0.55
