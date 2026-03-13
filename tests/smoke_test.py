def test_smoke_import():
    import waper

    assert waper is not None


def test_smoke_classes_exist():
    from waper import Waper, WaperConfig

    assert Waper is not None
    assert WaperConfig is not None
