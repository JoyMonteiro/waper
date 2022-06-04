def test_smoke_is_not_risingfrom_module():
    import my_new_project

    assert my_new_project is not None
