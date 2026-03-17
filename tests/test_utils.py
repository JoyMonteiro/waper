from waper.identification.utils import is_to_the_east, _longitude_separation

def test_east_simple():
    assert is_to_the_east(10, 5) is True

def test_west_simple():
    assert is_to_the_east(5, 10) is False

def test_same_longitude():
    assert is_to_the_east(10, 10) is False

def test_wraparound_east():
    assert is_to_the_east(5, 355) is True   # 5° is 10° east of 355°

def test_wraparound_west():
    assert is_to_the_east(355, 5) is False

def test_lon_separation_normal():
    assert _longitude_separation(10, 20) == 10

def test_lon_separation_wraparound():
    assert _longitude_separation(358, 2) == 4

def test_lon_separation_symmetric():
    assert _longitude_separation(2, 358) == 4
