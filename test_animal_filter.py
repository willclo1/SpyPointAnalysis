from animal_filter import _intersection_ratio, _matches


def test_keyword_matching_uses_word_boundaries():
    assert _matches("golf cart", {"golf cart"})
    assert _matches("pickup truck", {"truck"})
    assert not _matches("cardinal", {"car"})


def test_gate_intersection_is_relative_to_object():
    assert _intersection_ratio((0.5, 0.5, 0.7, 0.7), (0.35, 0.25, 0.98, 0.95)) == 1.0
    assert _intersection_ratio((0.0, 0.0, 0.1, 0.1), (0.35, 0.25, 0.98, 0.95)) == 0.0
