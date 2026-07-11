from species_normalization import normalize_species


def test_common_species_are_canonicalized():
    assert normalize_species("a;b;c;white_tailed_deer") == ("White-tailed Deer", "Deer")
    assert normalize_species("wild_boar") == ("Feral Hog", "Hogs")
    assert normalize_species("corvus species") == ("Other", "Other")
