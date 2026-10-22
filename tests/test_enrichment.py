from cnv_overlap.enrichment import ora


def test_ora_empty_and_background():
    terms = {"t1": ("one", {"a", "b", "x"}), "t2": ("two", {"c", "d"})}
    result = ora(["a", "x"], ["a", "b", "c", "d"], terms, min_size=1)
    assert (result.query_size == 1).all() and (result.background_size == 4).all()
    assert (ora([], ["a", "b"], terms, min_size=1).q == 1).all()
