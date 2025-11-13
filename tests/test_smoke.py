
def test_imports():
    import source.model as m
    import source.data_loader as dl
    assert hasattr(m, 'build_simple_cnn')
    assert hasattr(dl, 'create_generators')
