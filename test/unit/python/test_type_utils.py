import pytest

from python.type_utils import strtobool


@pytest.mark.parametrize("val", ["y", "yes", "t", "true", "on", "1", "True", "  YES  "])
def test_strtobool_true_values(val):
    assert strtobool(val) == 1


@pytest.mark.parametrize("val", ["n", "no", "f", "false", "off", "0", "False", " Off "])
def test_strtobool_false_values(val):
    assert strtobool(val) == 0


@pytest.mark.parametrize("val", ["", "maybe", "2", "yeah", "nope"])
def test_strtobool_invalid_raises(val):
    with pytest.raises(ValueError):
        strtobool(val)


def test_strtobool_accepts_non_str():
    # Mirrors usage with env/payload values that may already be bool/int.
    assert strtobool(True) == 1
    assert strtobool(False) == 0


if __name__ == "__main__":
    pytest.main()
