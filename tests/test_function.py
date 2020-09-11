from monolingual_segmentation import split_over_length
import pytest


@pytest.mark.parametrize('input, output', [
    ("अमेरिका", [('अ', 'मेरिका'), ('अमे', 'रिका'), ('अमेरि', 'का'), ('अमेरिका', '')]),
    ("नेपालको", [('ने', 'पालको'), ('नेपा', 'लको'), ('नेपाल', 'को'), ('नेपालको', '')])
])
def test_split(input, output):
    assert split_over_length(input) == output