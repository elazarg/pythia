import pprint
from difflib import ndiff

import pytest

#
# @pytest.hookimpl(tryfirst=True)
# def pytest_assertrepr_compare(config, op, left, right):
#     if op in ("==", "!="):
#         return [x for x in ndiff(left, right) if not x.startswith(" ")]
