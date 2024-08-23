import pytest
from pytest import approx
import subprocess

import igrf

time = "2010-07-12"


def test_bad_date():
    with pytest.raises(RuntimeError):
        igrf.igrf("2500-01-01", 65, 80, 0)


def test_igrf13():

    mag = igrf.igrf(time, 65, 85, 0)
    assert mag.north.item() == approx(9295.415460)
    assert mag.east.item() == approx(2559.7889298)
    assert mag.down.item() == approx(59670.379598)
    assert mag.total.item() == approx(60444.284008)

    assert mag.incl.item() == approx(80.821575)
    assert mag.decl.item() == approx(15.396590)


# def test_cli():
#     subprocess.check_call(["igrf", time, "-c", "65", "85", "-a", "0"])


# def test_igrf12():

#     mag = igrf.igrf(time, 65, 85, 0, model=12)

#     assert mag.north.item() == approx(9295.100256)
#     assert mag.east.item() == approx(2560.199706)
#     assert mag.down.item() == approx(59670.251893)
#     assert mag.total.item() == approx(60444.126863)

#     assert mag.incl.item() == approx(80.821738)
#     assert mag.decl.item() == approx(15.399442)


# def test_igrf11():

#     mag = igrf.igrf(time, 65, 85, 0, model=11)

#     assert mag.north.item() == approx(9301.523160)
#     assert mag.east.item() == approx(2563.450424)
#     assert mag.down.item() == approx(59666.132881)
#     assert mag.total.item() == approx(60441.186489)

#     assert mag.incl.item() == approx(80.814513)
#     assert mag.decl.item() == approx(15.407924)
