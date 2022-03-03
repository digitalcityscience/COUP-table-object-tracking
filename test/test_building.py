from building import Building, printJSON


def test_printJSON():
    buildingsDict = {19: Building(19, [10, 11], 1111)}
    assert printJSON(buildingsDict) == {19: [10, 11]}
