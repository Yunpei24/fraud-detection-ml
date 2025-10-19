from src.drift.psi import psi
def test_psi_zero_on_identical():
    a = [0,1,2,3,4,5]
    b = [0,1,2,3,4,5]
    assert psi(a, b) >= 0
