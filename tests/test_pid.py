import pytest

from cadet_live.control.pid import PIDController


def test_back_calculation_prevents_windup_at_upper_limit() -> None:
    controller = PIDController(kp=0, ki=1, kd=0, setpoint=1, output_limits=(0, 1))

    for _ in range(10):
        assert controller.update(measurement=0, dt=1) == 1

    assert controller.integral == pytest.approx(1)
    assert controller.update(measurement=2, dt=1) == 0


@pytest.mark.parametrize(
    ("k_aw", "expected_integral"),
    [(0.0, 0.5), (0.5, 0.4375), (1.0, 0.375)],
)
def test_k_aw_scales_back_calculation_with_dt(k_aw: float, expected_integral: float) -> None:
    controller = PIDController(kp=0, ki=1, kd=0, setpoint=1, output_limits=(0, 0.25), k_aw=k_aw)

    assert controller.update(measurement=0, dt=0.5) == 0.25
    assert controller.integral == pytest.approx(expected_integral)


@pytest.mark.parametrize("k_aw", [-1.0, float("inf"), float("nan")])
def test_k_aw_must_be_finite_and_non_negative(k_aw: float) -> None:
    with pytest.raises(ValueError, match="k_aw"):
        PIDController(kp=0, ki=1, kd=0, k_aw=k_aw)


def test_back_calculation_works_with_negative_integral_gain() -> None:
    controller = PIDController(kp=0, ki=-1, kd=0, setpoint=1, output_limits=(0, 1))

    for _ in range(10):
        assert controller.update(measurement=0, dt=1) == 0

    assert controller.integral == pytest.approx(0)
    assert controller.update(measurement=2, dt=1) == 1


def test_saturation_without_integral_gain() -> None:
    controller = PIDController(kp=10, ki=0, kd=0, setpoint=1, output_limits=(0, 1))

    assert controller.update(measurement=0, dt=1) == 1


def test_unsaturated_integral_is_unchanged() -> None:
    controller = PIDController(kp=0, ki=1, kd=0, setpoint=1)

    assert controller.update(measurement=0, dt=2) == 2
    assert controller.integral == pytest.approx(2)
