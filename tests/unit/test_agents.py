from backend.agents.schemas import PropertyInput
from backend.agents.supervisor import Supervisor
from backend.agents.validator import Validator


def good(**kw) -> PropertyInput:
    base = dict(area_total=54, floor=5, floors_total=9, build_year=1985,
                house_type="panel", condition="cosmetic", horizon_months=6)
    base.update(kw)
    return PropertyInput(**base)


def test_validator_ok():
    r = Validator().run(good())
    assert r.ok and not r.errors and not r.warnings


def test_validator_floor_above_total():
    r = Validator().run(good(floor=12))
    assert not r.ok
    assert any("Этаж" in e for e in r.errors)


def test_validator_unknown_house_type_and_region():
    r = Validator().run(good(house_type="glass", region="Moscow"))
    assert len(r.errors) == 2


def test_validator_missing_fields_are_warnings():
    r = Validator().run(good(build_year=None, house_type=None, condition=None))
    assert r.ok and len(r.warnings) == 3


def test_supervisor_normal_case():
    res = Supervisor(confidence_threshold=0.6).run(good())
    assert res.price and res.interval_low < res.price < res.interval_high
    assert res.confidence >= 0.6 and not res.needs_review
    assert [s.agent for s in res.trace] == ["Validator", "Forecaster", "Critic", "Supervisor", "Explainer"]


def test_supervisor_sends_doubtful_case_to_expert():
    res = Supervisor(confidence_threshold=0.6).run(
        PropertyInput(area_total=250, floor=1, floors_total=3, horizon_months=12)
    )
    assert res.price is not None
    assert res.needs_review


def test_supervisor_invalid_input_has_no_forecast():
    res = Supervisor().run(good(floor=12))
    assert res.price is None and res.trace[0].status == "error"


def test_longer_horizon_is_more_expensive_and_wider():
    s = Supervisor()
    a, b = s.run(good(horizon_months=1)), s.run(good(horizon_months=12))
    assert b.price > a.price
    assert (b.interval_high - b.interval_low) > (a.interval_high - a.interval_low)
