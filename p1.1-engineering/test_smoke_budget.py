import pytest

from run_cached_smoke import ReferenceBudget, ReferenceBudgetStop
from continue_smoke import ContinuationBudget


def test_reservations_prevent_concurrent_overcommit_and_release_after_usage():
    budget = ReferenceBudget(ceiling=.6)
    first = budget.reserve("synthetic")
    assert budget.reserved == first
    with pytest.raises(ReferenceBudgetStop):
        budget.reserve("x" * 1000000)
    budget.finish(first, {"prompt_tokens": 100, "completion_tokens": 10})
    assert budget.reserved == pytest.approx(0)
    assert budget.spent == pytest.approx(.00028)


def test_missing_usage_stops_new_paid_requests_instead_of_costing_it_zero():
    budget = ReferenceBudget()
    reserved = budget.reserve("synthetic")
    budget.finish(reserved, None)
    assert budget.unknown
    with pytest.raises(ReferenceBudgetStop):
        budget.reserve("another request")


def test_continuation_carries_prior_reference_and_unmetered_sdk_attempt():
    budget = ContinuationBudget(3.097576)
    reserved = budget.reserve("synthetic")
    budget.finish(reserved, {"prompt_tokens": 100, "completion_tokens": 10}, attempts=2)
    assert budget.spent == pytest.approx(3.097576 + .00028 + reserved / 2)
    assert budget.unmetered_upper == pytest.approx(reserved / 2)
    assert budget.reserved == pytest.approx(0)


def test_continuation_unknown_usage_retains_full_upper_and_stops():
    budget = ContinuationBudget(3.097576)
    reserved = budget.reserve("synthetic")
    budget.finish(reserved, None)
    assert budget.spent == pytest.approx(3.097576 + reserved)
    assert budget.unknown
    with pytest.raises(ReferenceBudgetStop):
        budget.reserve("must stop")


def test_bounded_unknown_retains_upper_and_allows_only_affordable_first_attempts():
    budget = ContinuationBudget(0, ceiling=1, stop_on_unknown_usage=False)
    reserved = budget.reserve("synthetic")
    budget.finish(reserved, None)
    assert budget.had_unknown_usage and not budget.unknown
    assert budget.spent == pytest.approx(reserved)
    next_reserved = budget.reserve("another first attempt")
    with pytest.raises(ReferenceBudgetStop):
        budget.reserve("x" * 1000000)
    budget.finish(next_reserved, {"prompt_tokens": 100, "completion_tokens": 10})
    assert budget.spent < budget.ceiling
