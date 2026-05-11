#!/usr/bin/env python3
"""
Estimate per-person page-access probability under a group-isolation constraint.

Model assumptions:
- Total population is P people.
- Population is split into groups of size N.
- Each person accesses the webpage independently with probability p.
- We want X% of all accesses to come from groups where at most one person
  accessed the webpage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    population: int
    group_size: int
    target_fraction: float
    groups: float
    per_person_probability: float
    expected_total_accesses: float
    expected_singleton_accesses: float


def compute_estimate(population: int, group_size: int, target_percent: float) -> Estimate:
    """
    Derivation:

    Let K be number of accessors in one group. K ~ Binomial(N, p).

    - Expected total accesses per group:
        E[K] = N * p

    - Accesses that are attributable to "at most one person in the group":
      only the K=1 case contributes one qualifying access.
        E[qualifying accesses per group] = P(K=1)
                                         = N * p * (1 - p)^(N - 1)

    - Fraction of accesses that qualify (ratio of expectations):
        f = (N * p * (1 - p)^(N - 1)) / (N * p)
          = (1 - p)^(N - 1)

    We set f = X (target fraction as [0,1]) and solve:
        (1 - p)^(N - 1) = X
        p = 1 - X^(1 / (N - 1))
    """
    if population <= 0:
        raise ValueError("population must be > 0")
    if group_size <= 1:
        raise ValueError("group_size must be > 1")
    if not (0 < target_percent < 100):
        raise ValueError("target_percent must be between 0 and 100 (exclusive)")

    x = target_percent / 100.0
    p = 1.0 - (x ** (1.0 / (group_size - 1)))

    expected_total_accesses = population * p
    expected_singleton_accesses = expected_total_accesses * x
    groups = population / group_size

    return Estimate(
        population=population,
        group_size=group_size,
        target_fraction=x,
        groups=groups,
        per_person_probability=p,
        expected_total_accesses=expected_total_accesses,
        expected_singleton_accesses=expected_singleton_accesses,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Join estimate calculator")
    parser.add_argument(
        "--population",
        type=int,
        default=450_000_000,
        help="Total population size (default: 450,000,000)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=20_000,
        help="People per group N (default: 20,000)",
    )
    parser.add_argument(
        "--target-percent",
        type=float,
        default=80.0,
        help="Desired percentage X of accesses from <=1 accessor groups (default: 80)",
    )
    args = parser.parse_args()

    est = compute_estimate(args.population, args.group_size, args.target_percent)

    print(f"Population: {est.population:,}")
    print(f"Group size (N): {est.group_size:,}")
    print(f"Estimated number of groups: {est.groups:,.2f}")
    print(f"Target qualifying-access fraction (X): {est.target_fraction:.4%}")
    print()
    print(f"Per-person access probability (p): {est.per_person_probability:.8f}")
    print(f"Expected total webpage accesses: {est.expected_total_accesses:,.2f}")
    print(f"Expected qualifying accesses (<=1 accessor/group): {est.expected_singleton_accesses:,.2f}")


if __name__ == "__main__":
    main()
