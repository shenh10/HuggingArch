"""`RouteFanout` against a second route: every top-k draw enumerated.

The dedup coefficient is a closed form — expected distinct destinations is
hypergeometric, `D·(1 - C(T-E_d,k)/C(T,k))` — and a closed form checked by
reading it again is checked by nothing. So this ENUMERATES: over small
parameters it walks every one of the `C(T,k)` draws a token can make,
counts the destinations that draw reaches, and averages with `Fraction`.
The two must agree EXACTLY, not to a tolerance: both are rationals, and a
difference in the last bit would be a different combinatorial claim.

Beside that the degeneracies, which the law must produce rather than branch
on: one pick per token leaves every destination distinct and the
coefficient is the ring's `(D-1)/D` (bit for bit what `ring_factor` returns
with no fan-out at all — the generalisation has to BE the ring there, not
merely round to it); dedup never sends more than the ring; and one
destination sends nothing.
"""
from __future__ import annotations

import math
import sys
import unittest
from fractions import Fraction
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout.comm_volume import RouteFanout, ring_factor


def enumerated_distinct(*, targets: int, picks: int, destinations: int,
                        experts_each: int) -> Fraction:
    """The distinct destinations one token reaches, averaged over every draw.

    The `targets` a top-k draws from are `0 … T-1`; destination `d` holds
    the weighted experts `[d·E_d, (d+1)·E_d)`, and any target past
    `destinations·E_d` is a zero-compute reserve that belongs to no
    destination and so sends nothing.
    """
    placed = destinations * experts_each
    total = 0
    draws = 0
    for draw in combinations(range(targets), picks):
        draws += 1
        total += len({pick // experts_each for pick in draw
                      if pick < placed})
    return Fraction(total, draws)


def enumerated_factor(*, targets: int, picks: int, destinations: int,
                      experts_each: int) -> Fraction:
    """The level's coefficient from the enumeration: the share whose
    destination is not this rank, times the distinct destinations a token
    reaches, over the copies of it the payload holds."""
    placed = destinations * experts_each
    copies = Fraction(picks * placed, targets)
    return (Fraction(destinations - 1, destinations)
            * enumerated_distinct(targets=targets, picks=picks,
                                  destinations=destinations,
                                  experts_each=experts_each)
            / copies)


class TestEnumeratedAgainstClosedForm(unittest.TestCase):
    """`E = T = 16`, `k = 3`, one world of 16 ranks read at four levels: a
    destination of one rank (16 of them), of two, of four, of eight."""

    WORLD = 16
    PICKS = 3

    def test_distinct_destinations_match_every_draw(self):
        for dest_members in (1, 2, 4, 8):
            destinations = self.WORLD // dest_members
            fanout = RouteFanout(picks=self.PICKS, targets=16, placed=16)
            experts_each = fanout.experts_per_destination(
                self.WORLD, dest_members)
            self.assertEqual(experts_each, dest_members)
            self.assertEqual(
                fanout.distinct(destinations, experts_each),
                enumerated_distinct(targets=16, picks=self.PICKS,
                                    destinations=destinations,
                                    experts_each=experts_each),
                f"destinations of {dest_members} ranks")

    def test_factors_match_every_draw(self):
        for dest_members in (1, 2, 4, 8):
            fanout = RouteFanout(picks=self.PICKS, targets=16, placed=16)
            destinations = self.WORLD // dest_members
            self.assertEqual(
                fanout.remote_factor(self.WORLD, scope_members=self.WORLD,
                                     dest_members=dest_members),
                enumerated_factor(
                    targets=16, picks=self.PICKS, destinations=destinations,
                    experts_each=fanout.experts_per_destination(
                        self.WORLD, dest_members)),
                f"destinations of {dest_members} ranks")

    def test_a_zero_compute_reserve_sends_nothing(self):
        """`T = 24 > E = 16` (LongCat's reserve): a pick that lands on a
        destination holding no weights is not a copy, and the enumeration
        says so the same way the closed form does."""
        fanout = RouteFanout(picks=3, targets=24, placed=16)
        self.assertEqual(fanout.copies_per_token, Fraction(2))
        self.assertEqual(fanout.experts_per_destination(8, 1), 2)
        self.assertEqual(
            fanout.remote_factor(8, scope_members=8, dest_members=1),
            enumerated_factor(targets=24, picks=3, destinations=8,
                              experts_each=2))


class TestDegeneracies(unittest.TestCase):

    def test_one_pick_is_the_ring_itself(self):
        """Every pick its own destination: the coefficient is `(D-1)/D`, and
        the same float `ring_factor` returns with no fan-out — the
        generalisation IS the ring there."""
        for world in (2, 4, 8, 16, 64):
            fanout = RouteFanout(picks=1, targets=world * 3, placed=world * 3)
            self.assertEqual(fanout.remote_factor(world),
                             Fraction(world - 1, world))
            self.assertEqual(
                ring_factor("all_to_all", world, fanout),
                ring_factor("all_to_all", world),
                f"a one-pick fan-out of {world} ranks is not the ring")

    def test_dedup_never_sends_more_than_the_ring(self):
        for picks in (1, 2, 4, 8):
            for world in (2, 4, 8, 32):
                fanout = RouteFanout(picks=picks, targets=256, placed=256)
                self.assertLessEqual(fanout.remote_factor(world),
                                     Fraction(world - 1, world),
                                     f"k={picks} over {world} ranks")

    def test_one_destination_sends_nothing(self):
        """A level with one destination is no exchange: the coefficient is
        0, so the level produces no phase without anyone testing for it."""
        fanout = RouteFanout(picks=8, targets=256, placed=256)
        self.assertEqual(fanout.remote_factor(8, scope_members=8,
                                              dest_members=8), Fraction(0))
        self.assertEqual(fanout.remote_factor(1), Fraction(0))

    def test_every_destination_is_reached_when_the_picks_cannot_miss(self):
        """More picks than the targets outside a destination: it is always
        reached, and the hypergeometric term is 0 rather than negative."""
        # Each of 4 destinations holds 2 of the 8 targets. At 6 picks the 2
        # unpicked targets can be exactly one destination's, so it is missed
        # in 1 of the C(8,6) draws and the average is just under 4.
        self.assertEqual(RouteFanout(picks=6, targets=8, placed=8)
                         .distinct(4, 2), 4 * (1 - Fraction(1, math.comb(8, 6))))
        # At 7 picks no destination can be missed: `comb(6, 7)` is 0.
        self.assertEqual(math.comb(8 - 2, 7), 0)
        self.assertEqual(RouteFanout(picks=7, targets=8, placed=8)
                         .distinct(4, 2), Fraction(4))


class TestRefusals(unittest.TestCase):

    def test_the_counts_are_positive_and_ordered(self):
        for kwargs in ({"picks": 0, "targets": 8, "placed": 8},
                       {"picks": 2, "targets": 8, "placed": 0},
                       {"picks": 9, "targets": 8, "placed": 8},
                       {"picks": 2, "targets": 8, "placed": 9}):
            with self.assertRaises(ValueError):
                RouteFanout(**kwargs)

    def test_an_uneven_expert_placement_is_refused(self):
        """Not rounded: an expert count the route group does not divide is a
        placement question, and the dispatch already refuses it
        (`ExpectedRoutedLoad._check_route_group`)."""
        fanout = RouteFanout(picks=2, targets=12, placed=12)
        with self.assertRaises(ValueError):
            fanout.experts_per_destination(8, 1)

    def test_a_round_trip_keeps_the_counts(self):
        fanout = RouteFanout(picks=8, targets=257, placed=256)
        self.assertEqual(RouteFanout.from_dict(fanout.as_dict()), fanout)
        self.assertEqual(fanout.symbol_suffix, "_k8t257e256")


if __name__ == "__main__":
    unittest.main()
