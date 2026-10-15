"""Tests for exponential-build phase 2.2: the stdlib CSV expression
domain (`genius/csvquery.py`, exposed as `genius data --csv ... --expr`).

The contract under test:

- named-column expressions over a data.parse_table() table evaluate per
  row (arithmetic, comparisons, Kleene and/or/not) or collapse to a
  scalar through the stats.py aggregates;
- a missing/non-numeric cell or a division by zero makes the row value
  None — never 0 by assumption — and the count is reported;
- the ast whitelist refuses everything else BY NAME before anything
  runs (no eval, no attributes, no subscripts, no unknown calls);
- quantile's q is rejected outside [0, 1], never clamped; unknown
  columns are an honest error listing what exists;
- TSV works through the same parse_table delimiter path; the CLI wires
  the evaluator read-only over a user-supplied file.
"""
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from assistant.capabilities.genius import csvquery, data as genius_data
from assistant.capabilities.genius.cli import main as genius_main

CSV = (
    "item,price,qty,note\n"
    "apple,2.0,4,\n"
    "pear,1.5,2,cheap\n"
    "kiwi,,10,\n"
    "plum,4.0,0,\n"
)


def _table(text=CSV, delimiter=","):
    return genius_data.parse_table(text, delimiter=delimiter)


class RowExpressionTests(unittest.TestCase):
    def test_arithmetic_over_named_columns(self):
        out = csvquery.evaluate(_table(), "price * qty")
        self.assertEqual(out["kind"], "column")
        self.assertEqual(out["values"], [8.0, 3.0, None, 0.0])
        self.assertEqual(out["rows_unavailable"], 1)  # kiwi's missing price

    def test_division_by_zero_is_unavailable_never_infinite(self):
        out = csvquery.evaluate(_table(), "price / qty")
        self.assertIsNone(out["values"][3])  # 4.0 / 0
        self.assertEqual(out["values"][:3], [0.5, 0.75, None])

    def test_comparisons_and_kleene_logic(self):
        out = csvquery.evaluate(_table(), "price > 1.5 and qty >= 2")
        # pear: price 1.5 > 1.5 is False (present, just false); kiwi: None
        self.assertEqual(out["values"], [True, False, None, False])

    def test_not_propagates_none(self):
        out = csvquery.evaluate(_table(), "not (price > 2.0)")
        self.assertEqual(out["values"], [True, True, None, False])

    def test_comparison_chains(self):
        out = csvquery.evaluate(_table(), "1.0 < price <= 2.0")
        self.assertEqual(out["values"], [True, True, None, False])


class AggregateTests(unittest.TestCase):
    def test_mean_is_the_stats_primitive(self):
        out = csvquery.evaluate(_table(), "mean(price)")
        self.assertEqual(out["kind"], "scalar")
        # prices 2.0, 1.5, 4.0 (kiwi's None never becomes zero)
        self.assertAlmostEqual(out["value"], 2.5)

    def test_count_counts_present_values(self):
        out = csvquery.evaluate(_table(), "count(price)")
        self.assertEqual(out["value"], 3)

    def test_quantile_median_matches_stats(self):
        med = csvquery.evaluate(_table(), "quantile(price, 0.5)")
        self.assertAlmostEqual(med["value"], 2.0)

    def test_quantile_q_out_of_range_is_rejected_not_clamped(self):
        with self.assertRaises(ValueError):
            csvquery.evaluate(_table(), "quantile(price, 1.5)")
        with self.assertRaises(ValueError):
            csvquery.evaluate(_table(), "quantile(price, -0.1)")

    def test_pearson_pairs_rows_positionally(self):
        csv = "a,b\n1,2\n2,4\n3,6\n4,8\n"
        out = csvquery.evaluate(_table(csv), "pearson(a, b)")
        self.assertAlmostEqual(out["value"], 1.0)

    def test_thin_pearson_is_none(self):
        csv = "a,b\n1,2\n2,4\n"
        out = csvquery.evaluate(_table(csv), "pearson(a, b)")
        self.assertIsNone(out["value"])

    def test_mixed_aggregate_and_row_math(self):
        out = csvquery.evaluate(_table(), "price - mean(price)")
        # prices 2.0, 1.5, None, 4.0 -> mean 2.5; kiwi stays unavailable
        self.assertAlmostEqual(out["values"][0], -0.5, places=6)
        self.assertAlmostEqual(out["values"][1], -1.0, places=6)
        self.assertIsNone(out["values"][2])
        self.assertAlmostEqual(out["values"][3], 1.5, places=6)


class WhitelistTests(unittest.TestCase):
    def test_unknown_column_is_an_honest_error(self):
        with self.assertRaises(ValueError) as ctx:
            csvquery.evaluate(_table(), "prce * 2")
        self.assertIn("unknown column 'prce'", str(ctx.exception))
        self.assertIn("price", str(ctx.exception))  # lists what exists

    def test_attribute_access_refused(self):
        with self.assertRaises(ValueError):
            csvquery.evaluate(_table(), "price.__class__")

    def test_subscripts_and_lambdas_refused(self):
        for bad in ("price[0]", "(lambda x: x)(price)"):
            with self.assertRaises(ValueError):
                csvquery.evaluate(_table(), bad)

    def test_unknown_call_refused(self):
        with self.assertRaises(ValueError):
            csvquery.evaluate(_table(), "median_abs_deviation(price)")
        with self.assertRaises(ValueError):
            csvquery.evaluate(_table(), "__import__('os').getcwd()")

    def test_string_constants_refused(self):
        with self.assertRaises(ValueError):
            csvquery.evaluate(_table(), "'x' + 'y'")


class LoadingAndCliTests(unittest.TestCase):
    def test_tsv_through_the_same_parser(self):
        tsv = "a\tb\n1\t10\n2\t20\n"
        out = csvquery.evaluate(_table(tsv, delimiter="\t"), "a * b")
        self.assertEqual(out["values"], [10, 40])

    def test_cli_data_expr_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text(CSV, encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = genius_main(["data", "--csv", str(path),
                                  "--expr", "mean(qty)"])
            self.assertEqual(rc, 0)
            # quantities 4, 2, 10, 0 -> mean 4.0 (the CLI's YAML-ish
            # default rendering, not JSON)
            self.assertIn("value: 4.0", buf.getvalue())

    def test_cli_refused_expression_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text(CSV, encoding="utf-8")
            import contextlib
            err = io.StringIO()
            with contextlib.redirect_stderr(err), \
                    redirect_stdout(io.StringIO()):
                rc = genius_main(["data", "--csv", str(path),
                                  "--expr", "nosuchcol * 2"])
            self.assertEqual(rc, 1)
            self.assertIn("genius data:", err.getvalue())
            self.assertIn("unknown column", err.getvalue())


if __name__ == "__main__":
    unittest.main()
