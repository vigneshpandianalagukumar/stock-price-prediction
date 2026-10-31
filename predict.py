#!/usr/bin/env python3
"""Command-line inference for the trained Bayesian stacking ensemble.

    python predict.py --all
    python predict.py --ticker RELIANCE
    python predict.py --top 20
    python predict.py --all --offline          # no network, uses data/raw_prices.csv
    python predict.py --all --out picks.csv

A note that matters for interpreting anything this prints: the model forecasts
SECTOR-NEUTRAL relative return. A positive number means the stock is expected
to beat its own sector's average over the next 20 trading days. It is not a
forecast that the share price rises.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import HORIZON, START_DATE
from src.inference import lookup, run


def _fmt(df):
    show = df.copy()
    show["PredRet"] = (show["PredRet"] * 100).map(lambda x: f"{x:+.2f}%")
    show["VaR95_Lo"] = (show["VaR95_Lo"] * 100).map(lambda x: f"{x:+.1f}%")
    show["VaR95_Hi"] = (show["VaR95_Hi"] * 100).map(lambda x: f"{x:+.1f}%")
    show["PredRank"] = show["PredRank"].map(lambda x: f"{x:+.4f}")
    return show[["Ticker", "PredRank", "PredRet", "VaR95_Lo", "VaR95_Hi"]]


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Score NSE equities with the trained stacking ensemble.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ticker", help="NSE symbol or fragment, e.g. RELIANCE or SBIN")
    g.add_argument("--all", action="store_true", help="Print the full ranking")
    g.add_argument("--top", type=int, metavar="N",
                   help="Print the top N and bottom N names")

    p.add_argument("--offline", action="store_true",
                   help="Use the cached price snapshot instead of downloading")
    p.add_argument("--start", default=START_DATE,
                   help=f"Download start date (default {START_DATE}; changing it "
                        "alters the expanding VIX z-score and degrades accuracy)")
    p.add_argument("--end", default=None, help="Download end date, default today")
    p.add_argument("--out", metavar="CSV", help="Also write predictions to this file")
    p.add_argument("--models", default=None, help="Directory holding the artefacts")
    p.add_argument("--quiet", action="store_true")

    a = p.parse_args(argv)
    if not (a.ticker or a.all or a.top):
        a.top = 10

    kw = dict(start=a.start, end=a.end, offline=a.offline, verbose=not a.quiet)
    if a.models:
        kw["models_dir"] = a.models

    try:
        results = run(**kw)
    except FileNotFoundError as e:
        print(f"\n{e}\n", file=sys.stderr)
        return 1

    if a.out:
        results.to_csv(a.out, index=False)
        if not a.quiet:
            print(f"Wrote {a.out}")

    as_of = results["AsOfDate"].iloc[0]
    print(f"\nPredictions as of {as_of} | horizon {HORIZON} trading days | "
          f"{len(results)} tickers")
    print("Values are relative to each stock's own sector, not absolute return.\n")

    pd.set_option("display.width", 120)

    if a.ticker:
        hits = lookup(results, a.ticker)
        if len(hits) == 0:
            print(f"No match for '{a.ticker}' among the {len(results)} names in "
                  "the trained universe.")
            print("This looks up the NSE symbol, not the company name -- try "
                  "'TMPV' rather than 'Tata Motors'.")
            return 1
        print(_fmt(hits).to_string(index=False))
    elif a.all:
        print(_fmt(results).to_string(index=False))
    else:
        n = a.top
        print(f"TOP {n} -- expected to outperform their sector")
        print(_fmt(results.head(n)).to_string(index=False))
        print(f"\nBOTTOM {n} -- expected to underperform their sector")
        print(_fmt(results.tail(n).iloc[::-1]).to_string(index=False))

    if bool(results["LowSampleWarning"].iloc[0]):
        print(f"\nCaution: only ~{results['NCycles'].iloc[0]:.0f} independent "
              f"{HORIZON}-day cycles support this horizon. Directional, not precise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
