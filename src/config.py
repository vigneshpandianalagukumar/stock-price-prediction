"""Pipeline constants.

Every value here is copied from Cell 1 of the training notebook. Changing any
of them makes the saved models invalid, because the artefacts in `models/`
were fitted against exactly these settings.

START_DATE in particular is load-bearing in a way that is easy to miss.
VIX_Z is an EXPANDING z-score, so its value on any given date depends on where
the series starts. Download from 2020 instead of 2015 and VIX_Z comes out
different for the same calendar day, the meta-learner receives a feature it was
never trained on, and the predictions degrade silently with no error raised.
"""

RNG_SEED = 42

START_DATE = "2015-01-01"

# Prediction horizon in trading days, and the LSTM lookback window.
HORIZON = 20
LOOKBACK = 20

# Minimum trading days of history before a ticker is admitted.
MIN_HISTORY = 1500

# Minimum names required on a date before that cross-section is usable.
# Percentile ranks computed across a thin cross-section are not comparable
# with ranks computed across the full universe.
MIN_NAMES_PER_DAY = 40

# Daily move above this is treated as a corporate action, not information.
CORP_ACTION_THRESH = 0.35

MACRO_VIX = "^INDIAVIX"
MACRO_INDEX = "^NSEI"

# Artefact tag. The notebook writes files suffixed by horizon so several
# horizons can coexist in one folder.
TAG = f"h{HORIZON}"
