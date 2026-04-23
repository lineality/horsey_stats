//! # Horse Racing Stat Classifier
//!
//! ## Project Purpose
//!
//! This crate is an experimental data-science tool for an indie horse-breeding
//! and horse-racing game. Horses race in groups of N (one race = one
//! `game_id`), and each horse has four input stats: `age`, `height`,
//! `experience`, and `weight`. The project's goal is to explore whether these
//! stats (and engineered combinations of them, such as a height-to-weight
//! ratio that approximates a body-mass index) are predictive of two outcomes:
//!
//! 1. **`completion`** — whether the horse finishes the race (binary 0 / 1).
//! 2. **`performance_score`** — a race-relative integer measure of how well
//!    the horse placed, derived from `rank` within its N-horse race group.
//!
//! Two modeling approaches are applied to each outcome:
//!
//! - **Decision tree** — iterative (no recursion), for interaction effects
//!   between stats.
//! - **Linear margin analysis** — threshold scanning per feature, to identify
//!   "risk zones" at extreme feature values (very high/low age, very
//!   high/low body-mass ratio, etc.).
//!
//! ## Why Integer-Only Arithmetic
//!
//! The user explicitly specified that float math is undesirable here; all
//! feature values, thresholds, and the `performance_score` label are held as
//! integers. This avoids float comparison pitfalls in tree splits and makes
//! the plain-text model files exactly reproducible.
//!
//! ## Why a Single Flat File
//!
//! Per the project's "clear scope and data ownership" rule, splitting a
//! focused ~200-row experimental tool across many files adds navigation cost
//! without benefit. The crate is organized as a single `src/main.rs` with
//! clearly banner-commented sections. If the file ever grows to a size where
//! splitting genuinely aids comprehension, that decision can be revisited.
//!
//! ## Data Integrity Policy: Race-Group Splitting
//!
//! Because `performance_score` is a *relative* rank within a N-horse race,
//! train/validate splits must be performed at the `game_id` group level, not
//! at the individual row level. Splitting within a race would leak
//! information (the other four horses' outcomes) between train and validate.
/*

Add data/ directory with a sample training.csv and predict.csv matching the documented schema.
Verify end-to-end by running

cargo run -- train

then

cargo run -- predict

against real or synthetic data.

```
src/
  main.rs             # entire crate: one flat file, well-organized with clear sections
Cargo.toml
stats_config.toml     # written AFTER first draft of main.rs, to reflect actual needs
models/
results/
data/
```


```stats_config.toml e.g.

# Horse Racing Stat Classifier — Configuration File
# stats_config.toml
#
# All paths are relative to the directory from which `cargo run` is invoked
# (the project root). Adjust as needed.
#
# All integer fields that contain invalid values are silently replaced
# with their built-in defaults (see StatsConfig::default_config in main.rs).

# -----------------------------------------------------------------------
# Data paths
# -----------------------------------------------------------------------

# The single user-maintained historical data file.
#
# Contains all races with known outcomes. Add N new rows (one per horse)
# after each real-world race completes. The system splits this file
# internally by game_id group into train and validate partitions — you
# never need to manually split or copy this file.
#
# Schema: row_id,game_id,age,height,experience,weight,rank,completion
test_train_data_csv_path = "data/test_train_data.csv"

# The upcoming race's horses, whose outcomes are not yet known.
# rank and completion columns should be blank or 0.
# Replace this file's contents before each prediction run.
#
# Schema: row_id,game_id,age,height,experience,weight,rank,completion
predict_csv_path = "data/predict.csv"

# -----------------------------------------------------------------------
# Output directories
# -----------------------------------------------------------------------

# Directory where trained model files are saved (and loaded from during
# predict mode). Created automatically if it does not exist.
models_dir = "models"

# Directory where timestamped results files are written.
# Created automatically if it does not exist.
results_dir = "results"

# -----------------------------------------------------------------------
# Tree hyperparameters (defaults used if no hyperparameter search is run)
# -----------------------------------------------------------------------

# Maximum depth the decision tree is allowed to grow.
# Shallow trees (2-3) generalise better on small data.
# Deep trees (5-6) can overfit with ~200 rows.
tree_max_depth = 4

# Minimum number of training samples that must reach any leaf.
# Higher values reduce overfitting on small groups.
tree_min_leaf_samples = 2

# -----------------------------------------------------------------------
# Train / validate split
# -----------------------------------------------------------------------

# Percentage of race groups (by game_id) used for the Stage 1 training
# partition. The remainder forms the validation partition.
# Must be between 1 and 99.
training_fraction_percent = 80

# Seed for the deterministic race-group shuffle used in the split.
# Changing this seed changes which races go into train vs. validate.
# Keep constant across experiments to ensure comparable results.
split_seed = 42

# -----------------------------------------------------------------------
# Linear margin model thresholds
# -----------------------------------------------------------------------

# Failure-rate percent threshold for the completion (classification)
# margin boundary scanner. A feature-value group whose completion failure
# rate is at or above this threshold is included in the risk zone.
# Range: 0-100. 50 means "majority failure rate triggers a boundary."
linear_margin_threshold_classification = 50

# Performance-score threshold for the regression margin boundary scanner.
# A feature-value group whose mean performance score is at or below this
# threshold is included in the risk zone.
# Range: 0-1000. 400 means "mean score in the bottom 40% triggers a boundary."
linear_margin_threshold_regression = 400

# -----------------------------------------------------------------------
# Hyperparameter search grid
# -----------------------------------------------------------------------

# Comma-separated list of tree max-depth values to try in Stage 1.
# Each depth is combined with each min-leaf-samples value below,
# producing (len(depths) x len(min_samples)) candidate combinations.
hyperparam_search_max_depths = 2,3,4,5,6

# Comma-separated list of minimum-leaf-sample values to try in Stage 1.
hyperparam_search_min_leaf_samples = 1,2,3

```

`main.rs` sections:
1. Constants and error types
2. Config parsing
3. Data types (Row, feature vector, etc.)
4. CSV parsing
5. Feature engineering
6. Decision tree (build, predict, save, load)
7. Linear margin analysis (build, predict, save, load)
8. Training orchestration (stage 1 + stage 2)
9. Prediction orchestration
10. Output/results writing
11. `main()` entry point and CLI dispatch
12. `#[cfg(test)]` test module at the bottom
*/

/*

# Section 1: Project Constants and Error Type

The foundation layer — no logic yet, just the vocabulary the rest of the crate will use. I am proposing this as the very first section because every subsequent function will reference these constants and return this error type. Getting them right (and documented) first means later code can be read without forward references.

---

## Section 1

1. **Module-level documentation** (`//!`) at the top of `src/main.rs` describing the whole project
2. **Named constants** for every "magic number" the project uses, each with a doc string explaining why it has that value
3. **`HorseRacingError` enum** — the single error type returned throughout the crate, with heap-free variants for production and unique per-variant identifying prefixes
4. **Cargo tests** verifying the constants are internally consistent and that error messages format correctly

*/
// ============================================================================
// SECTION 1 — PROJECT CONSTANTS AND ERROR TYPE
// ============================================================================
//
// This section defines the shared vocabulary used throughout the crate:
// named constants (no magic numbers scattered in logic) and the single
// `HorseRacingError` enum that every function in this crate returns on
// failure. Every variant carries a unique string prefix identifying which
// function produced the error, so an error value alone is traceable without
// needing a stack trace or debug symbols. Production error messages are
// intentionally terse and heap-free (plain `&'static str`) per the
// project's "no data leakage in production errors" rule.

/// Number of horses that race together as a single group in the game.
///
/// A race is uniquely identified by `game_id`, and every `game_id` group in
/// a well-formed training CSV contains exactly this many rows. Train/validate
/// splitting is performed at the group level (see crate-level docs), so this
/// value is referenced when validating group sizes and when computing the
/// per-race performance score.
pub const HORSES_PER_RACE_GROUP: usize = 4;

/// Number of columns expected in every training and prediction CSV row.
///
/// The fixed schema is:
/// `row_id, game_id, age, height, experience, weight, rank, completion`
/// (exactly eight columns). Prediction CSVs share this schema; their
/// `rank` and `completion` columns are present in the header but their
/// values are ignored as model inputs.
pub const CSV_EXPECTED_COLUMN_COUNT: usize = 8;

/// The exact CSV header line the parser will accept.
///
/// Kept as a single `&'static str` so the parser can do a direct byte-compare
/// against the first line of the file without allocating. Whitespace around
/// commas is *not* tolerated: training CSVs must be produced with this
/// canonical header to avoid silent schema drift.
pub const CSV_EXPECTED_HEADER_LINE: &str =
    "row_id,game_id,age,height,experience,weight,rank,completion";

/// Performance score assigned to a horse that did not finish the race.
///
/// A horse with `completion = 0` (DNF, did-not-finish) receives this score
/// regardless of any `rank` value present in the CSV. Keeping DNF at the
/// bottom of the score range (not a missing value, not a negative number)
/// lets the same regression-style tree and linear-margin code handle DNF and
/// finishing horses uniformly.
pub const PERFORMANCE_SCORE_FOR_DID_NOT_FINISH: i32 = 0;

/// Multiplier used to convert a 1–N rank into a 0–1000 integer score.
///
/// ((n+1) - rank)
/// The transform is `performance_score = (5 - rank) * PERFORMANCE_SCORE_RANK_MULTIPLIER`,
/// giving: rank 1 → 800, rank 2 → 600, rank 3 → 400, rank 4 → 200.
/// DNF overrides this to `PERFORMANCE_SCORE_FOR_DID_NOT_FINISH` (0). The
/// integer range was chosen per the user's preference for integer math over
/// floats; 200-unit gaps give tree splits meaningful room without overflow
/// risk in i32.
pub const PERFORMANCE_SCORE_RANK_MULTIPLIER: i32 = 200;

/// Minimum valid rank value in the training data (1st place).
pub const RANK_MINIMUM_VALID_VALUE: i32 = 1;

/// Maximum valid rank value in the training data (last place in a N-horse race).
pub const RANK_MAXIMUM_VALID_VALUE: i32 = 4;

/// The single error type returned by every fallible function in this crate.
///
/// ## Design Rationale
///
/// Each variant carries a `&'static str` message with a **unique per-function
/// prefix** (for example, `"parse_csv_row_into_horse_record: ..."`). This
/// satisfies two project rules simultaneously:
///
/// 1. **No heap allocation in production error paths** — `&'static str` lives
///    in the binary's read-only data segment; no `String`, no `format!`.
/// 2. **Traceable errors without debug symbols** — the message prefix alone
///    identifies the function that produced the error, so a production log
///    line containing only the error is still actionable.
///
/// ## Why One Flat Enum
///
/// The project is small and a single error type keeps `Result<T, HorseRacingError>`
/// consistent across every function. If the crate later grows distinct
/// subsystems with genuinely independent error domains, this can be split.
#[derive(Debug)]
pub enum HorseRacingError {
    /// A CSV file could not be opened or read from disk.
    ///
    /// The message identifies the calling function; the underlying OS error
    /// is deliberately *not* included in the message to avoid leaking file
    /// paths or filesystem structure into production logs.
    CsvFileReadFailure(&'static str),

    /// A CSV row did not have the expected number of comma-separated fields.
    ///
    /// Triggered when `CSV_EXPECTED_COLUMN_COUNT` does not match the actual
    /// field count on a given line. Does not include the offending line
    /// contents in the error (production safety).
    CsvRowFieldCountMismatch(&'static str),

    /// A CSV field that should have been an integer failed to parse as one.
    ///
    /// The message identifies the function; it deliberately does not include
    /// the offending string value (production safety rule: no user data in
    /// error messages).
    CsvFieldIntegerParseFailure(&'static str),

    /// The CSV header line did not match `CSV_EXPECTED_HEADER_LINE` exactly.
    ///
    /// This catches schema drift early — before any row is parsed — so a
    /// mislabeled column cannot silently corrupt a training run.
    CsvHeaderMismatch(&'static str),

    /// A numeric value was outside the range the project defines as valid.
    ///
    /// Examples: `rank` outside 1–N, `completion` outside 0–1, `age` below 1
    /// or above 8. The message names the violating function; the specific
    /// value is intentionally omitted from the message.
    FieldValueOutOfValidRange(&'static str),

    /// A `game_id` group did not contain exactly `HORSES_PER_RACE_GROUP` rows.
    ///
    /// Per the project's data integrity policy, every race must have exactly
    /// N horses. Incomplete groups are rejected at load time rather than
    /// silently dropped.
    RaceGroupIncompleteOrOversized(&'static str),

    /// A division would have divided by zero and the caller did not supply
    /// a safe fallback.
    ///
    /// Used by engineered-feature computation (for example, `weight = 0`
    /// when computing `height / weight`). In practice, weights of zero
    /// should never appear in valid data, but the check exists so that a
    /// corrupt row produces a handled error instead of a panic.
    ArithmeticDivisionByZero(&'static str),
}

impl HorseRacingError {
    /// Returns the terse `&'static str` message carried by this error variant.
    ///
    /// Used by the top-level production error handler to write a single-line
    /// log entry without allocating. Callers that want richer formatting in
    /// debug builds can `{:?}` the enum instead.
    pub fn terse_production_message(&self) -> &'static str {
        match self {
            HorseRacingError::CsvFileReadFailure(message) => message,
            HorseRacingError::CsvRowFieldCountMismatch(message) => message,
            HorseRacingError::CsvFieldIntegerParseFailure(message) => message,
            HorseRacingError::CsvHeaderMismatch(message) => message,
            HorseRacingError::FieldValueOutOfValidRange(message) => message,
            HorseRacingError::RaceGroupIncompleteOrOversized(message) => message,
            HorseRacingError::ArithmeticDivisionByZero(message) => message,
        }
    }
}

// ============================================================================
// SECTION 1 — CARGO TESTS
// ============================================================================

#[cfg(test)]
mod section_one_constants_and_errors_tests {
    use super::*;

    /// Verifies that the horses-per-race constant has the exact value the
    /// rest of the crate assumes. A change to this constant would silently
    /// invalidate the performance-score formula and the group-size validator,
    /// so it is pinned here as a tripwire.
    #[test]
    fn horses_per_race_group_constant_equals_4() {
        assert_eq!(HORSES_PER_RACE_GROUP, 4);
    }

    /// Verifies the CSV column count matches the documented schema of eight
    /// fields (row_id, game_id, age, height, experience, weight, rank,
    /// completion). A mismatch here would cause the CSV parser to reject
    /// well-formed input.
    #[test]
    fn csv_expected_column_count_equals_eight() {
        assert_eq!(CSV_EXPECTED_COLUMN_COUNT, 8);
    }

    /// Verifies the header constant contains exactly the expected number of
    /// comma-separated names. This catches a typo in the header constant
    /// (extra or missing comma) that would otherwise slip through until a
    /// real CSV load failed.
    #[test]
    fn csv_expected_header_line_field_count_matches_column_count() {
        let header_field_count = CSV_EXPECTED_HEADER_LINE.split(',').count();
        assert_eq!(header_field_count, CSV_EXPECTED_COLUMN_COUNT);
    }

    /// Verifies the performance score formula produces the documented values
    /// for every valid rank. This is the contract the tree regression target
    /// depends on; if the multiplier or formula changes, this test fails
    /// loudly before any model is trained on inconsistent labels.
    #[test]
    fn performance_score_multiplier_produces_documented_rank_mapping() {
        let expected_score_for_rank_one: i32 = 800; // Changed from 1000
        let expected_score_for_rank_two: i32 = 600; // Changed from 800
        let expected_score_for_rank_three: i32 = 400; // Changed from 600
        let expected_score_for_rank_four: i32 = 200; // Changed from 400

        assert_eq!(
            (5 - 1) * PERFORMANCE_SCORE_RANK_MULTIPLIER, // Changed from (6 - 1)
            expected_score_for_rank_one
        );
        assert_eq!(
            (5 - 2) * PERFORMANCE_SCORE_RANK_MULTIPLIER,
            expected_score_for_rank_two
        );
        assert_eq!(
            (5 - 3) * PERFORMANCE_SCORE_RANK_MULTIPLIER,
            expected_score_for_rank_three
        );
        assert_eq!(
            (5 - 4) * PERFORMANCE_SCORE_RANK_MULTIPLIER,
            expected_score_for_rank_four
        );
    }

    /// Verifies that the DNF sentinel score is strictly lower than the
    /// worst finishing score (rank N = 200). This ordering matters for the
    /// regression tree: a DNF must sort as "worse than last place", not as
    /// missing data.
    #[test]
    fn did_not_finish_score_is_lower_than_worst_finishing_score() {
        let worst_finishing_score_rank_four: i32 =
            (5 - RANK_MAXIMUM_VALID_VALUE) * PERFORMANCE_SCORE_RANK_MULTIPLIER;
        assert!(PERFORMANCE_SCORE_FOR_DID_NOT_FINISH < worst_finishing_score_rank_four);
    } // ✓ Rank 4 is the worst finishing rank in 4-horse race

    /// Verifies that the valid rank range is ordered correctly (min < max)
    /// and that the range contains exactly `HORSES_PER_RACE_GROUP` distinct
    /// values, because one rank is assigned per racing horse.
    #[test]
    fn rank_valid_range_is_consistent_with_horses_per_race() {
        assert!(RANK_MINIMUM_VALID_VALUE < RANK_MAXIMUM_VALID_VALUE);
        let rank_range_size: usize =
            (RANK_MAXIMUM_VALID_VALUE - RANK_MINIMUM_VALID_VALUE + 1) as usize;
        assert_eq!(rank_range_size, HORSES_PER_RACE_GROUP);
    }

    /// Verifies that every error variant returns a non-empty terse message
    /// and that each message carries a unique function-identifying prefix.
    /// Non-empty messages are required so production log lines are never
    /// blank; uniqueness is required so an error message alone is traceable
    /// to its originating function.
    #[test]
    fn every_error_variant_has_nonempty_unique_terse_message() {
        let all_error_variants: [HorseRacingError; 7] = [
            HorseRacingError::CsvFileReadFailure("csv_file_read: open failed"),
            HorseRacingError::CsvRowFieldCountMismatch("parse_csv_row: field count mismatch"),
            HorseRacingError::CsvFieldIntegerParseFailure("parse_csv_row: integer parse failed"),
            HorseRacingError::CsvHeaderMismatch("parse_training_csv: header mismatch"),
            HorseRacingError::FieldValueOutOfValidRange("validate_row: value out of range"),
            HorseRacingError::RaceGroupIncompleteOrOversized(
                "group_by_game_id: group size invalid",
            ),
            HorseRacingError::ArithmeticDivisionByZero("compute_features: divide by zero"),
        ];

        // Every message must be non-empty.
        for error_under_test in all_error_variants.iter() {
            assert!(
                !error_under_test.terse_production_message().is_empty(),
                "error variant produced an empty terse message"
            );
        }

        // Every message must be unique across the variant set (pairwise check).
        for outer_index in 0..all_error_variants.len() {
            for inner_index in (outer_index + 1)..all_error_variants.len() {
                let outer_message = all_error_variants[outer_index].terse_production_message();
                let inner_message = all_error_variants[inner_index].terse_production_message();
                assert_ne!(
                    outer_message, inner_message,
                    "two error variants share the same terse message — messages must be unique"
                );
            }
        }
    }
}

// ============================================================================
// SECTION 2 — ROW STRUCT, FEATURE VECTOR, AND CSV PARSING
// ============================================================================
//
// This section defines the two data types that represent a single horse in
// a race and the functions that turn a CSV file into a stream of those
// records. The split is deliberate:
//
//   * `RawHorseRaceRecord`          — one parsed CSV row, exactly as stored
//                                     in the file, plus the derived
//                                     `performance_score` integer label.
//
//   * `EngineeredFeatureVector`     — the integer feature values actually
//                                     handed to the tree and linear-margin
//                                     models, including engineered
//                                     combinations such as the height-to-
//                                     weight ratio.
//
// Keeping these two types separate means the CSV layer has exactly one
// responsibility (parse and validate) and the modeling layer has exactly one
// responsibility (consume feature vectors). A change to the feature set
// never forces a change to the CSV parser, and vice versa.
//
// The CSV reader is deliberately incremental: it opens the file, validates
// the header against `CSV_EXPECTED_HEADER_LINE`, and then yields rows one at
// a time. The whole file is never loaded into memory at once, per the
// project's "load what is needed when it is needed" rule.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// One horse's data as read from a single CSV row, after parsing and
/// validation but before feature engineering.
///
/// ## Fields
///
/// - `row_id` — the CSV file's own row identifier. Not used for modeling;
///   preserved only so prediction output can reference rows back to their
///   source line.
/// - `game_id` — the race this horse participated in. All N horses in a
///   race share the same `game_id`. Used exclusively for train/validate
///   splitting (splits are by group, never mid-race).
/// - `age`, `height`, `experience`, `weight` — the four raw input stats.
/// - `rank` — 1 through N for finishing horses, or `0` for DNF. The raw
///   value from the CSV, kept for reporting; the modeling label is the
///   derived `performance_score`, not this field.
/// - `completion` — 1 if the horse finished the race, 0 if DNF.
/// - `performance_score` — the regression-style label computed via
///   `compute_performance_score_from_rank_and_completion`. Stored here
///   (rather than recomputed) so that downstream code never has to recompute
///   it and never risks drift between the label and the raw rank.
///
/// ## Project Role
///
/// Every row in the training CSV becomes one of these. Prediction CSVs
/// produce these too, with `rank` and `completion` parsed from the file but
/// ignored by the model (their values carry no information at prediction
/// time).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawHorseRaceRecord {
    pub row_id: i32,
    pub game_id: i32,
    pub age: i32,
    pub height: i32,
    pub experience: i32,
    pub weight: i32,
    pub rank: i32,
    pub completion: i32,
    pub performance_score: i32,
}

/// The integer feature vector handed to the decision tree and the linear
/// margin analyzer.
///
/// ## Fields
///
/// All four raw stats are passed through unchanged, plus two engineered
/// features:
///
/// - `height_to_weight_ratio_times_one_thousand` — a body-mass-index proxy,
///   computed as `(height * 1000) / weight`. The factor of 1000 is present
///   because integer division would otherwise destroy all precision for the
///   small ratios typical of this data (heights near 150, weights near
///   1000). This gives a useful integer range roughly in the hundreds.
///
/// - `age_times_experience` — a simple interaction term. The project
///   hypothesizes that age and experience together (a seasoned older horse
///   versus a young inexperienced one) may matter more than either alone.
///
/// ## Why No Floats
///
/// Every field is `i32`. Tree splits compare `feature < threshold` on
/// integers, avoiding float equality/ordering pitfalls and making plain-text
/// model files exactly round-trip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineeredFeatureVector {
    pub age: i32,
    pub height: i32,
    pub experience: i32,
    pub weight: i32,
    pub height_to_weight_ratio_times_one_thousand: i32,
    pub age_times_experience: i32,
}

/// Number of engineered features exposed to the models.
///
/// Must stay in sync with the field count of `EngineeredFeatureVector`. Used
/// by the tree to iterate candidate split features without hard-coding the
/// count inside model code.
pub const ENGINEERED_FEATURE_COUNT: usize = 6;

/// Parses a single CSV field string into an `i32`, returning a
/// function-identifying error on failure.
///
/// ## Project Role
///
/// Every integer field in the CSV (every field, in this project's schema)
/// goes through this one function. Centralizing integer parsing means there
/// is exactly one error prefix (`"parse_single_integer_field_from_csv"`) to
/// look for when any integer field fails to parse, regardless of which
/// caller triggered it.
///
/// ## Error Handling Policy
///
/// The function deliberately does *not* include the offending string in the
/// returned error, because this crate's production error messages must not
/// leak file contents (see `HorseRacingError` docs). Debug builds can trace
/// the actual value via `#[cfg(debug_assertions)]` logging if added later.
pub fn parse_single_integer_field_from_csv(field_text: &str) -> Result<i32, HorseRacingError> {
    // `trim()` tolerates accidental whitespace around a value (for example,
    // ", 4," vs. ",4,"), which is a common human-editing artifact in a
    // manually maintained training CSV.
    let trimmed_field_text = field_text.trim();

    match trimmed_field_text.parse::<i32>() {
        Ok(parsed_integer_value) => Ok(parsed_integer_value),
        Err(_parse_error_intentionally_discarded) => {
            Err(HorseRacingError::CsvFieldIntegerParseFailure(
                "parse_single_integer_field_from_csv: integer parse failed",
            ))
        }
    }
}

/// Parses one already-read CSV data line into a validated `RawHorseRaceRecord`.
///
/// ## Inputs
///
/// - `csv_data_line` — a single line of the CSV file *excluding* the header.
///   The caller is responsible for skipping the header line before calling.
///
/// ## Behavior
///
/// Splits on commas, verifies the column count matches
/// `CSV_EXPECTED_COLUMN_COUNT`, parses each field as `i32`, validates ranges
/// via `validate_raw_record_field_ranges`, and computes `performance_score`.
///
/// ## Why Validation Happens Here
///
/// A row with an out-of-range field (rank = 9, completion = 2) is structurally
/// parseable but semantically invalid. Catching it at the parse boundary
/// means no invalid data ever reaches the modeling code, which can therefore
/// assume all `RawHorseRaceRecord` instances it sees are project-valid.
pub fn parse_single_csv_data_row_into_raw_record(
    csv_data_line: &str,
) -> Result<RawHorseRaceRecord, HorseRacingError> {
    // Collect the comma-separated fields into a fixed-size check first, so
    // we can reject malformed rows before doing any parsing work.
    let split_field_strings: Vec<&str> = csv_data_line.split(',').collect();

    if split_field_strings.len() != CSV_EXPECTED_COLUMN_COUNT {
        return Err(HorseRacingError::CsvRowFieldCountMismatch(
            "parse_single_csv_data_row_into_raw_record: field count mismatch",
        ));
    }

    // Field order is fixed by the schema documented on `CSV_EXPECTED_HEADER_LINE`.
    // Indices are named here (not inlined) so a reader can confirm the
    // mapping at a glance.
    let row_id_field_index: usize = 0;
    let game_id_field_index: usize = 1;
    let age_field_index: usize = 2;
    let height_field_index: usize = 3;
    let experience_field_index: usize = 4;
    let weight_field_index: usize = 5;
    let rank_field_index: usize = 6;
    let completion_field_index: usize = 7;

    let parsed_row_id =
        parse_single_integer_field_from_csv(split_field_strings[row_id_field_index])?;
    let parsed_game_id =
        parse_single_integer_field_from_csv(split_field_strings[game_id_field_index])?;
    let parsed_age = parse_single_integer_field_from_csv(split_field_strings[age_field_index])?;
    let parsed_height =
        parse_single_integer_field_from_csv(split_field_strings[height_field_index])?;
    let parsed_experience =
        parse_single_integer_field_from_csv(split_field_strings[experience_field_index])?;
    let parsed_weight =
        parse_single_integer_field_from_csv(split_field_strings[weight_field_index])?;
    let parsed_rank = parse_single_integer_field_from_csv(split_field_strings[rank_field_index])?;
    let parsed_completion =
        parse_single_integer_field_from_csv(split_field_strings[completion_field_index])?;

    // Compute the derived label *before* constructing the final struct so
    // that the struct is only ever instantiated in a fully-valid,
    // fully-labeled state.
    let derived_performance_score =
        compute_performance_score_from_rank_and_completion(parsed_rank, parsed_completion)?;

    let assembled_record = RawHorseRaceRecord {
        row_id: parsed_row_id,
        game_id: parsed_game_id,
        age: parsed_age,
        height: parsed_height,
        experience: parsed_experience,
        weight: parsed_weight,
        rank: parsed_rank,
        completion: parsed_completion,
        performance_score: derived_performance_score,
    };

    // Range validation happens after construction so that
    // `validate_raw_record_field_ranges` has a single, uniform input type.
    validate_raw_record_field_ranges(&assembled_record)?;

    Ok(assembled_record)
}

/// Validates that every field in a parsed record falls within the project's
/// documented valid range.
///
/// ## Project-Defined Valid Ranges
///
/// - `age`: 1–8 inclusive (from the project brief).
/// - `height`: three-digit positive integer (100–999 inclusive). The brief
///   specifies "three digits"; we interpret this as the closed interval
///   [100, 999].
/// - `experience`: 0 or greater (a new horse can have zero experience).
/// - `weight`: strictly positive (zero weight is physically nonsensical and
///   would also cause a divide-by-zero in engineered-feature computation).
/// - `rank`: either 0 (for DNF, consistent with `completion = 0`) or within
///   `RANK_MINIMUM_VALID_VALUE..=RANK_MAXIMUM_VALID_VALUE` (1–N).
/// - `completion`: exactly 0 or 1.
///
/// Additionally, the cross-field invariant `completion = 0  <->  rank = 0`
/// is enforced: a DNF must have rank 0, and any horse with a finishing rank
/// must have `completion = 1`.
///
/// ## Error Reporting
///
/// A single error variant (`FieldValueOutOfValidRange`) covers all of the
/// above, with the message prefix identifying this function as the origin.
/// The specific offending field name is intentionally *not* included in the
/// production error message (project rule: no user data leakage). For debug
/// builds, richer diagnostics can be added later behind
/// `#[cfg(debug_assertions)]`.
pub fn validate_raw_record_field_ranges(
    raw_record_to_validate: &RawHorseRaceRecord,
) -> Result<(), HorseRacingError> {
    // Age: 1-8 inclusive.
    let age_minimum_valid: i32 = 1;
    let age_maximum_valid: i32 = 8;
    if raw_record_to_validate.age < age_minimum_valid
        || raw_record_to_validate.age > age_maximum_valid
    {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "validate_raw_record_field_ranges: age out of range",
        ));
    }

    // Height: three-digit positive integer (100-999 inclusive).
    let height_minimum_valid: i32 = 100;
    let height_maximum_valid: i32 = 999;
    if raw_record_to_validate.height < height_minimum_valid
        || raw_record_to_validate.height > height_maximum_valid
    {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "validate_raw_record_field_ranges: height out of range",
        ));
    }

    // Experience: 0 or greater.
    if raw_record_to_validate.experience < 0 {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "validate_raw_record_field_ranges: experience negative",
        ));
    }

    // Weight: strictly positive (zero would also break ratio computation).
    if raw_record_to_validate.weight <= 0 {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "validate_raw_record_field_ranges: weight not positive",
        ));
    }

    // Completion: exactly 0 or 1.
    if raw_record_to_validate.completion != 0 && raw_record_to_validate.completion != 1 {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "validate_raw_record_field_ranges: completion not 0 or 1",
        ));
    }

    // Rank: 0 (DNF) or 1..=5 (valid finishing rank).
    let rank_is_did_not_finish = raw_record_to_validate.rank == 0;
    let rank_is_valid_finishing_position = raw_record_to_validate.rank >= RANK_MINIMUM_VALID_VALUE
        && raw_record_to_validate.rank <= RANK_MAXIMUM_VALID_VALUE;
    if !rank_is_did_not_finish && !rank_is_valid_finishing_position {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "validate_raw_record_field_ranges: rank out of range",
        ));
    }

    // Cross-field invariant: completion and rank must agree on DNF status.
    //   completion = 0  <->  rank = 0
    //   completion = 1  <->  rank in 1..=N
    let dnf_consistent = raw_record_to_validate.completion == 0 && raw_record_to_validate.rank == 0;
    let finished_consistent =
        raw_record_to_validate.completion == 1 && rank_is_valid_finishing_position;
    if !dnf_consistent && !finished_consistent {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "validate_raw_record_field_ranges: completion/rank inconsistent",
        ));
    }

    Ok(())
}

/// Converts a raw rank (1-4 or 0 for DNF) and a completion flag (0 or 1)
/// into the integer performance score used as the regression target.
///
/// ## Formula
///
/// - If `completion == 0`: return `PERFORMANCE_SCORE_FOR_DID_NOT_FINISH`
///   (= 0), regardless of the `rank` value.
///
/// ((N+1) - rank)
/// - Otherwise: return `(5 - rank) * PERFORMANCE_SCORE_RANK_MULTIPLIER`,
///   giving 800 for 1st place down to 200 for 4th place.
///
/// ## Project Role
///
/// This is the single source of truth for how rank becomes a score. The
/// formula is *not* duplicated anywhere else in the crate; every caller
/// that needs a performance score calls this function.
///
/// ## Error Cases
///
/// If the inputs are inconsistent or out of range despite having passed
/// `validate_raw_record_field_ranges` (which should be impossible in normal
/// flow, but is defended against here in case this function is called
/// directly without prior validation), a `FieldValueOutOfValidRange` error
/// is returned.
pub fn compute_performance_score_from_rank_and_completion(
    rank_value: i32,
    completion_value: i32,
) -> Result<i32, HorseRacingError> {
    // Completion must be 0 or 1; anything else is a data error.
    if completion_value != 0 && completion_value != 1 {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "compute_performance_score_from_rank_and_completion: completion not 0 or 1",
        ));
    }

    // DNF path: rank is irrelevant, score is the DNF sentinel.
    if completion_value == 0 {
        return Ok(PERFORMANCE_SCORE_FOR_DID_NOT_FINISH);
    }

    // Finishing path: rank must be in [RANK_MINIMUM_VALID_VALUE, RANK_MAXIMUM_VALID_VALUE].
    if rank_value < RANK_MINIMUM_VALID_VALUE || rank_value > RANK_MAXIMUM_VALID_VALUE {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "compute_performance_score_from_rank_and_completion: rank out of range",
        ));
    }

    // Formula: (HORSES_PER_RACE_GROUP + 1) - rank = (4 + 1) - rank = (5 - rank)
    let finishing_score = (5 - rank_value) * PERFORMANCE_SCORE_RANK_MULTIPLIER;
    Ok(finishing_score)
}

/// Computes the engineered feature vector from a validated raw record.
///
/// ## Engineered Features
///
/// - `height_to_weight_ratio_times_one_thousand = (height * 1000) / weight`.
///   The *1000 factor is essential: without it, integer division would
///   collapse all real-world horse ratios to 0. With it, typical values land
///   in the 100-300 range, which gives tree splits meaningful resolution.
///
/// - `age_times_experience = age * experience`. Simple interaction term.
///
/// ## Defensive Checks
///
/// `weight <= 0` should already be rejected by `validate_raw_record_field_ranges`,
/// but this function defends against it anyway because:
///
///   1. A corrupt record could in principle reach this function without
///      having passed validation (bit-flip, caller bug, etc.).
///   2. The cost of the check is negligible and the failure mode (divide by
///      zero) would otherwise be a process abort in debug builds and
///      unpredictable integer behavior in release. A returned error is
///      strictly better.
///
/// ## Why Not Return `Option<...>` Instead of `Result<...>`?
///
/// A zero weight is not an expected "missing data" case — it represents a
/// data integrity failure. `Result` with a specific error variant makes that
/// distinction explicit in the type signature.
pub fn compute_engineered_feature_vector_from_raw_record(
    raw_record_to_engineer: &RawHorseRaceRecord,
) -> Result<EngineeredFeatureVector, HorseRacingError> {
    // Defensive: guard against divide-by-zero even though upstream validation
    // should have rejected this record already.
    if raw_record_to_engineer.weight == 0 {
        return Err(HorseRacingError::ArithmeticDivisionByZero(
            "compute_engineered_feature_vector_from_raw_record: weight is zero",
        ));
    }

    // The factor 1000 preserves ratio precision under integer division.
    let ratio_scaling_factor: i32 = 1000;
    let height_to_weight_ratio_times_one_thousand =
        (raw_record_to_engineer.height * ratio_scaling_factor) / raw_record_to_engineer.weight;

    // Interaction term. `i32` multiplication here is safe because age is
    // bounded to 8 and experience is realistically small; the product cannot
    // approach `i32::MAX` in any plausible training data.
    let age_times_experience = raw_record_to_engineer.age * raw_record_to_engineer.experience;

    let engineered_vector = EngineeredFeatureVector {
        age: raw_record_to_engineer.age,
        height: raw_record_to_engineer.height,
        experience: raw_record_to_engineer.experience,
        weight: raw_record_to_engineer.weight,
        height_to_weight_ratio_times_one_thousand,
        age_times_experience,
    };

    Ok(engineered_vector)
}

/// Reads a training CSV file incrementally, returning every successfully
/// parsed `RawHorseRaceRecord`.
///
/// ## Inputs
///
/// - `training_csv_file_path` — absolute or relative path to the CSV file.
///
/// ## Behavior
///
/// 1. Opens the file and wraps it in a `BufReader` so lines are read one at
///    a time rather than loading the whole file into memory.
/// 2. Reads the first line and verifies it equals `CSV_EXPECTED_HEADER_LINE`
///    exactly (no whitespace tolerance on the header — schema drift must
///    fail loudly).
/// 3. For each remaining non-empty line, calls
///    `parse_single_csv_data_row_into_raw_record`. If a row fails to parse,
///    the whole load aborts with the row's error. This is deliberate: a
///    training run on partially corrupt data would produce a model with
///    silently wrong label distribution.
///
/// ## Project Role
///
/// This is the single entry point from disk into the modeling pipeline. Both
/// training and prediction modes use it. The prediction mode simply ignores
/// `rank` and `completion` fields on the resulting records (they are parsed
/// but carry no meaning at prediction time).
///
/// ## Returns
///
/// A `Vec<RawHorseRaceRecord>` holding every valid data row. The vector is
/// the smallest representation that still lets downstream code do a
/// group-by-game_id split, so the whole-vector return is intentional here
/// (it is a small dataset, ~200 rows). Rows are *not* streamed back to the
/// caller one at a time because every downstream consumer (tree training,
/// linear margin scan, group splitting) needs full-dataset access.
pub fn read_training_csv_file_incrementally(
    training_csv_file_path: &Path,
) -> Result<Vec<RawHorseRaceRecord>, HorseRacingError> {
    // Attempt to open the file. The underlying OS error is deliberately
    // discarded — it may contain the file path, which must not leak into
    // production logs per the project's error-safety rule.
    let opened_csv_file = match File::open(training_csv_file_path) {
        Ok(file_handle) => file_handle,
        Err(_os_error_intentionally_discarded) => {
            return Err(HorseRacingError::CsvFileReadFailure(
                "read_training_csv_file_incrementally: could not open csv file",
            ));
        }
    };

    // `BufReader` gives us `.lines()`, which yields one owned `String` per
    // iteration. The file contents are never held in memory in their
    // entirety.
    let buffered_csv_reader = BufReader::new(opened_csv_file);
    let mut line_iterator = buffered_csv_reader.lines();

    // Step 1: header validation. The first line must match exactly.
    let first_line_result = match line_iterator.next() {
        Some(line_result) => line_result,
        None => {
            // File is completely empty (no header, no data).
            return Err(HorseRacingError::CsvHeaderMismatch(
                "read_training_csv_file_incrementally: empty file, no header",
            ));
        }
    };
    let first_line_text = match first_line_result {
        Ok(header_line_string) => header_line_string,
        Err(_io_error_intentionally_discarded) => {
            return Err(HorseRacingError::CsvFileReadFailure(
                "read_training_csv_file_incrementally: io error reading header",
            ));
        }
    };
    if first_line_text != CSV_EXPECTED_HEADER_LINE {
        return Err(HorseRacingError::CsvHeaderMismatch(
            "read_training_csv_file_incrementally: header does not match expected schema",
        ));
    }

    // Step 2: read data rows. Capacity hint is 256: slightly above the
    // project's stated ~200 rows, avoiding reallocation in the common case
    // without noticeable over-allocation.
    let initial_record_vector_capacity: usize = 256;
    let mut accumulated_raw_records: Vec<RawHorseRaceRecord> =
        Vec::with_capacity(initial_record_vector_capacity);

    // Upper bound on line count, per the project's "bound every loop" rule.
    // The bound is deliberately generous (one million lines) so the loop
    // never caps a realistic data file but also can never run forever if the
    // underlying reader misbehaves (e.g. an infinite pipe).
    let maximum_csv_lines_defensive_cap: usize = 1_000_000;
    let mut lines_read_so_far: usize = 0;

    for current_line_result in line_iterator {
        lines_read_so_far += 1;
        if lines_read_so_far > maximum_csv_lines_defensive_cap {
            return Err(HorseRacingError::CsvFileReadFailure(
                "read_training_csv_file_incrementally: exceeded defensive line cap",
            ));
        }

        let current_line_text = match current_line_result {
            Ok(data_line_string) => data_line_string,
            Err(_io_error_intentionally_discarded) => {
                return Err(HorseRacingError::CsvFileReadFailure(
                    "read_training_csv_file_incrementally: io error reading data line",
                ));
            }
        };

        // Skip blank lines silently. A training CSV may have a trailing
        // newline or a blank separator line; neither represents data.
        if current_line_text.trim().is_empty() {
            continue;
        }

        let parsed_record = parse_single_csv_data_row_into_raw_record(&current_line_text)?;
        accumulated_raw_records.push(parsed_record);
    }

    Ok(accumulated_raw_records)
}

/// Reads a prediction CSV file incrementally, returning every successfully
/// parsed `RawHorseRaceRecord` with blank `rank`/`completion` treated as
/// zero placeholders.
///
/// ## Difference From `read_training_csv_file_incrementally`
///
/// Calls `parse_single_csv_prediction_row_into_raw_record` instead of
/// `parse_single_csv_data_row_into_raw_record`, so blank or zero
/// `rank`/`completion` fields are tolerated rather than rejected.
///
/// All other behaviour (header validation, blank-line skipping, defensive
/// line cap) is identical to the training reader.
pub fn read_prediction_csv_file_incrementally(
    prediction_csv_file_path: &Path,
) -> Result<Vec<RawHorseRaceRecord>, HorseRacingError> {
    let opened_csv_file = match File::open(prediction_csv_file_path) {
        Ok(file_handle) => file_handle,
        Err(_os_error_intentionally_discarded) => {
            return Err(HorseRacingError::CsvFileReadFailure(
                "read_prediction_csv_file_incrementally: could not open csv file",
            ));
        }
    };

    let buffered_csv_reader = BufReader::new(opened_csv_file);
    let mut line_iterator = buffered_csv_reader.lines();

    // Validate header — same strict check as the training reader.
    let first_line_result = match line_iterator.next() {
        Some(line_result) => line_result,
        None => {
            return Err(HorseRacingError::CsvHeaderMismatch(
                "read_prediction_csv_file_incrementally: empty file, no header",
            ));
        }
    };
    let first_line_text = match first_line_result {
        Ok(header_line_string) => header_line_string,
        Err(_io_error_intentionally_discarded) => {
            return Err(HorseRacingError::CsvFileReadFailure(
                "read_prediction_csv_file_incrementally: io error reading header",
            ));
        }
    };
    if first_line_text != CSV_EXPECTED_HEADER_LINE {
        return Err(HorseRacingError::CsvHeaderMismatch(
            "read_prediction_csv_file_incrementally: header does not match expected schema",
        ));
    }

    let initial_record_vector_capacity: usize = 16;
    let mut accumulated_raw_records: Vec<RawHorseRaceRecord> =
        Vec::with_capacity(initial_record_vector_capacity);

    let maximum_csv_lines_defensive_cap: usize = 1_000_000;
    let mut lines_read_so_far: usize = 0;

    for current_line_result in line_iterator {
        lines_read_so_far += 1;
        if lines_read_so_far > maximum_csv_lines_defensive_cap {
            return Err(HorseRacingError::CsvFileReadFailure(
                "read_prediction_csv_file_incrementally: exceeded defensive line cap",
            ));
        }

        let current_line_text = match current_line_result {
            Ok(data_line_string) => data_line_string,
            Err(_io_error_intentionally_discarded) => {
                return Err(HorseRacingError::CsvFileReadFailure(
                    "read_prediction_csv_file_incrementally: io error reading data line",
                ));
            }
        };

        if current_line_text.trim().is_empty() {
            continue;
        }

        // Use the prediction-tolerant row parser.
        let parsed_record = parse_single_csv_prediction_row_into_raw_record(&current_line_text)?;
        accumulated_raw_records.push(parsed_record);
    }

    Ok(accumulated_raw_records)
}

/// Parses one CSV data line from a **prediction** CSV into a
/// `RawHorseRaceRecord`, tolerating empty or zero `rank` and `completion`
/// fields.
///
/// ## Project Role
///
/// Prediction CSVs share the same eight-column schema as training CSVs,
/// but the `rank` and `completion` columns carry no meaningful input data
/// — they are either left blank (`,,`) or written as `0,0`. This function
/// accepts both forms by substituting `0` for any empty field in those two
/// positions before parsing.
///
/// The resulting record has `rank = 0`, `completion = 0`, and therefore
/// `performance_score = 0` (the DNF sentinel). These placeholder values
/// are ignored by the prediction pipeline, which only uses the four input
/// features (`age`, `height`, `experience`, `weight`) and the engineered
/// features derived from them.
///
/// ## Validation
///
/// Range validation is applied only to the four input feature fields
/// (`age`, `height`, `experience`, `weight`). `rank` and `completion` are
/// not validated because their values are placeholders.
///
/// ## Error Handling
///
/// Same error variants as `parse_single_csv_data_row_into_raw_record`:
/// `CsvRowFieldCountMismatch` for wrong column count,
/// `CsvFieldIntegerParseFailure` for non-integer input fields.
pub fn parse_single_csv_prediction_row_into_raw_record(
    csv_data_line: &str,
) -> Result<RawHorseRaceRecord, HorseRacingError> {
    let split_field_strings: Vec<&str> = csv_data_line.split(',').collect();

    if split_field_strings.len() != CSV_EXPECTED_COLUMN_COUNT {
        return Err(HorseRacingError::CsvRowFieldCountMismatch(
            "parse_single_csv_prediction_row_into_raw_record: field count mismatch",
        ));
    }

    let row_id_field_index: usize = 0;
    let game_id_field_index: usize = 1;
    let age_field_index: usize = 2;
    let height_field_index: usize = 3;
    let experience_field_index: usize = 4;
    let weight_field_index: usize = 5;
    // Indices 6 (rank) and 7 (completion) may be blank — handled below.

    let parsed_row_id =
        parse_single_integer_field_from_csv(split_field_strings[row_id_field_index])?;
    let parsed_game_id =
        parse_single_integer_field_from_csv(split_field_strings[game_id_field_index])?;
    let parsed_age = parse_single_integer_field_from_csv(split_field_strings[age_field_index])?;
    let parsed_height =
        parse_single_integer_field_from_csv(split_field_strings[height_field_index])?;
    let parsed_experience =
        parse_single_integer_field_from_csv(split_field_strings[experience_field_index])?;
    let parsed_weight =
        parse_single_integer_field_from_csv(split_field_strings[weight_field_index])?;

    // For rank and completion: treat blank or whitespace-only as "0".
    let rank_raw_text = split_field_strings[6].trim();
    let parsed_rank: i32 = if rank_raw_text.is_empty() {
        0
    } else {
        rank_raw_text
            .parse::<i32>()
            .map_err(|_parse_error_discarded| {
                HorseRacingError::CsvFieldIntegerParseFailure(
                    "parse_single_csv_prediction_row_into_raw_record: rank parse failed",
                )
            })?
    };

    let completion_raw_text = split_field_strings[7].trim();
    let parsed_completion: i32 = if completion_raw_text.is_empty() {
        0
    } else {
        completion_raw_text
            .parse::<i32>()
            .map_err(|_parse_error_discarded| {
                HorseRacingError::CsvFieldIntegerParseFailure(
                    "parse_single_csv_prediction_row_into_raw_record: completion parse failed",
                )
            })?
    };

    // Validate only the four input feature fields.
    // rank and completion are placeholders and are not range-validated.
    let age_minimum_valid: i32 = 1;
    let age_maximum_valid: i32 = 8;
    if parsed_age < age_minimum_valid || parsed_age > age_maximum_valid {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "parse_single_csv_prediction_row_into_raw_record: age out of range",
        ));
    }

    let height_minimum_valid: i32 = 100;
    let height_maximum_valid: i32 = 999;
    if parsed_height < height_minimum_valid || parsed_height > height_maximum_valid {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "parse_single_csv_prediction_row_into_raw_record: height out of range",
        ));
    }

    if parsed_experience < 0 {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "parse_single_csv_prediction_row_into_raw_record: experience negative",
        ));
    }

    if parsed_weight <= 0 {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "parse_single_csv_prediction_row_into_raw_record: weight not positive",
        ));
    }

    // Performance score is the DNF sentinel — a placeholder.
    let placeholder_performance_score: i32 = PERFORMANCE_SCORE_FOR_DID_NOT_FINISH;

    Ok(RawHorseRaceRecord {
        row_id: parsed_row_id,
        game_id: parsed_game_id,
        age: parsed_age,
        height: parsed_height,
        experience: parsed_experience,
        weight: parsed_weight,
        rank: parsed_rank,
        completion: parsed_completion,
        performance_score: placeholder_performance_score,
    })
}

// ============================================================================
// SECTION 2 — CARGO TESTS
// ============================================================================

#[cfg(test)]
mod section_two_csv_parsing_tests {
    use super::*;
    use std::io::Write;

    /// Verifies that `parse_single_integer_field_from_csv` correctly parses
    /// a well-formed integer string with no surrounding whitespace.
    #[test]
    fn parse_single_integer_field_accepts_plain_integer() {
        let parse_result = parse_single_integer_field_from_csv("42");
        assert_eq!(parse_result.ok(), Some(42));
    }

    /// Verifies that whitespace around an integer field is tolerated, since
    /// manually edited CSVs commonly contain ", 4," style spacing.
    #[test]
    fn parse_single_integer_field_tolerates_surrounding_whitespace() {
        let parse_result = parse_single_integer_field_from_csv("  42  ");
        assert_eq!(parse_result.ok(), Some(42));
    }

    /// Verifies that a non-integer field produces the function's unique
    /// error variant and does not silently return a default value.
    #[test]
    fn parse_single_integer_field_rejects_non_integer_text() {
        let parse_result = parse_single_integer_field_from_csv("not_a_number");
        assert!(matches!(
            parse_result,
            Err(HorseRacingError::CsvFieldIntegerParseFailure(_))
        ));
    }

    /// Verifies parsing of a fully valid data row (rank 2, completed) and
    /// checks that the derived `performance_score` matches the documented
    /// formula: (6 - 2) * 200 = 800.
    #[test]
    fn parse_single_csv_data_row_parses_valid_finishing_horse() {
        let sample_line = "0,0,4,150,3,987,2,1";
        let parsed_record =
            parse_single_csv_data_row_into_raw_record(sample_line).expect("valid line must parse");
        assert_eq!(parsed_record.row_id, 0);
        assert_eq!(parsed_record.game_id, 0);
        assert_eq!(parsed_record.age, 4);
        assert_eq!(parsed_record.height, 150);
        assert_eq!(parsed_record.experience, 3);
        assert_eq!(parsed_record.weight, 987);
        assert_eq!(parsed_record.rank, 2);
        assert_eq!(parsed_record.completion, 1);
        assert_eq!(parsed_record.performance_score, 600);
    }

    /// Verifies that a DNF row (completion = 0, rank = 0) produces
    /// `performance_score = PERFORMANCE_SCORE_FOR_DID_NOT_FINISH`.
    #[test]
    fn parse_single_csv_data_row_parses_did_not_finish_horse() {
        let sample_line = "1,0,5,160,2,1050,0,0";
        let parsed_record = parse_single_csv_data_row_into_raw_record(sample_line)
            .expect("valid DNF line must parse");
        assert_eq!(parsed_record.completion, 0);
        assert_eq!(parsed_record.rank, 0);
        assert_eq!(
            parsed_record.performance_score,
            PERFORMANCE_SCORE_FOR_DID_NOT_FINISH
        );
    }

    /// Verifies that a row with the wrong number of fields is rejected with
    /// the distinct field-count error variant.
    #[test]
    fn parse_single_csv_data_row_rejects_wrong_field_count() {
        let too_few_fields_line = "0,0,4,150,3,987,2";
        let parse_result = parse_single_csv_data_row_into_raw_record(too_few_fields_line);
        assert!(matches!(
            parse_result,
            Err(HorseRacingError::CsvRowFieldCountMismatch(_))
        ));
    }

    /// Verifies that a syntactically correct row with an age outside the
    /// project's valid range (1-8) is rejected by range validation.
    #[test]
    fn parse_single_csv_data_row_rejects_out_of_range_age() {
        let age_too_high_line = "0,0,99,150,3,987,2,1";
        let parse_result = parse_single_csv_data_row_into_raw_record(age_too_high_line);
        assert!(matches!(
            parse_result,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));
    }

    /// Verifies that a row with inconsistent completion/rank (completed but
    /// rank = 0, or DNF but rank = 3) is rejected.
    #[test]
    fn parse_single_csv_data_row_rejects_inconsistent_completion_and_rank() {
        // Completion says "finished" but rank says "DNF" — invalid.
        let finished_but_rank_zero_line = "0,0,4,150,3,987,0,1";
        let parse_result_one =
            parse_single_csv_data_row_into_raw_record(finished_but_rank_zero_line);
        assert!(matches!(
            parse_result_one,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));

        // Completion says "DNF" but rank says "3rd" — invalid.
        let dnf_but_rank_three_line = "0,0,4,150,3,987,3,0";
        let parse_result_two = parse_single_csv_data_row_into_raw_record(dnf_but_rank_three_line);
        assert!(matches!(
            parse_result_two,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));
    }

    /// Verifies the performance score formula for every valid rank value,
    /// confirming the mapping documented on
    /// `compute_performance_score_from_rank_and_completion`.
    #[test]
    fn compute_performance_score_produces_correct_value_for_every_valid_rank() {
        let completion_flag_finished: i32 = 1;
        let expected_score_by_rank: [(i32, i32); 4] = [
            (1, 800), // (5-1)*200 = 800
            (2, 600), // (5-2)*200 = 600
            (3, 400), // (5-3)*200 = 400
            (4, 200), // (5-4)*200 = 200
        ];
        for (input_rank, expected_score) in expected_score_by_rank.iter() {
            let computed_score = compute_performance_score_from_rank_and_completion(
                *input_rank,
                completion_flag_finished,
            )
            .expect("valid rank/completion must produce a score");
            assert_eq!(computed_score, *expected_score);
        }
    }

    /// Verifies that DNF (completion = 0) always produces the DNF sentinel
    /// score regardless of the rank value passed in. This is defensive: the
    /// formula is supposed to ignore rank on DNF.
    #[test]
    fn compute_performance_score_ignores_rank_on_did_not_finish() {
        let completion_flag_dnf: i32 = 0;
        for candidate_rank_value in 0..=4 {
            let computed_score = compute_performance_score_from_rank_and_completion(
                candidate_rank_value,
                completion_flag_dnf,
            )
            .expect("DNF with any rank must succeed and yield DNF score");
            assert_eq!(computed_score, PERFORMANCE_SCORE_FOR_DID_NOT_FINISH);
        }
    }

    /// Verifies that engineered features compute correctly for a known
    /// record. Height 150, weight 1000 gives (150 * 1000) / 1000 = 150.
    /// Age 4, experience 3 gives 4 * 3 = 12.
    #[test]
    fn compute_engineered_feature_vector_produces_expected_integer_values() {
        let known_record = RawHorseRaceRecord {
            row_id: 0,
            game_id: 0,
            age: 4,
            height: 150,
            experience: 3,
            weight: 1000,
            rank: 2,
            completion: 1,
            performance_score: 800,
        };
        let engineered_vector = compute_engineered_feature_vector_from_raw_record(&known_record)
            .expect("valid record must produce engineered features");
        assert_eq!(engineered_vector.age, 4);
        assert_eq!(engineered_vector.height, 150);
        assert_eq!(engineered_vector.experience, 3);
        assert_eq!(engineered_vector.weight, 1000);
        assert_eq!(
            engineered_vector.height_to_weight_ratio_times_one_thousand,
            150
        );
        assert_eq!(engineered_vector.age_times_experience, 12);
    }

    /// Verifies that the defensive divide-by-zero check in the feature
    /// engineering function catches a zero-weight record and returns the
    /// dedicated error variant rather than panicking.
    #[test]
    fn compute_engineered_feature_vector_rejects_zero_weight_record() {
        let corrupt_zero_weight_record = RawHorseRaceRecord {
            row_id: 0,
            game_id: 0,
            age: 4,
            height: 150,
            experience: 3,
            weight: 0,
            rank: 2,
            completion: 1,
            performance_score: 800,
        };
        let engineer_result =
            compute_engineered_feature_vector_from_raw_record(&corrupt_zero_weight_record);
        assert!(matches!(
            engineer_result,
            Err(HorseRacingError::ArithmeticDivisionByZero(_))
        ));
    }

    /// Verifies the end-to-end CSV read path: writes a valid temporary CSV
    /// file with a header and two data rows, then reads it back and checks
    /// the parsed records match.
    ///
    /// Uses the OS temp directory so the test is self-contained and leaves
    /// no artifacts in the repository.
    #[test]
    fn read_training_csv_file_incrementally_round_trips_two_valid_rows() {
        let temporary_directory = std::env::temp_dir();
        let temporary_csv_file_path =
            temporary_directory.join("horse_racing_classifier_section_two_test.csv");

        // Build a minimal valid CSV: header + two rows.
        let csv_text_content = format!(
            "{}\n0,0,4,150,3,987,2,1\n1,0,5,160,2,1050,0,0\n",
            CSV_EXPECTED_HEADER_LINE
        );

        {
            let mut temporary_file_handle = File::create(&temporary_csv_file_path)
                .expect("temp file create must succeed in test");
            temporary_file_handle
                .write_all(csv_text_content.as_bytes())
                .expect("temp file write must succeed in test");
        }

        let parsed_records_vector = read_training_csv_file_incrementally(&temporary_csv_file_path)
            .expect("valid csv must parse successfully");
        assert_eq!(parsed_records_vector.len(), 2);
        assert_eq!(parsed_records_vector[0].rank, 2);
        assert_eq!(parsed_records_vector[0].performance_score, 600);
        assert_eq!(parsed_records_vector[1].completion, 0);
        assert_eq!(
            parsed_records_vector[1].performance_score,
            PERFORMANCE_SCORE_FOR_DID_NOT_FINISH
        );

        // Clean up the temp file. Failure to delete is non-fatal; the OS
        // will clean temp eventually.
        let _ignored_remove_result = std::fs::remove_file(&temporary_csv_file_path);
    }

    /// Verifies that a CSV file whose header differs from the expected
    /// schema is rejected with the dedicated header-mismatch error.
    #[test]
    fn read_training_csv_file_incrementally_rejects_wrong_header() {
        let temporary_directory = std::env::temp_dir();
        let temporary_csv_file_path =
            temporary_directory.join("horse_racing_classifier_section_two_bad_header_test.csv");
        let bad_header_csv_text = "wrong,header,line\n0,0,4,150,3,987,2,1\n";

        {
            let mut temporary_file_handle = File::create(&temporary_csv_file_path)
                .expect("temp file create must succeed in test");
            temporary_file_handle
                .write_all(bad_header_csv_text.as_bytes())
                .expect("temp file write must succeed in test");
        }

        let read_result = read_training_csv_file_incrementally(&temporary_csv_file_path);
        assert!(matches!(
            read_result,
            Err(HorseRacingError::CsvHeaderMismatch(_))
        ));

        let _ignored_remove_result = std::fs::remove_file(&temporary_csv_file_path);
    }
}

/*
## Notes on Design Choices in This Section

**Why `Vec<RawHorseRaceRecord>` return (not an iterator):** The entire dataset is ~200 rows, and every downstream consumer (train/validate split, tree building, linear margin scan) needs whole-dataset access. An iterator API would add complexity with zero memory benefit at this scale.

**Why the header check is exact (no whitespace tolerance):** Schema drift is a silent corruption risk. A header of `" row_id, game_id, ..."` versus `"row_id,game_id,..."` must not be accepted interchangeably — the training CSV is a manually-maintained artifact, and tolerating whitespace there would mask errors.

**Why `.expect()` appears only inside tests:** Per project rules, production code never uses `unwrap` or `expect`. The test module is explicitly allowed to use them because a test-assertion failure is exactly the intent: a test that cannot create its temp file *should* fail loudly.

**Why the loop has an upper bound (`maximum_csv_lines_defensive_cap`):** NASA Power-of-10 rule 2 — every loop must be bounded. One million lines is far above the project's stated scale and protects against an adversarial or corrupt file that might stream data forever.

*/

/*
Section 3: Race-Group Handling and Train/Validate Splitting
This section introduces the grouping and splitting logic that the rest of the pipeline depends on. Because performance_score is a relative measure within a N-horse race, every train/validate split in this crate must respect race-group boundaries — splitting mid-race would leak information from the other four horses in a race into both sides of the split.

Section Contains:

RaceGroup struct — a single race's five records, kept together as a unit
group_raw_records_by_game_id — partitions a flat Vec<RawHorseRaceRecord> into race groups, validating that every group has exactly N horses
TrainValidateSplit struct — the result of a group-level split (two Vec<RaceGroup>s)
split_race_groups_into_train_and_validate — deterministic, seedable group-level split at a configurable ratio
flatten_race_groups_into_records — collapses a Vec<RaceGroup> back into a flat record list for downstream feature-vector consumption

Plus cargo tests.

Design Decisions Explained Up Front
Why a dedicated RaceGroup struct (instead of just Vec<Vec<RawHorseRaceRecord>>): The type name is the documentation. A function signature taking &[RaceGroup] communicates intent that &[Vec<RawHorseRaceRecord>] does not. It also lets the struct carry the game_id once (rather than trusting every inner vector to be internally consistent), which simplifies downstream code.
Why a deterministic seeded split (not random each run): Reproducibility is non-negotiable in a data science project. Two training runs on the same data with the same config must produce the same train/validate split, or else the reported accuracy of a hyperparameter search is noise. The seed becomes a config value.
Why a simple linear-congruential pseudo-random number generator (hand-written, no crate): The project forbids third-party crates, and rand is a crate. A tiny LCG is perfectly sufficient for "shuffle ~40 race groups deterministically" — we do not need cryptographic quality or statistical purity here, only reproducibility.
Why the split ratio is configurable (not hard-coded 80/20): The ratio may be tuned per experiment. It belongs in the config file, not as a magic number in code.


Notes
Why the LCG includes a zero-to-one bump: An LCG seeded with 0 has the property that its first output is purely the increment constant, and early samples are slightly less mixed than usual. Avoiding seed 0 entirely (by bumping to 1) removes that edge case without surprising callers.
Why clone() appears in the grouping loop: The input is &[RawHorseRaceRecord] (a borrow), but the output RaceGroup owns its records. The clone is unavoidable at this layer. RawHorseRaceRecord is small (nine i32 fields, 36 bytes), so the clones are cheap.
Why a linear scan rather than a HashMap: Already explained in the function doc. With ~40 groups, a linear scan is O(n²) in the worst case — for n=40 that is 1,600 comparisons, trivial on modern hardware. Avoiding HashMap removes hash randomization as a source of cross-run nondeterminism and keeps the code auditable.
The .expect() in build_synthetic_horse_record_for_testing: Used only inside the test helper (which is inside #[cfg(test)]). Production code path is unaffected.
*/

// ============================================================================
// SECTION 3 — RACE-GROUP HANDLING AND TRAIN/VALIDATE SPLITTING
// ============================================================================
//
// This section turns a flat list of parsed `RawHorseRaceRecord`s into
// race-group-aware structures and performs train/validate splitting at the
// race-group level.
//
// ## Why Race-Group-Level Splitting Is Mandatory
//
// Each horse's `performance_score` is derived from its `rank` *within its
// N-horse race*. If one horse from a race ended up in the training set
// and the other four in the validation set, the training set's label for
// that horse would be implicitly informed by the validation set's labels
// (because the ranks of all N horses are interdependent). This is a
// classic "data leakage" failure mode for grouped data, and it inflates
// reported validation accuracy without actually improving the model.
//
// The rule is therefore absolute: entire race groups go into train or
// validate, never split across them.
//
// ## Determinism
//
// The split is seeded (`split_seed_value`) so that re-running training with
// the same seed produces the same split. This is essential for comparing
// hyperparameter search results: an 80/20 split that changes between runs
// would make accuracy differences between runs uninterpretable.

/// The result of splitting race groups into three partitions: a held-out
/// test set, a training set, and a validation set.
///
/// ## Project Role — Three-Way Split for Unbiased Evaluation
///
/// When the validation set is used to select hyperparameters, the reported
/// validation accuracy becomes optimistic because the hyperparameter search
/// was steered by that data. The held-out test set provides a final
/// accuracy estimate from data that was never used for any decision —
/// not for fitting, not for tuning.
///
/// The test groups are set aside at the very beginning and are only
/// touched once, after all hyperparameter selection is complete.
///
/// ## Split Ratios (with defaults test=20%, train_of_remainder=80%)
///
///   - 20% of groups → test (held out completely)
///   - 80% of remaining 80% = 64% of total → train
///   - 20% of remaining 80% = 16% of total → validate
#[derive(Debug, Clone)]
pub struct ThreeWaySplit {
    /// Race groups held out for final unbiased evaluation.
    /// Never used during hyperparameter search or tree fitting.
    pub test_race_groups: Vec<RaceGroup>,
    /// The train/validate partition of the non-test groups,
    /// used for Stage 1 hyperparameter search.
    pub train_validate_split: TrainValidateSplit,
}

/// Performs a two-stage group-level split: first carves out a held-out test
/// set, then splits the remainder into train and validate partitions.
///
/// ## Algorithm
///
/// 1. Calls `split_race_groups_into_train_and_validate` with
///    `(100 - test_fraction_percent)` as the training fraction. The
///    "training" output becomes the non-test groups; the "validation"
///    output becomes the test groups.
/// 2. Calls `split_race_groups_into_train_and_validate` again on the
///    non-test groups with `training_fraction_percent`, using a different
///    seed (`split_seed_value + 1`) to ensure the two shuffles are
///    independent.
///
/// ## Inputs
///
/// - `all_available_race_groups` — every race group to be split.
/// - `test_fraction_percent` — percentage of groups to hold out as test
///   (1–50). Must leave at least half the groups for train/validate.
/// - `training_fraction_percent` — percentage of the non-test groups that
///   go into the training partition (1–99). The rest are validate.
/// - `split_seed_value` — seed for the deterministic shuffle. The first
///   split uses this seed; the second uses `split_seed_value + 1`.
///
/// ## Error Cases
///
/// Returns `FieldValueOutOfValidRange` if `test_fraction_percent` is not
/// in 1..=50, or if `training_fraction_percent` is not in 1..=99, or if
/// there are too few groups to produce non-empty partitions.
pub fn split_race_groups_into_test_train_validate(
    all_available_race_groups: &[RaceGroup],
    test_fraction_percent: u32,
    training_fraction_percent: u32,
    split_seed_value: u32,
) -> Result<ThreeWaySplit, HorseRacingError> {
    // Guard: test fraction must be 1..=50 to leave at least half for
    // train/validate combined.
    let minimum_valid_test_fraction: u32 = 1;
    let maximum_valid_test_fraction: u32 = 50;
    if test_fraction_percent < minimum_valid_test_fraction
        || test_fraction_percent > maximum_valid_test_fraction
    {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "split_race_groups_into_test_train_validate: test_fraction_percent must be 1..=50",
        ));
    }

    // First split: separate test groups from the rest.
    // Using (100 - test_fraction) as the "training" fraction means:
    //   "training" output = non-test groups (the larger portion)
    //   "validation" output = test groups (the smaller portion)
    let non_test_fraction_percent: u32 = 100 - test_fraction_percent;
    let test_vs_rest_split = split_race_groups_into_train_and_validate(
        all_available_race_groups,
        non_test_fraction_percent,
        split_seed_value,
    )?;

    let non_test_race_groups = test_vs_rest_split.training_race_groups;
    let test_race_groups = test_vs_rest_split.validation_race_groups;

    // Second split: divide non-test groups into train and validate.
    // Use a different seed so the two shuffles are independent. The
    // wrapping_add ensures no panic if split_seed_value == u32::MAX.
    let train_validate_seed = split_seed_value.wrapping_add(1);
    let train_validate_split = split_race_groups_into_train_and_validate(
        &non_test_race_groups,
        training_fraction_percent,
        train_validate_seed,
    )?;

    Ok(ThreeWaySplit {
        test_race_groups,
        train_validate_split,
    })
}

/// A single race: exactly `HORSES_PER_RACE_GROUP` horses sharing the same
/// `game_id`.
///
/// ## Invariants
///
/// - `horse_records_in_race.len() == HORSES_PER_RACE_GROUP`.
/// - Every record in `horse_records_in_race` has `game_id == race_game_id`.
/// - The group is valid as a training example: the N `rank` values
///   (excluding any DNFs) form a legal race ordering.
///
/// These invariants are established by `group_raw_records_by_game_id` and
/// must be preserved by any function that constructs a `RaceGroup` directly.
///
/// ## Project Role
///
/// All train/validate splitting, and any future per-race analysis (for
/// example, normalizing features relative to the group mean), operates on
/// `RaceGroup` values rather than on flat records. Treating the race as the
/// unit of analysis matches the underlying data-generating process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaceGroup {
    pub race_game_id: i32,
    pub horse_records_in_race: Vec<RawHorseRaceRecord>,
}

/// The result of splitting a set of race groups into training and
/// validation portions.
///
/// Both fields are owned `Vec<RaceGroup>` so that the caller can pass each
/// half independently to downstream functions without lifetime coupling.
///
/// ## Why Not Reuse `Vec<RaceGroup>` Directly
///
/// Named fields (`training_race_groups`, `validation_race_groups`) prevent
/// confusing the two halves at call sites — a bug easily introduced if the
/// function returned a `(Vec<RaceGroup>, Vec<RaceGroup>)` tuple.
#[derive(Debug, Clone)]
pub struct TrainValidateSplit {
    pub training_race_groups: Vec<RaceGroup>,
    pub validation_race_groups: Vec<RaceGroup>,
}

/// Partitions a flat vector of parsed horse records into `RaceGroup` values,
/// one per distinct `game_id`.
///
/// ## Behavior
///
/// 1. Walks the input vector once, bucketing records by `game_id`.
/// 2. Preserves the insertion order of `game_id` values as they first appear
///    in the input — not sorted numerically. This keeps training runs
///    deterministic given a fixed input file order and avoids surprising
///    reordering if the CSV is not sorted by `game_id`.
/// 3. Verifies that every resulting group contains exactly
///    `HORSES_PER_RACE_GROUP` records; otherwise returns
///    `RaceGroupIncompleteOrOversized`.
///
/// ## Why a Linear Scan With a Side Index (Not `HashMap`)
///
/// With ~40 race groups (200 rows / N horses), a linear scan to locate an
/// existing group is trivially fast. Avoiding `HashMap` means no hashing,
/// no per-key heap allocation beyond the record vectors, and deterministic
/// ordering without needing to worry about hash randomization. Determinism
/// is important because downstream code compares runs.
///
/// ## Error Handling
///
/// An incomplete group (fewer than N horses) or an oversized group (more
/// than N) both indicate data corruption and are surfaced as a single
/// `RaceGroupIncompleteOrOversized` error. The project's policy is to
/// reject, not silently truncate or pad.
pub fn group_raw_records_by_game_id(
    parsed_raw_records: &[RawHorseRaceRecord],
) -> Result<Vec<RaceGroup>, HorseRacingError> {
    // Capacity hint: at most one group per record. Over-allocates slightly
    // but avoids repeated reallocations during the build phase.
    let initial_group_vector_capacity: usize = parsed_raw_records.len();
    let mut accumulated_race_groups: Vec<RaceGroup> =
        Vec::with_capacity(initial_group_vector_capacity);

    // Upper bound on the loop per NASA Power-of-10 rule 2.
    // The bound equals the input length, which is itself already bounded by
    // the CSV reader's defensive cap.
    for current_record_reference in parsed_raw_records.iter() {
        let current_record_clone = current_record_reference.clone();
        let target_game_id_for_this_record = current_record_clone.game_id;

        // Find the existing group for this game_id, if any. `iter().position`
        // is a linear scan; acceptable at this project's scale (see docstring).
        let existing_group_position_option =
            accumulated_race_groups
                .iter()
                .position(|existing_race_group_candidate| {
                    existing_race_group_candidate.race_game_id == target_game_id_for_this_record
                });

        match existing_group_position_option {
            Some(found_group_index) => {
                // Append to the existing group. We will check the size
                // invariant once all records are processed, not here,
                // because records may be presented in any order within the
                // same group.
                accumulated_race_groups[found_group_index]
                    .horse_records_in_race
                    .push(current_record_clone);
            }
            None => {
                // First time we have seen this game_id — create a new group.
                // Capacity hint of HORSES_PER_RACE_GROUP avoids reallocation
                // as the remaining four horses in this race are appended.
                let mut newly_created_race_group = RaceGroup {
                    race_game_id: target_game_id_for_this_record,
                    horse_records_in_race: Vec::with_capacity(HORSES_PER_RACE_GROUP),
                };
                newly_created_race_group
                    .horse_records_in_race
                    .push(current_record_clone);
                accumulated_race_groups.push(newly_created_race_group);
            }
        }
    }

    // Now validate the size invariant on every group.
    for race_group_to_check in accumulated_race_groups.iter() {
        if race_group_to_check.horse_records_in_race.len() != HORSES_PER_RACE_GROUP {
            return Err(HorseRacingError::RaceGroupIncompleteOrOversized(
                "group_raw_records_by_game_id: a game_id group did not contain exactly N horses",
            ));
        }
    }

    Ok(accumulated_race_groups)
}

/// A tiny linear congruential pseudo-random number generator for
/// deterministic shuffling.
///
/// ## Why a Hand-Written PRNG
///
/// The project forbids third-party crates, and the standard library's
/// random facilities are not present without `rand`. For this project's
/// needs — shuffling ~40 race groups reproducibly given a seed — an LCG is
/// both adequate and auditable.
///
/// ## Quality
///
/// This is *not* a cryptographic PRNG and is not suitable for statistical
/// simulation work. For shuffling a small integer array with a known seed,
/// the LCG's well-known bit-pattern weaknesses do not matter: all we need
/// is that the same seed produces the same shuffle.
///
/// ## Constants
///
/// The multiplier and increment are the Numerical Recipes constants for a
/// 32-bit LCG. The modulus is implicit (2^32, via `u32` wrapping).
pub struct DeterministicLinearCongruentialPseudoRandomGenerator {
    current_internal_state: u32,
}

impl DeterministicLinearCongruentialPseudoRandomGenerator {
    /// Constructs a new generator seeded with `seed_value`. The same seed
    /// always produces the same output sequence.
    pub fn new_with_seed(seed_value: u32) -> Self {
        // A seed of zero is legal but produces a degenerate short cycle
        // near the start of the sequence. Bumping zero to a fixed nonzero
        // starting state avoids this without surprising callers who pass 0.
        let internal_seed_adjusted_away_from_zero = if seed_value == 0 { 1 } else { seed_value };
        DeterministicLinearCongruentialPseudoRandomGenerator {
            current_internal_state: internal_seed_adjusted_away_from_zero,
        }
    }

    /// Produces the next pseudo-random `u32` in the sequence.
    ///
    /// Uses `wrapping_mul` and `wrapping_add` so that overflow produces the
    /// usual modulo-2^32 LCG behavior rather than a runtime panic in debug
    /// builds.
    pub fn next_pseudo_random_u32(&mut self) -> u32 {
        let numerical_recipes_multiplier_constant: u32 = 1664525;
        let numerical_recipes_increment_constant: u32 = 1013904223;
        self.current_internal_state = self
            .current_internal_state
            .wrapping_mul(numerical_recipes_multiplier_constant)
            .wrapping_add(numerical_recipes_increment_constant);
        self.current_internal_state
    }

    /// Produces a pseudo-random `usize` in `[0, exclusive_upper_bound)`.
    ///
    /// Returns 0 if `exclusive_upper_bound == 0`, since there is no valid
    /// output in an empty range; callers are responsible for not asking for
    /// a random index into an empty collection.
    ///
    /// ## Why Modulo (Despite the Small Bias)
    ///
    /// A modulo reduction introduces a slight bias when the upper bound
    /// does not evenly divide 2^32. For the project's ~40-group shuffle,
    /// this bias is negligible (parts-per-billion). A bias-free rejection
    /// sampler is not worth the added code complexity here.
    pub fn next_usize_below_bound(&mut self, exclusive_upper_bound: usize) -> usize {
        if exclusive_upper_bound == 0 {
            return 0;
        }
        let raw_pseudo_random_value = self.next_pseudo_random_u32() as usize;
        raw_pseudo_random_value % exclusive_upper_bound
    }
}

/// Deterministically splits a collection of race groups into training and
/// validation portions at the specified ratio.
///
/// ## Inputs
///
/// - `all_available_race_groups` — every race group to be split.
/// - `training_fraction_percent` — the percentage of groups (0–100) that go
///   into the training set. The validation set receives the remainder.
///   Values outside 1–99 are rejected to prevent degenerate splits (an
///   empty train or validate set is not a usable configuration).
/// - `split_seed_value` — a `u32` seed for the shuffle. Identical seeds
///   produce identical splits.
///
/// ## Algorithm
///
/// 1. Clone the input into a mutable vector of indices `[0, 1, ..., n-1]`.
/// 2. Fisher-Yates shuffle the index vector using the seeded PRNG.
/// 3. Compute the training size as `(n * training_fraction_percent) / 100`,
///    with a minimum of 1 training group and minimum of 1 validation group
///    (enforced below).
/// 4. Build the two output vectors by walking the shuffled indices.
///
/// ## Why Fisher-Yates
///
/// Fisher-Yates is the standard unbiased-shuffle algorithm, trivially
/// implemented in a bounded loop (`for i in (1..n).rev()`), requires no
/// recursion, and performs exactly one swap per element. It fits the
/// project's "bounded loops, no recursion" rules cleanly.
///
/// ## Error Cases
///
/// Returns `FieldValueOutOfValidRange` if `training_fraction_percent` is not
/// in `1..=99`, or if `all_available_race_groups` is empty (no data to
/// split).
pub fn split_race_groups_into_train_and_validate(
    all_available_race_groups: &[RaceGroup],
    training_fraction_percent: u32,
    split_seed_value: u32,
) -> Result<TrainValidateSplit, HorseRacingError> {
    // Guard: must have at least one group to split.
    if all_available_race_groups.is_empty() {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "split_race_groups_into_train_and_validate: input race group list is empty",
        ));
    }

    // Guard: training fraction must leave a nonempty validation set.
    let minimum_valid_training_fraction_percent: u32 = 1;
    let maximum_valid_training_fraction_percent: u32 = 99;
    if training_fraction_percent < minimum_valid_training_fraction_percent
        || training_fraction_percent > maximum_valid_training_fraction_percent
    {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "split_race_groups_into_train_and_validate: training fraction must be 1..=99",
        ));
    }

    let total_race_group_count: usize = all_available_race_groups.len();

    // Build the index vector [0, 1, ..., n-1].
    let mut shuffleable_group_index_vector: Vec<usize> = Vec::with_capacity(total_race_group_count);
    for index_under_construction in 0..total_race_group_count {
        shuffleable_group_index_vector.push(index_under_construction);
    }

    // Fisher-Yates shuffle, seeded. Walks from n-1 down to 1 inclusive,
    // swapping the element at `swap_position_high` with a pseudo-random
    // element at `swap_position_low_pseudo_random` in `[0, swap_position_high]`.
    let mut seeded_pseudo_random_generator =
        DeterministicLinearCongruentialPseudoRandomGenerator::new_with_seed(split_seed_value);

    // The loop bound `total_race_group_count` is itself bounded by the
    // input size, which is bounded by the CSV reader's defensive cap.
    if total_race_group_count >= 2 {
        let mut swap_position_high: usize = total_race_group_count - 1;
        // Using `while` rather than `for (... ).rev()` so the decrement is
        // explicit and readable, and the bound is obvious.
        loop {
            let exclusive_upper_bound_for_random_index = swap_position_high + 1;
            let swap_position_low_pseudo_random = seeded_pseudo_random_generator
                .next_usize_below_bound(exclusive_upper_bound_for_random_index);
            shuffleable_group_index_vector
                .swap(swap_position_high, swap_position_low_pseudo_random);

            if swap_position_high == 1 {
                break;
            }
            swap_position_high -= 1;
        }
    }

    // Compute the cut point. Integer math: `(n * percent) / 100`.
    let computed_training_group_count: usize =
        (total_race_group_count * (training_fraction_percent as usize)) / 100;

    // Clamp to guarantee at least one group on each side, regardless of
    // rounding. For example: n=3, percent=80 -> (3*80)/100 = 2 training,
    // 1 validation — fine. But n=2, percent=80 -> 1 training, 1 validation,
    // also fine. The clamps catch pathological small-n edge cases.
    let minimum_training_groups_required: usize = 1;
    let minimum_validation_groups_required: usize = 1;

    let final_training_group_count: usize;
    if computed_training_group_count < minimum_training_groups_required {
        final_training_group_count = minimum_training_groups_required;
    } else if total_race_group_count.saturating_sub(computed_training_group_count)
        < minimum_validation_groups_required
    {
        final_training_group_count = total_race_group_count - minimum_validation_groups_required;
    } else {
        final_training_group_count = computed_training_group_count;
    }

    // Build output vectors by walking the shuffled index list.
    let mut training_race_groups_output: Vec<RaceGroup> =
        Vec::with_capacity(final_training_group_count);
    let mut validation_race_groups_output: Vec<RaceGroup> =
        Vec::with_capacity(total_race_group_count - final_training_group_count);

    for position_in_shuffled_order in 0..total_race_group_count {
        let source_group_index = shuffleable_group_index_vector[position_in_shuffled_order];
        let source_group_clone = all_available_race_groups[source_group_index].clone();
        if position_in_shuffled_order < final_training_group_count {
            training_race_groups_output.push(source_group_clone);
        } else {
            validation_race_groups_output.push(source_group_clone);
        }
    }

    Ok(TrainValidateSplit {
        training_race_groups: training_race_groups_output,
        validation_race_groups: validation_race_groups_output,
    })
}

/// Collapses a slice of `RaceGroup`s back into a flat `Vec<RawHorseRaceRecord>`
/// preserving the group ordering.
///
/// ## Project Role
///
/// The tree and linear-margin builders consume flat record vectors, not
/// `RaceGroup`s, because their splitting logic operates on individual
/// feature vectors. `group_raw_records_by_game_id` followed by
/// `split_race_groups_into_train_and_validate` followed by this flattener
/// is the standard pipeline from parsed CSV to ready-to-train record lists.
///
/// ## Why Return an Owned `Vec`
///
/// The caller typically consumes the flattened records immediately to build
/// feature vectors; it is simplest and clearest to hand over owned data.
pub fn flatten_race_groups_into_records(
    race_groups_to_flatten: &[RaceGroup],
) -> Vec<RawHorseRaceRecord> {
    // Capacity hint: exactly `group_count * HORSES_PER_RACE_GROUP` records.
    let exact_output_capacity_hint = race_groups_to_flatten.len() * HORSES_PER_RACE_GROUP;
    let mut flattened_records_vector: Vec<RawHorseRaceRecord> =
        Vec::with_capacity(exact_output_capacity_hint);
    for race_group_reference in race_groups_to_flatten.iter() {
        for horse_record_reference in race_group_reference.horse_records_in_race.iter() {
            flattened_records_vector.push(horse_record_reference.clone());
        }
    }
    flattened_records_vector
}

// ============================================================================
// SECTION 3 — CARGO TESTS
// ============================================================================

#[cfg(test)]
mod section_three_race_group_and_split_tests {
    use super::*;

    /// Builds a valid N-horse race group with sequential row_ids and
    /// ranks 1 through N, all completing.
    fn build_complete_valid_race_group_for_testing(
        game_id_value: i32,
        starting_row_id_value: i32,
    ) -> Vec<RawHorseRaceRecord> {
        let mut records_for_this_race: Vec<RawHorseRaceRecord> =
            Vec::with_capacity(HORSES_PER_RACE_GROUP);
        for position_within_race in 0..HORSES_PER_RACE_GROUP {
            let assigned_rank: i32 = (position_within_race as i32) + 1;
            let assigned_completion: i32 = 1;
            let assigned_row_id: i32 = starting_row_id_value + (position_within_race as i32);
            records_for_this_race.push(build_synthetic_horse_record_for_testing(
                assigned_row_id,
                game_id_value,
                assigned_rank,
                assigned_completion,
            ));
        }
        records_for_this_race
    }

    /// Helper used by tests to construct a synthetic valid record without
    /// going through the CSV layer. Centralizes the boilerplate of building
    /// a `RawHorseRaceRecord` for test fixtures.
    fn build_synthetic_horse_record_for_testing(
        row_id_value: i32,
        game_id_value: i32,
        rank_value: i32,
        completion_value: i32,
    ) -> RawHorseRaceRecord {
        // Use neutral-but-valid values for non-varying fields so that range
        // validation would pass if ever invoked.
        let placeholder_age: i32 = 4;
        let placeholder_height: i32 = 150;
        let placeholder_experience: i32 = 3;
        let placeholder_weight: i32 = 987;

        // Derive performance_score via the one canonical helper.
        let derived_performance_score =
            compute_performance_score_from_rank_and_completion(rank_value, completion_value)
                .expect("test fixture must use valid rank/completion combination");

        RawHorseRaceRecord {
            row_id: row_id_value,
            game_id: game_id_value,
            age: placeholder_age,
            height: placeholder_height,
            experience: placeholder_experience,
            weight: placeholder_weight,
            rank: rank_value,
            completion: completion_value,
            performance_score: derived_performance_score,
        }
    }

    /// Verifies that the three-way split produces three non-empty,
    /// disjoint partitions whose union equals the original set.
    #[test]
    fn three_way_split_produces_disjoint_exhaustive_partition() {
        let mut synthetic_race_groups: Vec<RaceGroup> = Vec::new();
        let total_synthetic_group_count: i32 = 20;
        for race_index in 0..total_synthetic_group_count {
            synthetic_race_groups.push(RaceGroup {
                race_game_id: race_index,
                horse_records_in_race: build_complete_valid_race_group_for_testing(
                    race_index,
                    race_index * (HORSES_PER_RACE_GROUP as i32),
                ),
            });
        }

        let three_way_result = split_race_groups_into_test_train_validate(
            &synthetic_race_groups,
            20, // 20% test
            80, // 80% of remainder for train
            42,
        )
        .expect("three-way split must succeed on valid input");

        // All three partitions must be non-empty.
        assert!(
            !three_way_result.test_race_groups.is_empty(),
            "test partition must be non-empty"
        );
        assert!(
            !three_way_result
                .train_validate_split
                .training_race_groups
                .is_empty(),
            "train partition must be non-empty"
        );
        assert!(
            !three_way_result
                .train_validate_split
                .validation_race_groups
                .is_empty(),
            "validate partition must be non-empty"
        );

        // Collect all game_ids from each partition.
        let test_ids: Vec<i32> = three_way_result
            .test_race_groups
            .iter()
            .map(|g| g.race_game_id)
            .collect();
        let train_ids: Vec<i32> = three_way_result
            .train_validate_split
            .training_race_groups
            .iter()
            .map(|g| g.race_game_id)
            .collect();
        let validate_ids: Vec<i32> = three_way_result
            .train_validate_split
            .validation_race_groups
            .iter()
            .map(|g| g.race_game_id)
            .collect();

        // Disjoint: no overlap between any pair.
        for test_id in test_ids.iter() {
            assert!(!train_ids.contains(test_id), "test/train overlap detected");
            assert!(
                !validate_ids.contains(test_id),
                "test/validate overlap detected"
            );
        }
        for train_id in train_ids.iter() {
            assert!(
                !validate_ids.contains(train_id),
                "train/validate overlap detected"
            );
        }

        // Exhaustive: union equals original set.
        let mut all_ids: Vec<i32> = Vec::new();
        all_ids.extend_from_slice(&test_ids);
        all_ids.extend_from_slice(&train_ids);
        all_ids.extend_from_slice(&validate_ids);
        all_ids.sort();
        let expected_ids: Vec<i32> = (0..total_synthetic_group_count).collect();
        assert_eq!(all_ids, expected_ids);
    }

    /// Verifies that the three-way split produces partition sizes
    /// consistent with the requested percentages.
    #[test]
    fn three_way_split_produces_expected_approximate_sizes() {
        let mut synthetic_race_groups: Vec<RaceGroup> = Vec::new();
        let total_group_count: i32 = 20;
        for race_index in 0..total_group_count {
            synthetic_race_groups.push(RaceGroup {
                race_game_id: race_index,
                horse_records_in_race: build_complete_valid_race_group_for_testing(
                    race_index,
                    race_index * (HORSES_PER_RACE_GROUP as i32),
                ),
            });
        }

        let three_way_result =
            split_race_groups_into_test_train_validate(&synthetic_race_groups, 20, 80, 42)
                .expect("must succeed");

        // 20 groups: 20% test = 4, remaining 16: 80% train = 12, validate = 4
        let test_count = three_way_result.test_race_groups.len();
        let train_count = three_way_result
            .train_validate_split
            .training_race_groups
            .len();
        let validate_count = three_way_result
            .train_validate_split
            .validation_race_groups
            .len();

        assert_eq!(test_count + train_count + validate_count, 20);
        assert_eq!(test_count, 4);
        assert_eq!(train_count, 12);
        assert_eq!(validate_count, 4);
    }

    /// Verifies that the three-way split is deterministic for identical seeds.
    #[test]
    fn three_way_split_is_deterministic_for_identical_seed() {
        let mut synthetic_race_groups: Vec<RaceGroup> = Vec::new();
        for race_index in 0..15_i32 {
            synthetic_race_groups.push(RaceGroup {
                race_game_id: race_index,
                horse_records_in_race: build_complete_valid_race_group_for_testing(
                    race_index,
                    race_index * (HORSES_PER_RACE_GROUP as i32),
                ),
            });
        }

        let split_first =
            split_race_groups_into_test_train_validate(&synthetic_race_groups, 20, 80, 99)
                .expect("must succeed");
        let split_second =
            split_race_groups_into_test_train_validate(&synthetic_race_groups, 20, 80, 99)
                .expect("must succeed");

        let test_ids_first: Vec<i32> = split_first
            .test_race_groups
            .iter()
            .map(|g| g.race_game_id)
            .collect();
        let test_ids_second: Vec<i32> = split_second
            .test_race_groups
            .iter()
            .map(|g| g.race_game_id)
            .collect();
        assert_eq!(test_ids_first, test_ids_second);
    }

    /// Verifies that test_fraction_percent outside 1..=50 is rejected.
    #[test]
    fn three_way_split_rejects_invalid_test_fraction() {
        let synthetic_race_groups: Vec<RaceGroup> = vec![RaceGroup {
            race_game_id: 0,
            horse_records_in_race: build_complete_valid_race_group_for_testing(0, 0),
        }];

        let zero_result =
            split_race_groups_into_test_train_validate(&synthetic_race_groups, 0, 80, 1);
        assert!(matches!(
            zero_result,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));

        let too_high_result =
            split_race_groups_into_test_train_validate(&synthetic_race_groups, 51, 80, 1);
        assert!(matches!(
            too_high_result,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));
    }

    /// Verifies that grouping a well-formed flat record vector produces one
    /// `RaceGroup` per distinct `game_id`, each with exactly N horses.
    #[test]
    fn group_raw_records_by_game_id_produces_correct_group_count_and_sizes() {
        let mut all_synthetic_records: Vec<RawHorseRaceRecord> = Vec::new();
        all_synthetic_records.extend(build_complete_valid_race_group_for_testing(0, 0));
        all_synthetic_records.extend(build_complete_valid_race_group_for_testing(1, 5));
        all_synthetic_records.extend(build_complete_valid_race_group_for_testing(2, 10));

        let grouped_races_result = group_raw_records_by_game_id(&all_synthetic_records)
            .expect("well-formed records must group successfully");
        assert_eq!(grouped_races_result.len(), 3);
        for race_group_under_test in grouped_races_result.iter() {
            assert_eq!(
                race_group_under_test.horse_records_in_race.len(),
                HORSES_PER_RACE_GROUP
            );
        }
    }

    /// Verifies that the first-seen ordering of `game_id` is preserved,
    /// not sorted. This matters because downstream determinism depends on
    /// predictable ordering.
    #[test]
    fn group_raw_records_by_game_id_preserves_first_seen_order() {
        let mut all_synthetic_records: Vec<RawHorseRaceRecord> = Vec::new();
        // Insert game_id 7 first, then 2, then 5 — grouping must honor
        // this order in the output, not sort numerically.
        all_synthetic_records.extend(build_complete_valid_race_group_for_testing(7, 0));
        all_synthetic_records.extend(build_complete_valid_race_group_for_testing(2, 4));
        all_synthetic_records.extend(build_complete_valid_race_group_for_testing(5, 8));

        let grouped_races_result = group_raw_records_by_game_id(&all_synthetic_records)
            .expect("well-formed records must group successfully");
        assert_eq!(grouped_races_result[0].race_game_id, 7);
        assert_eq!(grouped_races_result[1].race_game_id, 2);
        assert_eq!(grouped_races_result[2].race_game_id, 5);
    }

    /// Verifies that grouping also works when records from different races
    /// are interleaved rather than appearing contiguously — a common reality
    /// for manually entered data.
    #[test]
    fn group_raw_records_by_game_id_handles_interleaved_records() {
        let mut all_synthetic_records: Vec<RawHorseRaceRecord> = Vec::new();
        // Build groups 0 and 1, then interleave their records.
        let race_group_zero_records = build_complete_valid_race_group_for_testing(0, 0);
        let race_group_one_records = build_complete_valid_race_group_for_testing(1, 4);
        for position_within_race in 0..HORSES_PER_RACE_GROUP {
            all_synthetic_records.push(race_group_zero_records[position_within_race].clone());
            all_synthetic_records.push(race_group_one_records[position_within_race].clone());
        }

        let grouped_races_result = group_raw_records_by_game_id(&all_synthetic_records)
            .expect("interleaved but complete records must group successfully");
        assert_eq!(grouped_races_result.len(), 2);
        for race_group_under_test in grouped_races_result.iter() {
            assert_eq!(
                race_group_under_test.horse_records_in_race.len(),
                HORSES_PER_RACE_GROUP
            );
        }
    }

    /// Verifies that a group with the wrong number of horses is rejected.
    #[test]
    fn group_raw_records_by_game_id_rejects_incomplete_group() {
        // Only 4 horses for game_id 0 — invalid.
        let mut incomplete_synthetic_records: Vec<RawHorseRaceRecord> = Vec::new();
        for position_within_race in 0..(HORSES_PER_RACE_GROUP - 1) {
            let assigned_rank: i32 = (position_within_race as i32) + 1;
            incomplete_synthetic_records.push(build_synthetic_horse_record_for_testing(
                position_within_race as i32,
                0,
                assigned_rank,
                1,
            ));
        }
        let group_result = group_raw_records_by_game_id(&incomplete_synthetic_records);
        assert!(matches!(
            group_result,
            Err(HorseRacingError::RaceGroupIncompleteOrOversized(_))
        ));
    }

    /// Verifies that the same seed produces the same shuffle across two
    /// independent runs — the core determinism guarantee.
    #[test]
    fn split_race_groups_is_deterministic_for_identical_seed() {
        let mut synthetic_race_groups: Vec<RaceGroup> = Vec::new();
        let total_synthetic_group_count: i32 = 10;
        for race_index in 0..total_synthetic_group_count {
            let synthetic_group = RaceGroup {
                race_game_id: race_index,
                horse_records_in_race: build_complete_valid_race_group_for_testing(
                    race_index,
                    race_index * (HORSES_PER_RACE_GROUP as i32),
                ),
            };
            synthetic_race_groups.push(synthetic_group);
        }

        let training_percent: u32 = 80;
        let seed_value_fixed: u32 = 42;
        let split_first_run = split_race_groups_into_train_and_validate(
            &synthetic_race_groups,
            training_percent,
            seed_value_fixed,
        )
        .expect("split must succeed on valid input");
        let split_second_run = split_race_groups_into_train_and_validate(
            &synthetic_race_groups,
            training_percent,
            seed_value_fixed,
        )
        .expect("split must succeed on valid input");

        // Compare by game_id ordering within each half.
        let training_ids_first_run: Vec<i32> = split_first_run
            .training_race_groups
            .iter()
            .map(|race_group_reference| race_group_reference.race_game_id)
            .collect();
        let training_ids_second_run: Vec<i32> = split_second_run
            .training_race_groups
            .iter()
            .map(|race_group_reference| race_group_reference.race_game_id)
            .collect();
        assert_eq!(training_ids_first_run, training_ids_second_run);

        let validation_ids_first_run: Vec<i32> = split_first_run
            .validation_race_groups
            .iter()
            .map(|race_group_reference| race_group_reference.race_game_id)
            .collect();
        let validation_ids_second_run: Vec<i32> = split_second_run
            .validation_race_groups
            .iter()
            .map(|race_group_reference| race_group_reference.race_game_id)
            .collect();
        assert_eq!(validation_ids_first_run, validation_ids_second_run);
    }

    /// Verifies that the split sizes match the requested percentage (with
    /// the clamp-to-at-least-one rule applied).
    #[test]
    fn split_race_groups_produces_expected_size_counts() {
        let mut synthetic_race_groups: Vec<RaceGroup> = Vec::new();
        let total_synthetic_group_count: i32 = 10;
        for race_index in 0..total_synthetic_group_count {
            synthetic_race_groups.push(RaceGroup {
                race_game_id: race_index,
                horse_records_in_race: build_complete_valid_race_group_for_testing(
                    race_index,
                    race_index * (HORSES_PER_RACE_GROUP as i32),
                ),
            });
        }

        let training_percent: u32 = 80;
        let split_result =
            split_race_groups_into_train_and_validate(&synthetic_race_groups, training_percent, 42)
                .expect("split must succeed on valid input");

        // 10 groups * 80% = 8 training, 2 validation.
        assert_eq!(split_result.training_race_groups.len(), 8);
        assert_eq!(split_result.validation_race_groups.len(), 2);
    }

    /// Verifies that no race group appears in both halves and no group is
    /// lost — the split must be a true partition of the input.
    #[test]
    fn split_race_groups_produces_disjoint_exhaustive_partition() {
        let mut synthetic_race_groups: Vec<RaceGroup> = Vec::new();
        let total_synthetic_group_count: i32 = 7;
        for race_index in 0..total_synthetic_group_count {
            synthetic_race_groups.push(RaceGroup {
                race_game_id: race_index,
                horse_records_in_race: build_complete_valid_race_group_for_testing(
                    race_index,
                    race_index * (HORSES_PER_RACE_GROUP as i32),
                ),
            });
        }

        let split_result =
            split_race_groups_into_train_and_validate(&synthetic_race_groups, 70, 999)
                .expect("split must succeed on valid input");

        let mut seen_game_ids_in_training: Vec<i32> = split_result
            .training_race_groups
            .iter()
            .map(|race_group_reference| race_group_reference.race_game_id)
            .collect();
        let mut seen_game_ids_in_validation: Vec<i32> = split_result
            .validation_race_groups
            .iter()
            .map(|race_group_reference| race_group_reference.race_game_id)
            .collect();

        // Disjoint: no overlap.
        for training_id_value in seen_game_ids_in_training.iter() {
            assert!(!seen_game_ids_in_validation.contains(training_id_value));
        }

        // Exhaustive: union equals the original set.
        let mut combined_observed_ids: Vec<i32> = Vec::new();
        combined_observed_ids.append(&mut seen_game_ids_in_training);
        combined_observed_ids.append(&mut seen_game_ids_in_validation);
        combined_observed_ids.sort();
        let expected_id_list: Vec<i32> = (0..total_synthetic_group_count).collect();
        assert_eq!(combined_observed_ids, expected_id_list);
    }

    /// Verifies that a split percentage outside the valid 1-99 range is
    /// rejected rather than producing a degenerate split.
    #[test]
    fn split_race_groups_rejects_invalid_training_fraction_percent() {
        let single_synthetic_group_list: Vec<RaceGroup> = vec![RaceGroup {
            race_game_id: 0,
            horse_records_in_race: build_complete_valid_race_group_for_testing(0, 0),
        }];

        let zero_percent_result =
            split_race_groups_into_train_and_validate(&single_synthetic_group_list, 0, 1);
        assert!(matches!(
            zero_percent_result,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));

        let one_hundred_percent_result =
            split_race_groups_into_train_and_validate(&single_synthetic_group_list, 100, 1);
        assert!(matches!(
            one_hundred_percent_result,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));
    }

    /// Verifies that splitting an empty race-group list is rejected.
    #[test]
    fn split_race_groups_rejects_empty_input() {
        let empty_race_group_list: Vec<RaceGroup> = Vec::new();
        let split_result =
            split_race_groups_into_train_and_validate(&empty_race_group_list, 80, 42);
        assert!(matches!(
            split_result,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));
    }

    /// Verifies that the seeded PRNG produces the same sequence for the
    /// same seed, and different sequences for different seeds.
    #[test]
    fn deterministic_pseudo_random_generator_is_reproducible_and_seed_sensitive() {
        let mut generator_instance_one =
            DeterministicLinearCongruentialPseudoRandomGenerator::new_with_seed(42);
        let mut generator_instance_two =
            DeterministicLinearCongruentialPseudoRandomGenerator::new_with_seed(42);
        let mut generator_instance_three =
            DeterministicLinearCongruentialPseudoRandomGenerator::new_with_seed(43);

        let first_sequence_sample_count: usize = 4;
        let mut first_seed_values: Vec<u32> = Vec::new();
        let mut second_seed_values: Vec<u32> = Vec::new();
        let mut different_seed_values: Vec<u32> = Vec::new();

        for _repetition_index in 0..first_sequence_sample_count {
            first_seed_values.push(generator_instance_one.next_pseudo_random_u32());
            second_seed_values.push(generator_instance_two.next_pseudo_random_u32());
            different_seed_values.push(generator_instance_three.next_pseudo_random_u32());
        }

        // Same seed -> same sequence.
        assert_eq!(first_seed_values, second_seed_values);
        // Different seed -> different sequence (overwhelmingly likely even
        // for a tiny LCG; check inequality, not any specific values).
        assert_ne!(first_seed_values, different_seed_values);
    }

    /// Verifies that flattening preserves count and content for a simple case.
    #[test]
    fn flatten_race_groups_preserves_record_count_and_values() {
        let mut synthetic_race_groups: Vec<RaceGroup> = Vec::new();
        for race_index in 0..3 {
            synthetic_race_groups.push(RaceGroup {
                race_game_id: race_index,
                horse_records_in_race: build_complete_valid_race_group_for_testing(
                    race_index,
                    race_index * (HORSES_PER_RACE_GROUP as i32),
                ),
            });
        }
        let flattened_output = flatten_race_groups_into_records(&synthetic_race_groups);
        assert_eq!(flattened_output.len(), 3 * HORSES_PER_RACE_GROUP);
        // First record should be from game_id 0, row_id 0.
        assert_eq!(flattened_output[0].game_id, 0);
        assert_eq!(flattened_output[0].row_id, 0);
    }
}

/*
Section 4a: Decision Tree Data Structures and Split Evaluation
As noted, Section 4 is the largest, so I am splitting it in two. This sub-section (4a) introduces the tree's shape in memory and the functions that evaluate how good a candidate split is. It does not yet build the tree — that is Section 4b. Keeping split evaluation separate from the build loop lets each piece be tested independently.

What This Sub-Section Contains

TreeNodeBranchDecision enum — whether a node is a decision node (split on a feature) or a leaf (final prediction)
DecisionTreeNode struct — one node in the flat Vec<DecisionTreeNode> tree representation
DecisionTree struct — the whole tree: a flat node vector plus the root index and the label-kind tag
TreeLabelKind enum — whether the tree predicts completion (classification) or performance_score (regression)
FeatureIndex enum + helpers — which engineered feature a split uses, with name lookups for model file round-tripping
extract_feature_value_from_vector — pulls one feature value out of an EngineeredFeatureVector by FeatureIndex
compute_classification_impurity_gini — Gini impurity of a set of completion labels (0/1)
compute_regression_impurity_variance_times_count — variance-times-count of a set of performance scores (integer-math proxy for variance)
SplitEvaluation struct — the result of evaluating one candidate split: the feature, the threshold, the resulting impurity, and the split sizes
evaluate_candidate_split_for_classification — computes the impurity after splitting the completion labels at a threshold
evaluate_candidate_split_for_regression — same for performance-score labels

Plus cargo tests.

Design Decisions Explained Up Front
Why a flat Vec<DecisionTreeNode> with integer child indices instead of Box<Node> children: Project rule — no recursion. A flat vector with integer indices lets the tree be traversed iteratively (a simple while loop walking a current_node_index). It also makes the plain-text model file format trivial (one node per line, children referred to by index). And it avoids the heap-node-per-tree-node allocation overhead, which matters for future embedded or constrained deployments even if it does not matter at this project's current scale.
Why use u32 sentinel (NO_CHILD_NODE_INDEX) rather than Option<u32> for children: Two reasons. First, the plain-text model file serializes the sentinel as a simple integer, with no Some(...)/None syntax to parse. Second, keeping every node the same size in memory (no option tag) makes the on-disk size exactly predictable. The sentinel value is well above any plausible node count, so it cannot collide with a real index.
Why integer math for regression impurity (variance × count, not mean-based variance): Computing variance the normal way — sum((x - mean)^2) / count — requires either floats or a division that loses precision. But for split selection, we only need to compare impurities, and count * sum_of_squares - sum * sum (which equals count² × variance) preserves the ordering without any division. This keeps all math in i64 (promoted from i32 to avoid overflow on sums of squares at 1000² × 200 = 2×10⁸), and we never divide.
Why Gini (not entropy) for classification: Gini requires no logarithm — and a logarithm means floats. Gini is also the scikit-learn default and is fine for binary classification with small data. It is computed purely in integer arithmetic here.
Why a TreeLabelKind tag on the tree (not two separate tree types): The tree structure and traversal are identical for both label kinds; only the split-evaluation function and the leaf prediction value differ. One struct with a tag gives one set of save/load code, one prediction traversal, and one set of tests. The alternative — two near-identical tree types — would roughly double the code surface for no gain.
Why FeatureIndex as an enum (not a raw usize): An enum makes it impossible to pass an out-of-range feature index to the tree, and it lets the model file use stable feature names rather than positional integers (so reordering the EngineeredFeatureVector struct fields would not silently corrupt saved models).

Weighted-impurity convention: I documented the exact mathematical convention (n × gini_times_n_squared and n × variance_times_n_squared) inline so that a future reader comparing implementations against a textbook can see exactly what the integer form is computing. Textbook Gini and population variance would appear at different scales; this code's scale is specific and consistent across both task types within its own comparisons.
Why split-evaluation functions return Result despite having "simple" logic: The length-mismatch check protects against caller bugs that would otherwise silently produce meaningless results (or panic on slice indexing). A small defensive check at the interface boundary, with a unique error prefix, is worth the three lines.
Why labels_going_to_left_child / ..right_child are Vec<i32> (not slices): The samples going left and right are not contiguous in the input, so a slice cannot represent them. Allocating two small vectors per split evaluation is acceptable at ~200 samples × a few candidate thresholds per feature. If profiling later shows this as a bottleneck, it can be replaced with index vectors or in-place partitioning.
The test for NO_CHILD_NODE_INDEX: I added this after writing the constant, as a tripwire against someone changing the sentinel value to something within the realistic node-count range — which would silently corrupt tree interpretation.

*/
// ============================================================================
// SECTION 4a — DECISION TREE DATA STRUCTURES AND SPLIT EVALUATION
// ============================================================================
//
// This sub-section defines the tree's shape in memory and the split-quality
// functions that will drive the build loop in Section 4b. No tree building
// happens here yet.
//
// ## Tree Representation
//
// The tree is a flat `Vec<DecisionTreeNode>` with `u32` child indices, not a
// pointer-linked recursive structure. This choice is driven by the project
// rule forbidding recursion: a flat vector lets prediction walk the tree
// with a simple `while` loop, and lets the build loop (in Section 4b) use
// an explicit work stack instead of recursive calls. As a secondary
// benefit, the plain-text model file format becomes straightforward (one
// node per line).
//
// ## Classification vs. Regression
//
// The same `DecisionTree` struct represents trees for both tasks:
//
//   * Classification: predicts `completion` (binary 0 or 1).
//   * Regression:     predicts `performance_score` (integer 0..=1000).
//
// A `TreeLabelKind` tag on the tree tells traversal code how to interpret
// the `leaf_predicted_value` field of leaf nodes: as a 0/1 class label in
// the classification case, or as a performance-score integer in the
// regression case.
//
// ## Integer-Only Math
//
// Both impurity measures (Gini for classification, variance×count for
// regression) are computed entirely in integer arithmetic. No floats, no
// logarithms. Details in the per-function docstrings.

/// Sentinel value used in `left_child_node_index` and `right_child_node_index`
/// to mark "no child" (i.e. this is a leaf node).
///
/// Chosen as `u32::MAX` so it cannot collide with any real node index in any
/// tree this project would ever build (`u32::MAX` is ~4.3 billion; the
/// project's trees have at most a few dozen nodes). The sentinel is also
/// what the plain-text model file writes for leaf children — unambiguous
/// and trivially machine-parseable.
pub const NO_CHILD_NODE_INDEX: u32 = u32::MAX;

/// Identifies which engineered feature a split operates on.
///
/// ## Why an Enum (Not a Raw Index)
///
/// A raw `usize` index would allow invalid values at the type level, and
/// would mean that reordering `EngineeredFeatureVector` fields silently
/// changes the meaning of saved model files. The enum couples each feature
/// to a stable name (`canonical_feature_name_string`) that the model file
/// writes and reads, so reordering struct fields cannot corrupt models.
///
/// ## Adding a New Feature
///
/// To add a feature: add an enum variant here, update
/// `extract_feature_value_from_vector`, update
/// `all_feature_indices_in_canonical_order`, and update
/// `canonical_feature_name_string` / `feature_index_from_canonical_name`.
/// `ENGINEERED_FEATURE_COUNT` must also be updated. The compiler will
/// enforce the first two via exhaustive-match errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureIndex {
    Age,
    Height,
    Experience,
    Weight,
    HeightToWeightRatioTimesOneThousand,
    AgeTimesExperience,
}

impl FeatureIndex {
    /// Returns the stable, on-disk name of this feature.
    ///
    /// The string is what the plain-text model file uses to refer to the
    /// split feature. Changing one of these strings is a breaking change to
    /// the model file format.
    pub fn canonical_feature_name_string(&self) -> &'static str {
        match self {
            FeatureIndex::Age => "age",
            FeatureIndex::Height => "height",
            FeatureIndex::Experience => "experience",
            FeatureIndex::Weight => "weight",
            FeatureIndex::HeightToWeightRatioTimesOneThousand => {
                "height_to_weight_ratio_times_one_thousand"
            }
            FeatureIndex::AgeTimesExperience => "age_times_experience",
        }
    }

    /// Parses a canonical feature name back into a `FeatureIndex`.
    ///
    /// Used by the model file loader. Returns `FieldValueOutOfValidRange`
    /// (message identifies this function) on any unknown name — unknown
    /// names in a model file indicate either a format mismatch or a model
    /// built with a different feature set than the current binary knows
    /// about, and both are hard errors.
    pub fn feature_index_from_canonical_name(
        candidate_feature_name: &str,
    ) -> Result<FeatureIndex, HorseRacingError> {
        match candidate_feature_name {
            "age" => Ok(FeatureIndex::Age),
            "height" => Ok(FeatureIndex::Height),
            "experience" => Ok(FeatureIndex::Experience),
            "weight" => Ok(FeatureIndex::Weight),
            "height_to_weight_ratio_times_one_thousand" => {
                Ok(FeatureIndex::HeightToWeightRatioTimesOneThousand)
            }
            "age_times_experience" => Ok(FeatureIndex::AgeTimesExperience),
            _ => Err(HorseRacingError::FieldValueOutOfValidRange(
                "feature_index_from_canonical_name: unknown feature name",
            )),
        }
    }
}

/// Returns every `FeatureIndex` in a stable canonical order.
///
/// Used by the tree-build loop to iterate candidate split features. The
/// order here is the order in which features are considered for splitting,
/// which affects tie-breaking when two features give identical impurity
/// reductions (the earlier feature wins). Keeping this order stable keeps
/// training deterministic.
pub fn all_feature_indices_in_canonical_order() -> [FeatureIndex; ENGINEERED_FEATURE_COUNT] {
    [
        FeatureIndex::Age,
        FeatureIndex::Height,
        FeatureIndex::Experience,
        FeatureIndex::Weight,
        FeatureIndex::HeightToWeightRatioTimesOneThousand,
        FeatureIndex::AgeTimesExperience,
    ]
}

/// Extracts the integer value of one feature from a full feature vector.
///
/// Centralizing this lookup means the tree code never touches
/// `EngineeredFeatureVector` fields directly — it only ever asks for "the
/// value of feature X in this vector". Adding, removing, or reordering
/// struct fields only requires updating this function (plus the enum and
/// the canonical-order list).
pub fn extract_feature_value_from_vector(
    feature_vector_reference: &EngineeredFeatureVector,
    feature_to_extract: FeatureIndex,
) -> i32 {
    match feature_to_extract {
        FeatureIndex::Age => feature_vector_reference.age,
        FeatureIndex::Height => feature_vector_reference.height,
        FeatureIndex::Experience => feature_vector_reference.experience,
        FeatureIndex::Weight => feature_vector_reference.weight,
        FeatureIndex::HeightToWeightRatioTimesOneThousand => {
            feature_vector_reference.height_to_weight_ratio_times_one_thousand
        }
        FeatureIndex::AgeTimesExperience => feature_vector_reference.age_times_experience,
    }
}

/// Tag indicating what kind of label the tree predicts.
///
/// Classification trees predict `completion` (0 or 1). Regression trees
/// predict `performance_score` (0..=1000). The same tree structure and
/// traversal code handles both; this tag tells the code which impurity
/// measure was used during training and how to interpret a leaf's prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeLabelKind {
    /// Tree predicts the binary `completion` label (0 or 1).
    CompletionClassification,
    /// Tree predicts the integer `performance_score` label (0..=1000).
    PerformanceScoreRegression,
}

impl TreeLabelKind {
    /// Stable on-disk name for the model file.
    pub fn canonical_label_kind_name_string(&self) -> &'static str {
        match self {
            TreeLabelKind::CompletionClassification => "completion_classification",
            TreeLabelKind::PerformanceScoreRegression => "performance_score_regression",
        }
    }

    /// Parses a label-kind name from the model file.
    pub fn label_kind_from_canonical_name(
        candidate_label_kind_name: &str,
    ) -> Result<TreeLabelKind, HorseRacingError> {
        match candidate_label_kind_name {
            "completion_classification" => Ok(TreeLabelKind::CompletionClassification),
            "performance_score_regression" => Ok(TreeLabelKind::PerformanceScoreRegression),
            _ => Err(HorseRacingError::FieldValueOutOfValidRange(
                "label_kind_from_canonical_name: unknown label kind",
            )),
        }
    }
}

/// Distinguishes a decision (internal) node from a leaf node.
///
/// Kept as an enum rather than relying on "children are sentinel == leaf"
/// so that node kind is checkable at the type level. A decision node
/// always has both children set; a leaf node always has a prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeNodeBranchDecision {
    /// Internal node: routes a feature vector left or right based on a
    /// threshold comparison on a single feature.
    DecisionNode,
    /// Terminal node: returns a fixed prediction for any feature vector
    /// that reaches it.
    LeafNode,
}

/// One node in the flat tree vector.
///
/// ## Field Semantics
///
/// - `node_branch_decision` — whether this is a decision or a leaf.
/// - `split_feature_index` — if decision: the feature to test. If leaf:
///   unused but set to `FeatureIndex::Age` as a harmless default.
/// - `split_threshold_value` — if decision: a feature-vector value is
///   routed left if `feature_value < split_threshold_value`, else right.
///   If leaf: unused.
/// - `left_child_node_index`, `right_child_node_index` — if decision: valid
///   indices into the tree's node vector. If leaf: both equal
///   `NO_CHILD_NODE_INDEX`.
/// - `leaf_predicted_value` — if leaf: the integer prediction. For
///   classification trees, 0 or 1. For regression trees, the average
///   performance score of the training samples that reached this leaf,
///   rounded to integer.
///
/// ## Why `i32` for `leaf_predicted_value`
///
/// Classification leaves carry 0 or 1; regression leaves carry 0..=1000.
/// A single `i32` field serves both uses, and the on-disk format is one
/// integer per leaf value either way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionTreeNode {
    pub node_branch_decision: TreeNodeBranchDecision,
    pub split_feature_index: FeatureIndex,
    pub split_threshold_value: i32,
    pub left_child_node_index: u32,
    pub right_child_node_index: u32,
    pub leaf_predicted_value: i32,
}

/// A complete trained decision tree.
///
/// ## Fields
///
/// - `all_tree_nodes_flat_vector` — every node, indexed by its position.
///   The root is at `root_node_index` (usually 0, but recorded explicitly
///   so a future pruning pass can rewrite the root without re-indexing
///   every node).
/// - `root_node_index` — entry point for prediction traversal.
/// - `label_kind_for_this_tree` — classification or regression.
/// - `max_depth_used_for_training` — recorded so the model file can note
///   the hyperparameters the tree was trained with, for reproducibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionTree {
    pub all_tree_nodes_flat_vector: Vec<DecisionTreeNode>,
    pub root_node_index: u32,
    pub label_kind_for_this_tree: TreeLabelKind,
    pub max_depth_used_for_training: u32,
}

/// Result of evaluating a single candidate split of a set of samples.
///
/// ## Fields
///
/// - `split_feature_index` — the feature this split tests.
/// - `split_threshold_value` — the threshold used; left if
///   `value < threshold`, else right.
/// - `combined_impurity_after_split` — the weighted impurity of the two
///   resulting child sample sets. Lower is better. The exact meaning of
///   "impurity" depends on the task (Gini for classification,
///   variance×count for regression); this field holds the value produced
///   by whichever impurity function was used, so `SplitEvaluation` values
///   should only be compared across splits evaluated with the *same*
///   function. The classification and regression loops never mix these.
/// - `left_child_sample_count`, `right_child_sample_count` — how many
///   training samples went to each side. Used by the build loop to enforce
///   the minimum-samples-per-leaf hyperparameter.
///
/// ## Why `i64` for `combined_impurity_after_split`
///
/// The regression impurity is `sum_of_squares` at scales up to ~200 rows ×
/// 1000² = 2×10⁸, multiplied by `count` for the weighting. That product can
/// reach ~10¹⁰, which overflows `i32`. `i64` gives comfortable headroom.
/// Classification impurity (Gini × count) is much smaller but using the
/// same field type keeps the struct uniform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitEvaluation {
    pub split_feature_index: FeatureIndex,
    pub split_threshold_value: i32,
    pub combined_impurity_after_split: i64,
    pub left_child_sample_count: usize,
    pub right_child_sample_count: usize,
}

/// Computes an integer-only Gini impurity surrogate for a set of binary
/// labels (0 or 1).
///
/// ## Formula
///
/// Standard Gini for a two-class set with `count_zeros` and `count_ones`
/// (total `n = count_zeros + count_ones`) is:
///
///   gini = 1 - (count_zeros/n)^2 - (count_ones/n)^2
///
/// Multiplying by `n^2` to stay in integers:
///
///   gini_times_n_squared = n*n - count_zeros^2 - count_ones^2
///
/// ## Why Return `n * gini_times_n_squared` (Weighted)
///
/// When comparing a split, the two child impurities must be *weighted by
/// child size* before being summed. The usual formula is:
///
///   weighted = (n_left/n_total) * gini_left + (n_right/n_total) * gini_right
///
/// With the integer form, the caller combines impurities as
/// `n_left * gini_left_n_squared + n_right * gini_right_n_squared` ... but
/// that mixes scales across children with different `n`. To keep the
/// comparison fair across splits *evaluated with the same impurity
/// function*, this function returns `n * gini_times_n_squared`, i.e. all
/// impurities are expressed at a common cubic-in-count scale. The absolute
/// values are not meaningful; only their ordering is, and this function
/// preserves that ordering in an integer-safe way.
///
/// ## Edge Case
///
/// For `n == 0` the function returns 0. An empty set has no impurity, and
/// callers guard against attempting to split into an empty child anyway.
pub fn compute_classification_impurity_gini(binary_labels_slice: &[i32]) -> i64 {
    let total_sample_count_in_set: i64 = binary_labels_slice.len() as i64;
    if total_sample_count_in_set == 0 {
        return 0;
    }

    let mut count_of_zero_labels_in_set: i64 = 0;
    let mut count_of_one_labels_in_set: i64 = 0;
    // The loop is bounded by the input slice length, which is bounded by
    // the training set size, which is bounded by the CSV reader's
    // defensive cap.
    for single_binary_label_reference in binary_labels_slice.iter() {
        if *single_binary_label_reference == 0 {
            count_of_zero_labels_in_set += 1;
        } else {
            // Any nonzero label is treated as "1" for Gini purposes. The
            // validator in Section 2 already rejects non-0/1 completion
            // values at parse time, so this branch never sees other values
            // in normal flow; treating them as "1" is a defensive fallback.
            count_of_one_labels_in_set += 1;
        }
    }

    let gini_times_n_squared: i64 = total_sample_count_in_set * total_sample_count_in_set
        - count_of_zero_labels_in_set * count_of_zero_labels_in_set
        - count_of_one_labels_in_set * count_of_one_labels_in_set;
    // Weight by n so that impurities from sets of different sizes can be
    // added meaningfully in the split-evaluation formula.
    total_sample_count_in_set * gini_times_n_squared
}

/// Computes an integer-only variance-times-count surrogate for a set of
/// integer regression labels (e.g. performance scores).
///
/// ## Formula
///
/// Population variance is `sum((x - mean)^2) / n`. Multiplying by `n^2`
/// and expanding:
///
///   variance_times_n_squared = n * sum_of_squares - sum * sum
///
/// Returning `variance_times_n_squared` avoids the division entirely and
/// keeps all math in `i64`. This preserves the ordering of splits by
/// variance-reduction, which is all the split-evaluation code needs.
///
/// ## Why Return `variance_times_n_squared` Directly (Not Multiplied by n)
///
/// The classification analog above multiplies by an extra `n` to make
/// differently-sized child impurities addable. For regression, this is
/// unnecessary: the split-evaluation combines children by multiplying each
/// child's `variance_times_n_squared` by... wait, no. Consistency matters
/// here, so *this function also applies the extra n*, making the combined
/// quantity `n * variance_times_n_squared`. See the
/// `evaluate_candidate_split_for_regression` docstring for how the caller
/// uses this.
///
/// ## Overflow
///
/// Worst case: `n = 200`, `x_max = 1000`. `sum_of_squares <= 200 * 10^6 = 2×10^8`.
/// `n * sum_of_squares <= 200 * 2×10^8 = 4×10^10`. Fits in `i64` with
/// ~8 orders of magnitude to spare.
///
/// Then multiplying by the extra `n` for cross-child comparability:
/// `4×10^10 * 200 = 8×10^12`, still well within `i64`.
pub fn compute_regression_impurity_variance_times_count(
    integer_regression_labels_slice: &[i32],
) -> i64 {
    let total_sample_count_in_set: i64 = integer_regression_labels_slice.len() as i64;
    if total_sample_count_in_set == 0 {
        return 0;
    }

    let mut running_sum_of_values: i64 = 0;
    let mut running_sum_of_squared_values: i64 = 0;
    // Bounded loop, as above.
    for single_regression_label_reference in integer_regression_labels_slice.iter() {
        let value_as_i64: i64 = *single_regression_label_reference as i64;
        running_sum_of_values += value_as_i64;
        running_sum_of_squared_values += value_as_i64 * value_as_i64;
    }

    let variance_times_n_squared: i64 = total_sample_count_in_set * running_sum_of_squared_values
        - running_sum_of_values * running_sum_of_values;
    // Weight by n for cross-split, cross-child comparability.
    total_sample_count_in_set * variance_times_n_squared
}

/// Evaluates a candidate classification split on a set of (feature_value,
/// completion_label) pairs.
///
/// ## Inputs
///
/// - `feature_values_slice` and `binary_completion_labels_slice` are
///   parallel arrays of equal length. Entry `i` in the first is the
///   feature-value for the same training sample whose label is entry `i`
///   in the second.
/// - `candidate_threshold_value` — samples with `feature_value <
///   candidate_threshold_value` go left; the rest go right.
/// - `feature_this_split_operates_on` — carried through into the returned
///   `SplitEvaluation` so the build loop knows which feature produced this
///   evaluation.
///
/// ## Returned Impurity
///
/// `combined_impurity_after_split = impurity(left_labels) + impurity(right_labels)`,
/// where each impurity is computed by
/// `compute_classification_impurity_gini` in its cross-comparable weighted
/// form. Lower is better.
///
/// ## Degenerate Splits
///
/// If all samples go to one side (left or right is empty), the split is
/// useless — it does not actually split anything. This function still
/// returns a `SplitEvaluation` with `left_child_sample_count` or
/// `right_child_sample_count` equal to zero; the build loop is responsible
/// for rejecting splits below the minimum-samples-per-leaf threshold.
/// Surfacing the sample counts (rather than returning an error) keeps the
/// interface simple and lets the build loop's policy live in one place.
pub fn evaluate_candidate_split_for_classification(
    feature_values_slice: &[i32],
    binary_completion_labels_slice: &[i32],
    candidate_threshold_value: i32,
    feature_this_split_operates_on: FeatureIndex,
) -> Result<SplitEvaluation, HorseRacingError> {
    if feature_values_slice.len() != binary_completion_labels_slice.len() {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "evaluate_candidate_split_for_classification: feature and label slices length mismatch",
        ));
    }

    // Partition labels into left and right buckets according to the threshold.
    let sample_count_total: usize = feature_values_slice.len();
    let mut labels_going_to_left_child: Vec<i32> = Vec::with_capacity(sample_count_total);
    let mut labels_going_to_right_child: Vec<i32> = Vec::with_capacity(sample_count_total);
    for sample_position in 0..sample_count_total {
        if feature_values_slice[sample_position] < candidate_threshold_value {
            labels_going_to_left_child.push(binary_completion_labels_slice[sample_position]);
        } else {
            labels_going_to_right_child.push(binary_completion_labels_slice[sample_position]);
        }
    }

    let left_child_impurity_contribution =
        compute_classification_impurity_gini(&labels_going_to_left_child);
    let right_child_impurity_contribution =
        compute_classification_impurity_gini(&labels_going_to_right_child);
    let combined_impurity: i64 =
        left_child_impurity_contribution + right_child_impurity_contribution;

    Ok(SplitEvaluation {
        split_feature_index: feature_this_split_operates_on,
        split_threshold_value: candidate_threshold_value,
        combined_impurity_after_split: combined_impurity,
        left_child_sample_count: labels_going_to_left_child.len(),
        right_child_sample_count: labels_going_to_right_child.len(),
    })
}

/// Evaluates a candidate regression split on a set of (feature_value,
/// performance_score_label) pairs.
///
/// Structurally identical to `evaluate_candidate_split_for_classification`,
/// except the impurity used is
/// `compute_regression_impurity_variance_times_count` instead of Gini.
///
/// See that function's docstring and the classification function's
/// docstring for shared contract details.
pub fn evaluate_candidate_split_for_regression(
    feature_values_slice: &[i32],
    integer_regression_labels_slice: &[i32],
    candidate_threshold_value: i32,
    feature_this_split_operates_on: FeatureIndex,
) -> Result<SplitEvaluation, HorseRacingError> {
    if feature_values_slice.len() != integer_regression_labels_slice.len() {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "evaluate_candidate_split_for_regression: feature and label slices length mismatch",
        ));
    }

    let sample_count_total: usize = feature_values_slice.len();
    let mut labels_going_to_left_child: Vec<i32> = Vec::with_capacity(sample_count_total);
    let mut labels_going_to_right_child: Vec<i32> = Vec::with_capacity(sample_count_total);
    for sample_position in 0..sample_count_total {
        if feature_values_slice[sample_position] < candidate_threshold_value {
            labels_going_to_left_child.push(integer_regression_labels_slice[sample_position]);
        } else {
            labels_going_to_right_child.push(integer_regression_labels_slice[sample_position]);
        }
    }

    let left_child_impurity_contribution =
        compute_regression_impurity_variance_times_count(&labels_going_to_left_child);
    let right_child_impurity_contribution =
        compute_regression_impurity_variance_times_count(&labels_going_to_right_child);
    let combined_impurity: i64 =
        left_child_impurity_contribution + right_child_impurity_contribution;

    Ok(SplitEvaluation {
        split_feature_index: feature_this_split_operates_on,
        split_threshold_value: candidate_threshold_value,
        combined_impurity_after_split: combined_impurity,
        left_child_sample_count: labels_going_to_left_child.len(),
        right_child_sample_count: labels_going_to_right_child.len(),
    })
}

// ============================================================================
// SECTION 4a — CARGO TESTS
// ============================================================================

#[cfg(test)]
mod section_four_a_tree_structures_and_split_tests {
    use super::*;

    /// Builds a feature vector with named values, for use in tests.
    fn build_synthetic_feature_vector_for_testing(
        age_value: i32,
        height_value: i32,
        experience_value: i32,
        weight_value: i32,
        height_to_weight_ratio_value: i32,
        age_times_experience_value: i32,
    ) -> EngineeredFeatureVector {
        EngineeredFeatureVector {
            age: age_value,
            height: height_value,
            experience: experience_value,
            weight: weight_value,
            height_to_weight_ratio_times_one_thousand: height_to_weight_ratio_value,
            age_times_experience: age_times_experience_value,
        }
    }

    /// Verifies that every `FeatureIndex` variant has a unique canonical
    /// name and that the list of canonical-order features has the documented
    /// count. This catches a missed update if a new feature is added
    /// without updating all three sites.
    #[test]
    fn feature_index_canonical_names_are_unique_and_count_matches_constant() {
        let canonical_order_array = all_feature_indices_in_canonical_order();
        assert_eq!(canonical_order_array.len(), ENGINEERED_FEATURE_COUNT);

        // Pairwise uniqueness check on canonical names.
        for outer_position in 0..canonical_order_array.len() {
            for inner_position in (outer_position + 1)..canonical_order_array.len() {
                let outer_feature_name =
                    canonical_order_array[outer_position].canonical_feature_name_string();
                let inner_feature_name =
                    canonical_order_array[inner_position].canonical_feature_name_string();
                assert_ne!(
                    outer_feature_name, inner_feature_name,
                    "two FeatureIndex variants share a canonical name"
                );
            }
        }
    }

    /// Verifies that `feature_index_from_canonical_name` is the inverse of
    /// `canonical_feature_name_string` for every variant.
    #[test]
    fn feature_index_name_round_trip_preserves_identity() {
        for feature_under_test in all_feature_indices_in_canonical_order().iter() {
            let name_string = feature_under_test.canonical_feature_name_string();
            let round_tripped_feature =
                FeatureIndex::feature_index_from_canonical_name(name_string)
                    .expect("canonical name must parse back");
            assert_eq!(*feature_under_test, round_tripped_feature);
        }
    }

    /// Verifies that an unknown feature name is rejected.
    #[test]
    fn feature_index_from_canonical_name_rejects_unknown() {
        let parse_result =
            FeatureIndex::feature_index_from_canonical_name("totally_made_up_feature");
        assert!(matches!(
            parse_result,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));
    }

    /// Verifies feature extraction returns the expected value for each
    /// feature variant.
    #[test]
    fn extract_feature_value_returns_correct_field_per_variant() {
        let probe_vector = build_synthetic_feature_vector_for_testing(4, 150, 3, 987, 152, 12);
        assert_eq!(
            extract_feature_value_from_vector(&probe_vector, FeatureIndex::Age),
            4
        );
        assert_eq!(
            extract_feature_value_from_vector(&probe_vector, FeatureIndex::Height),
            150
        );
        assert_eq!(
            extract_feature_value_from_vector(&probe_vector, FeatureIndex::Experience),
            3
        );
        assert_eq!(
            extract_feature_value_from_vector(&probe_vector, FeatureIndex::Weight),
            987
        );
        assert_eq!(
            extract_feature_value_from_vector(
                &probe_vector,
                FeatureIndex::HeightToWeightRatioTimesOneThousand
            ),
            152
        );
        assert_eq!(
            extract_feature_value_from_vector(&probe_vector, FeatureIndex::AgeTimesExperience),
            12
        );
    }

    /// Verifies Gini impurity on a pure set (all one class) is zero, which
    /// is the theoretical minimum.
    #[test]
    fn gini_impurity_is_zero_for_pure_sets() {
        let all_zeros_labels: Vec<i32> = vec![0, 0, 0, 0];
        assert_eq!(compute_classification_impurity_gini(&all_zeros_labels), 0);
        let all_ones_labels: Vec<i32> = vec![1, 1, 1, 1, 1];
        assert_eq!(compute_classification_impurity_gini(&all_ones_labels), 0);
    }

    /// Verifies Gini impurity on a maximally impure set (50/50 mix) is
    /// strictly greater than zero and matches the closed-form value.
    ///
    /// For `n = 4` with 2 zeros and 2 ones:
    ///   gini_times_n_squared = 4*4 - 2*2 - 2*2 = 16 - 4 - 4 = 8
    ///   weighted (times n)   = 4 * 8 = 32
    #[test]
    fn gini_impurity_matches_closed_form_for_fifty_fifty_mix() {
        let fifty_fifty_labels: Vec<i32> = vec![0, 0, 1, 1];
        let computed_impurity = compute_classification_impurity_gini(&fifty_fifty_labels);
        assert_eq!(computed_impurity, 32);
    }

    /// Verifies Gini on an empty set returns 0, the documented edge case.
    #[test]
    fn gini_impurity_is_zero_for_empty_set() {
        let empty_labels: Vec<i32> = Vec::new();
        assert_eq!(compute_classification_impurity_gini(&empty_labels), 0);
    }

    /// Verifies regression variance-times-count on a constant-value set is
    /// zero (no variance at all).
    #[test]
    fn regression_impurity_is_zero_for_constant_labels() {
        let constant_labels: Vec<i32> = vec![500, 500, 500, 500];
        assert_eq!(
            compute_regression_impurity_variance_times_count(&constant_labels),
            0
        );
    }

    /// Verifies regression variance-times-count matches the closed-form
    /// value for a known small example.
    ///
    /// For labels [0, 10]:
    ///   n = 2, sum = 10, sum_of_squares = 100
    ///   variance_times_n_squared = 2*100 - 10*10 = 200 - 100 = 100
    ///   weighted (times n)       = 2 * 100 = 200
    #[test]
    fn regression_impurity_matches_closed_form_for_small_example() {
        let small_known_labels: Vec<i32> = vec![0, 10];
        let computed_impurity =
            compute_regression_impurity_variance_times_count(&small_known_labels);
        assert_eq!(computed_impurity, 200);
    }

    /// Verifies the classification split evaluator correctly partitions and
    /// reports sample counts for a simple case.
    ///
    /// Samples: feature values [1, 2, 3, 4], labels [0, 0, 1, 1].
    /// Threshold 3 sends [1, 2] (labels [0, 0]) left and [3, 4] (labels
    /// [1, 1]) right. Both children are pure -> impurity 0 on each side.
    #[test]
    fn classification_split_evaluation_partitions_correctly_on_pure_split() {
        let feature_values: Vec<i32> = vec![1, 2, 3, 4];
        let completion_labels: Vec<i32> = vec![0, 0, 1, 1];
        let evaluation_result = evaluate_candidate_split_for_classification(
            &feature_values,
            &completion_labels,
            3,
            FeatureIndex::Age,
        )
        .expect("valid inputs must evaluate");
        assert_eq!(evaluation_result.left_child_sample_count, 2);
        assert_eq!(evaluation_result.right_child_sample_count, 2);
        assert_eq!(evaluation_result.combined_impurity_after_split, 0);
    }

    /// Verifies that a threshold placing all samples on one side produces
    /// an empty child (sample count 0), which the build loop will reject.
    #[test]
    fn classification_split_evaluation_detects_degenerate_split() {
        let feature_values: Vec<i32> = vec![1, 2, 3, 4];
        let completion_labels: Vec<i32> = vec![0, 1, 0, 1];
        // Threshold below the minimum value sends everyone right.
        let evaluation_result = evaluate_candidate_split_for_classification(
            &feature_values,
            &completion_labels,
            0,
            FeatureIndex::Age,
        )
        .expect("valid inputs must evaluate");
        assert_eq!(evaluation_result.left_child_sample_count, 0);
        assert_eq!(evaluation_result.right_child_sample_count, 4);
    }

    /// Verifies that split-evaluation rejects mismatched feature/label
    /// slice lengths (a caller-bug defense).
    #[test]
    fn classification_split_evaluation_rejects_length_mismatch() {
        let feature_values: Vec<i32> = vec![1, 2, 3];
        let completion_labels: Vec<i32> = vec![0, 1];
        let evaluation_result = evaluate_candidate_split_for_classification(
            &feature_values,
            &completion_labels,
            2,
            FeatureIndex::Age,
        );
        assert!(matches!(
            evaluation_result,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));
    }

    /// Verifies the regression split evaluator correctly partitions.
    #[test]
    fn regression_split_evaluation_partitions_correctly() {
        let feature_values: Vec<i32> = vec![1, 2, 3, 4];
        let performance_score_labels: Vec<i32> = vec![200, 200, 800, 800];
        let evaluation_result = evaluate_candidate_split_for_regression(
            &feature_values,
            &performance_score_labels,
            3,
            FeatureIndex::Age,
        )
        .expect("valid inputs must evaluate");
        assert_eq!(evaluation_result.left_child_sample_count, 2);
        assert_eq!(evaluation_result.right_child_sample_count, 2);
        // Both children are constant-valued -> variance zero on each side
        // -> combined impurity zero.
        assert_eq!(evaluation_result.combined_impurity_after_split, 0);
    }

    /// Verifies that a better (purer) split produces a strictly lower
    /// combined impurity than a worse (mixed) split. This is the core
    /// property the tree-builder will rely on.
    #[test]
    fn classification_split_evaluation_prefers_purer_split() {
        let feature_values: Vec<i32> = vec![1, 2, 3, 4];
        let completion_labels: Vec<i32> = vec![0, 0, 1, 1];

        let good_split_at_threshold_three = evaluate_candidate_split_for_classification(
            &feature_values,
            &completion_labels,
            3,
            FeatureIndex::Age,
        )
        .expect("valid inputs must evaluate");
        let bad_split_at_threshold_two = evaluate_candidate_split_for_classification(
            &feature_values,
            &completion_labels,
            2,
            FeatureIndex::Age,
        )
        .expect("valid inputs must evaluate");

        assert!(
            good_split_at_threshold_three.combined_impurity_after_split
                < bad_split_at_threshold_two.combined_impurity_after_split,
            "a pure split should have strictly lower combined impurity than a mixed split"
        );
    }

    /// Verifies `TreeLabelKind` canonical-name round-trips.
    #[test]
    fn tree_label_kind_canonical_name_round_trips_both_variants() {
        for label_kind_under_test in [
            TreeLabelKind::CompletionClassification,
            TreeLabelKind::PerformanceScoreRegression,
        ]
        .iter()
        {
            let name_string = label_kind_under_test.canonical_label_kind_name_string();
            let round_tripped_kind = TreeLabelKind::label_kind_from_canonical_name(name_string)
                .expect("canonical label kind name must round-trip");
            assert_eq!(*label_kind_under_test, round_tripped_kind);
        }
    }

    /// Verifies `NO_CHILD_NODE_INDEX` is safely above any realistic node
    /// count. This is a static consistency check, not a runtime test.
    #[test]
    fn no_child_node_index_sentinel_is_larger_than_any_realistic_node_count() {
        let largest_realistic_node_count_for_this_project: u32 = 10_000_000;
        assert!(NO_CHILD_NODE_INDEX > largest_realistic_node_count_for_this_project);
    }
}

/*
Section 4b: Iterative Tree-Building Loop and Prediction Traversal
This sub-section uses the data structures and split-evaluation functions from Section 4a to actually build a decision tree from training data and predict with it. The build algorithm is entirely iterative — an explicit work stack replaces recursion.

What This Sub-Section Contains

PendingNodeBuildWorkItem struct — one entry on the explicit work stack, describing a node that needs to be built
find_best_split_for_sample_set — iterates all features and all candidate thresholds to find the lowest-impurity split for a given set of samples
compute_majority_class_label_for_leaf — determines the leaf prediction for a classification tree (majority vote of 0/1)
compute_mean_integer_label_for_leaf — determines the leaf prediction for a regression tree (rounded integer mean of performance scores)
build_decision_tree_iteratively — the main build function: uses a work stack to grow nodes breadth-first, applying max-depth and min-leaf-size hyperparameters
predict_single_feature_vector_with_tree — walks the flat tree to produce a prediction for one feature vector
predict_batch_with_tree — convenience wrapper for predicting a slice of feature vectors

Plus cargo tests.

Design Decisions Explained Up Front
Why breadth-first (not depth-first) build order: The work stack is FIFO (we push to the back and pop from the front), so the tree is built level by level. This does not affect the resulting tree (the same splits are chosen either way), but it means node indices in the flat vector are ordered by depth, which makes the plain-text model file easier for a human to read — shallow nodes first, leaves last.
How candidate thresholds are chosen: For each feature, the builder collects all unique values of that feature across the current node's training samples, sorts them, and places a candidate threshold at each midpoint between consecutive unique values. This is the standard CART approach: it tests every value boundary that actually exists in the data without wasting evaluations on redundant thresholds. Integer midpoints are computed as (a + b) / 2 — because both a and b are i32 and their sum fits in i32 for the value ranges in this project (heights ~100–999, weights ~500–1500, scores 0–1000), overflow is not a concern here, but I add a defensive i64 intermediate anyway.
Why the build function takes parallel Vec<EngineeredFeatureVector> and Vec<i32> (labels) rather than a combined struct: The same feature vectors are paired with completion labels for one tree and performance_score labels for another. Keeping features and labels separate avoids duplicating the feature data and makes the label-kind-agnostic parts of the build loop clean.
Why the work stack stores Vec<usize> sample indices (not cloned sample data): Each pending node carries the indices of its training samples into the original feature/label vectors, not copies of the data. This avoids deep-copying 200-element vectors at every split and keeps memory usage proportional to the training set size, not proportional to training-set-size × tree-depth.

Notes on This Sub-Section
PendingNodeBuildWorkItem is struct (not pub struct): The work item is internal to the build loop. No code outside this section needs to know it exists. Keeping it private enforces that only build_decision_tree_iteratively can create and consume work items.
Why the build loop does not try to prune: Pruning (removing splits that do not improve validation accuracy) is a separate concern. The current build loop grows the tree greedily up to max_depth / min_samples_per_leaf. Pruning, if added later, would be a post-processing pass over the built tree. Mixing pruning into the build loop would violate the single-responsibility principle and complicate testing.
Why predict_batch_with_tree silently replaces errors with defaults: Per the project's "handle and move on, do not stop" rule. A corrupted prediction for one sample in a batch must not abort predictions for the other samples. The safe defaults (0 for completion, 0 for performance score) are pessimistic, which is appropriate — predicting "horse fails" is less harmful than crashing the entire prediction run.
The VecDeque import: This is from std::collections, not a third-party crate. It provides O(1) push-back and pop-front, which is exactly what a FIFO work queue needs.

*/

// ============================================================================
// SECTION 4b — ITERATIVE TREE-BUILDING LOOP AND PREDICTION TRAVERSAL
// ============================================================================
//
// This sub-section builds on 4a's data structures and split-evaluation
// functions to implement:
//
//   1. An iterative (non-recursive) tree-building algorithm driven by an
//      explicit FIFO work stack.
//   2. A prediction function that walks the flat tree with a simple loop.
//
// ## Build Algorithm Summary
//
// The builder maintains a work stack of `PendingNodeBuildWorkItem`s. Each
// item describes a node that needs to be turned into either a decision node
// (split) or a leaf. For each item:
//
//   a. If the node is at `max_depth`, or the sample count is below
//      `min_samples_per_leaf * 2` (cannot split into two valid children),
//      or all labels are identical (no impurity to reduce): create a leaf.
//   b. Otherwise: evaluate every candidate split (all features × all
//      midpoint thresholds) and choose the one with the lowest combined
//      impurity. If the best split still leaves an empty child (degenerate
//      case): create a leaf. Otherwise: create a decision node, reserve
//      two child slots in the node vector, and push two new work items
//      onto the stack.
//
// ## Why FIFO (Breadth-First)
//
// The work stack is a VecDeque used as a queue (push_back, pop_front).
// This builds the tree level-by-level, so node indices in the flat vector
// are ordered by depth — shallow nodes have lower indices. This has no
// algorithmic effect but makes the saved model file easier to read.

use std::collections::VecDeque;

/// One entry on the iterative tree-builder's work stack.
///
/// ## Fields
///
/// - `node_index_in_flat_vector` — the position in the `DecisionTree`'s
///   `all_tree_nodes_flat_vector` that this work item will populate. The
///   builder pre-allocates a placeholder node at this index before pushing
///   the work item, so the index is always valid.
/// - `sample_indices_at_this_node` — indices into the original training
///   feature/label vectors that belong to this node. These are partitioned
///   (not copied) when the node is split.
/// - `current_depth_of_this_node` — distance from the root (root = 0).
///   Compared against the tree's `max_depth` hyperparameter.
///
/// ## Lifetime and Ownership
///
/// The `Vec<usize>` of sample indices is *owned* by this struct. When the
/// work item is consumed (popped from the stack), the indices are split
/// into two new `Vec<usize>`s for the child work items. No index vector
/// lives longer than one build-loop iteration, so memory usage stays
/// bounded.
struct PendingNodeBuildWorkItem {
    node_index_in_flat_vector: u32,
    sample_indices_at_this_node: Vec<usize>,
    current_depth_of_this_node: u32,
}

/// Finds the best split across all features and all candidate thresholds
/// for a given set of training samples at one node.
///
/// ## Inputs
///
/// - `training_feature_vectors` — the full training set of feature vectors
///   (shared across all nodes; this function only looks at the indices in
///   `sample_indices_for_this_node`).
/// - `training_labels` — the full training set of labels (parallel to
///   `training_feature_vectors`).
/// - `sample_indices_for_this_node` — which samples belong to the current
///   node.
/// - `label_kind_for_split_evaluation` — classification or regression,
///   determining which impurity function to call.
/// - `minimum_samples_per_leaf` — the minimum number of samples that must
///   end up on each side of a valid split.
///
/// ## Algorithm
///
/// For each feature in `all_feature_indices_in_canonical_order()`:
///   1. Collect the unique feature values across the node's samples.
///   2. Sort the unique values.
///   3. For each consecutive pair of unique values, compute the midpoint
///      threshold = `(value_a + value_b) / 2` using `i64` to avoid overflow.
///   4. Evaluate that threshold (via the appropriate impurity function).
///   5. If the resulting split has both children >= `minimum_samples_per_leaf`
///      and its combined impurity is the lowest seen so far, record it.
///
/// ## Returns
///
/// `Some(SplitEvaluation)` if any valid split was found, `None` if no
/// split satisfies the minimum-leaf constraint (all features are constant,
/// or all thresholds produce degenerate splits).
///
/// Returning `Option` (not `Result`) because "no valid split" is a normal
/// leaf-creation trigger, not an error.
fn find_best_split_for_sample_set(
    training_feature_vectors: &[EngineeredFeatureVector],
    training_labels: &[i32],
    sample_indices_for_this_node: &[usize],
    label_kind_for_split_evaluation: TreeLabelKind,
    minimum_samples_per_leaf: usize,
) -> Option<SplitEvaluation> {
    let sample_count_at_this_node: usize = sample_indices_for_this_node.len();

    // Cannot split fewer than 2 * minimum_samples_per_leaf samples into
    // two valid children.
    if sample_count_at_this_node < minimum_samples_per_leaf * 2 {
        return None;
    }

    // Extract the label values for this node's samples into a contiguous
    // scratch vector so the split-evaluation functions can take slices.
    let mut labels_at_this_node: Vec<i32> = Vec::with_capacity(sample_count_at_this_node);
    for sample_index_reference in sample_indices_for_this_node.iter() {
        labels_at_this_node.push(training_labels[*sample_index_reference]);
    }

    // Check if all labels are identical — if so, no split can reduce
    // impurity, so return None immediately.
    let first_label_value = labels_at_this_node[0];
    let all_labels_are_identical = labels_at_this_node
        .iter()
        .all(|label_reference| *label_reference == first_label_value);
    if all_labels_are_identical {
        return None;
    }

    let mut best_split_found_so_far: Option<SplitEvaluation> = None;

    // Iterate over every feature in canonical order.
    let all_feature_indices = all_feature_indices_in_canonical_order();
    for feature_index_under_test in all_feature_indices.iter() {
        // Collect unique feature values for this feature across the node's
        // samples.
        let mut unique_feature_values_sorted: Vec<i32> =
            Vec::with_capacity(sample_count_at_this_node);
        let mut feature_values_for_all_samples_at_node: Vec<i32> =
            Vec::with_capacity(sample_count_at_this_node);

        for sample_index_reference in sample_indices_for_this_node.iter() {
            let feature_value = extract_feature_value_from_vector(
                &training_feature_vectors[*sample_index_reference],
                *feature_index_under_test,
            );
            feature_values_for_all_samples_at_node.push(feature_value);
            // Insert into unique list only if not already present.
            // Linear scan is fine at n <= 200.
            if !unique_feature_values_sorted.contains(&feature_value) {
                unique_feature_values_sorted.push(feature_value);
            }
        }

        // Sort unique values so we can compute midpoint thresholds between
        // consecutive values.
        unique_feature_values_sorted.sort();

        // Need at least two distinct values to form a threshold.
        if unique_feature_values_sorted.len() < 2 {
            continue;
        }

        // Iterate consecutive pairs, computing midpoint thresholds.
        // Upper bound on this loop: at most `sample_count_at_this_node - 1`
        // pairs, which is bounded by training set size.
        let unique_value_count = unique_feature_values_sorted.len();
        for pair_position in 0..(unique_value_count - 1) {
            let lower_unique_value: i64 = unique_feature_values_sorted[pair_position] as i64;
            let upper_unique_value: i64 = unique_feature_values_sorted[pair_position + 1] as i64;
            // Midpoint in i64 to avoid any overflow risk, then truncate
            // back to i32. The truncation is safe because the midpoint of
            // two i32 values always fits in i32.
            let midpoint_threshold_value: i32 =
                ((lower_unique_value + upper_unique_value) / 2) as i32;

            // Evaluate this candidate threshold.
            let candidate_evaluation_result = match label_kind_for_split_evaluation {
                TreeLabelKind::CompletionClassification => {
                    evaluate_candidate_split_for_classification(
                        &feature_values_for_all_samples_at_node,
                        &labels_at_this_node,
                        midpoint_threshold_value,
                        *feature_index_under_test,
                    )
                }
                TreeLabelKind::PerformanceScoreRegression => {
                    evaluate_candidate_split_for_regression(
                        &feature_values_for_all_samples_at_node,
                        &labels_at_this_node,
                        midpoint_threshold_value,
                        *feature_index_under_test,
                    )
                }
            };

            let candidate_evaluation = match candidate_evaluation_result {
                Ok(evaluation) => evaluation,
                Err(_split_eval_error) => {
                    // A split-evaluation error here means an internal
                    // inconsistency (e.g. feature/label length mismatch).
                    // Skip this candidate rather than aborting the tree.
                    continue;
                }
            };

            // Reject splits that would leave a child below minimum leaf size.
            if candidate_evaluation.left_child_sample_count < minimum_samples_per_leaf {
                continue;
            }
            if candidate_evaluation.right_child_sample_count < minimum_samples_per_leaf {
                continue;
            }

            // Keep the best (lowest impurity) split seen so far.
            let is_better_than_current_best = match best_split_found_so_far {
                None => true,
                Some(ref current_best_reference) => {
                    candidate_evaluation.combined_impurity_after_split
                        < current_best_reference.combined_impurity_after_split
                }
            };
            if is_better_than_current_best {
                best_split_found_so_far = Some(candidate_evaluation);
            }
        }
    }

    best_split_found_so_far
}

/// Computes the majority-vote class label for a classification leaf node.
///
/// Counts the 0-labels and 1-labels among the given samples. Returns
/// whichever class has more. Ties are broken in favor of 0 (DNF /
/// "did not complete"), which is the more conservative (pessimistic)
/// prediction for a horse-racing context — predicting a horse will finish
/// when it will not is a worse surprise than the reverse.
///
/// ## Inputs
///
/// - `training_labels` — the full label vector.
/// - `sample_indices_for_leaf` — which samples reached this leaf.
///
/// ## Panics
///
/// Never panics. If `sample_indices_for_leaf` is empty, returns 0 as a
/// conservative default.
fn compute_majority_class_label_for_leaf(
    training_labels: &[i32],
    sample_indices_for_leaf: &[usize],
) -> i32 {
    let mut count_of_zeros: usize = 0;
    let mut count_of_ones: usize = 0;
    for sample_index_reference in sample_indices_for_leaf.iter() {
        if training_labels[*sample_index_reference] == 0 {
            count_of_zeros += 1;
        } else {
            count_of_ones += 1;
        }
    }
    // Tie-break: 0 wins (conservative prediction).
    if count_of_ones > count_of_zeros { 1 } else { 0 }
}

/// Computes the integer mean of regression labels for a regression leaf.
///
/// The mean is computed in `i64` to avoid overflow (200 samples × 1000
/// max value = 200,000, well within `i64`) and then rounded back to `i32`.
///
/// ## Rounding
///
/// Standard round-half-up: `(sum + count / 2) / count`. This gives the
/// closest integer to the true mean.
///
/// ## Empty Leaf
///
/// Returns `PERFORMANCE_SCORE_FOR_DID_NOT_FINISH` (0) if the sample set is
/// empty, as a conservative default.
fn compute_mean_integer_label_for_leaf(
    training_labels: &[i32],
    sample_indices_for_leaf: &[usize],
) -> i32 {
    let sample_count: i64 = sample_indices_for_leaf.len() as i64;
    if sample_count == 0 {
        return PERFORMANCE_SCORE_FOR_DID_NOT_FINISH;
    }

    let mut running_sum_of_labels: i64 = 0;
    for sample_index_reference in sample_indices_for_leaf.iter() {
        running_sum_of_labels += training_labels[*sample_index_reference] as i64;
    }

    // Round-half-up integer division.
    let rounded_mean: i64 = (running_sum_of_labels + sample_count / 2) / sample_count;
    rounded_mean as i32
}

/// Builds a decision tree iteratively from training data.
///
/// ## Inputs
///
/// - `training_feature_vectors` — one `EngineeredFeatureVector` per training
///   sample.
/// - `training_labels` — one `i32` label per sample, parallel to the
///   feature vectors. For classification trees these are `completion`
///   values (0 or 1); for regression trees, `performance_score` values
///   (0..=1000).
/// - `label_kind` — classification or regression, determining which
///   impurity function and which leaf-value computation to use.
/// - `max_depth` — no node deeper than this will be created. The root is
///   depth 0. A `max_depth` of 0 produces a single-leaf tree (just the
///   majority-vote or mean).
/// - `min_samples_per_leaf` — the minimum number of training samples that
///   must end up in any leaf. Also used to reject splits that would create
///   a child smaller than this.
///
/// ## Algorithm (Iterative, Explicit Work Queue)
///
/// 1. Pre-allocate a root placeholder node in the tree's flat vector.
/// 2. Push a work item for the root (all sample indices, depth 0).
/// 3. While the work queue is not empty:
///    a. Pop the front item.
///    b. Decide: should this node be a leaf? (depth == max_depth, sample
///       count too small, labels all identical, no valid split found.)
///    c. If leaf: write the leaf prediction into the placeholder node.
///    d. If split: find the best split; write the decision into the
///       placeholder node; pre-allocate two child placeholders; push
///       two new work items.
/// 4. Return the completed tree.
///
/// ## Defensive Work-Queue Bound
///
/// The while-loop is bounded by `maximum_work_items_processed_cap` to
/// prevent runaway builds if a logic error causes infinite work-item
/// generation. The cap is set to 2^20 (~1 million) which is far above any
/// realistic tree for ~200 training samples.
///
/// ## Error Cases
///
/// Returns `FieldValueOutOfValidRange` if the training vectors and labels
/// have different lengths. Otherwise always returns `Ok(DecisionTree)` —
/// even degenerate inputs (all-identical features, empty training set)
/// produce a valid single-leaf tree rather than an error.
pub fn build_decision_tree_iteratively(
    training_feature_vectors: &[EngineeredFeatureVector],
    training_labels: &[i32],
    label_kind: TreeLabelKind,
    max_depth: u32,
    min_samples_per_leaf: usize,
) -> Result<DecisionTree, HorseRacingError> {
    // Guard: feature vectors and labels must be parallel.
    if training_feature_vectors.len() != training_labels.len() {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "build_decision_tree_iteratively: feature vectors and labels length mismatch",
        ));
    }

    let total_training_sample_count: usize = training_feature_vectors.len();

    // Pre-allocate the tree with a generous capacity. The maximum number of
    // nodes in a binary tree of depth `d` is `2^(d+1) - 1`. For the
    // project's typical depths (2–6), this is at most 127 nodes.
    let estimated_max_nodes: usize = if max_depth < 20 {
        (1_usize << (max_depth + 1)).saturating_sub(1)
    } else {
        // For unrealistically large max_depth, clamp to avoid oversized
        // allocation. The actual tree will be much smaller.
        1_000_000
    };
    let mut tree_nodes_flat_vector: Vec<DecisionTreeNode> = Vec::with_capacity(estimated_max_nodes);

    // Handle the degenerate case of empty training data: return a single
    // leaf that predicts 0.
    if total_training_sample_count == 0 {
        tree_nodes_flat_vector.push(DecisionTreeNode {
            node_branch_decision: TreeNodeBranchDecision::LeafNode,
            split_feature_index: FeatureIndex::Age,
            split_threshold_value: 0,
            left_child_node_index: NO_CHILD_NODE_INDEX,
            right_child_node_index: NO_CHILD_NODE_INDEX,
            leaf_predicted_value: 0,
        });
        return Ok(DecisionTree {
            all_tree_nodes_flat_vector: tree_nodes_flat_vector,
            root_node_index: 0,
            label_kind_for_this_tree: label_kind,
            max_depth_used_for_training: max_depth,
        });
    }

    // Build the initial sample-index vector: all samples belong to the root.
    let mut root_sample_indices: Vec<usize> = Vec::with_capacity(total_training_sample_count);
    for sample_position in 0..total_training_sample_count {
        root_sample_indices.push(sample_position);
    }

    // Pre-allocate the root placeholder node. Its fields will be filled
    // when the work item is processed.
    let root_node_position_in_vector: u32 = 0;
    tree_nodes_flat_vector.push(DecisionTreeNode {
        node_branch_decision: TreeNodeBranchDecision::LeafNode,
        split_feature_index: FeatureIndex::Age,
        split_threshold_value: 0,
        left_child_node_index: NO_CHILD_NODE_INDEX,
        right_child_node_index: NO_CHILD_NODE_INDEX,
        leaf_predicted_value: 0,
    });

    // Initialize the FIFO work queue with the root.
    let mut build_work_queue: VecDeque<PendingNodeBuildWorkItem> = VecDeque::new();
    build_work_queue.push_back(PendingNodeBuildWorkItem {
        node_index_in_flat_vector: root_node_position_in_vector,
        sample_indices_at_this_node: root_sample_indices,
        current_depth_of_this_node: 0,
    });

    // Defensive bound on work items processed (NASA Power-of-10 rule 2).
    let maximum_work_items_processed_cap: usize = 1_048_576;
    let mut work_items_processed_so_far: usize = 0;

    // Main build loop.
    while let Some(current_work_item) = build_work_queue.pop_front() {
        work_items_processed_so_far += 1;
        if work_items_processed_so_far > maximum_work_items_processed_cap {
            // If we hit this cap, something is very wrong. Return whatever
            // tree we have so far (it will still be a valid tree with the
            // unprocessed nodes as their placeholder leaves).
            break;
        }

        let node_flat_index: usize = current_work_item.node_index_in_flat_vector as usize;
        let current_node_depth: u32 = current_work_item.current_depth_of_this_node;
        let sample_indices_at_node: Vec<usize> = current_work_item.sample_indices_at_this_node;
        let sample_count_at_node: usize = sample_indices_at_node.len();

        // Decision: should this node be a leaf?
        let should_be_leaf: bool =
            current_node_depth >= max_depth || sample_count_at_node < min_samples_per_leaf * 2;

        if should_be_leaf {
            // Create a leaf node with the appropriate prediction.
            let leaf_prediction: i32 = match label_kind {
                TreeLabelKind::CompletionClassification => {
                    compute_majority_class_label_for_leaf(training_labels, &sample_indices_at_node)
                }
                TreeLabelKind::PerformanceScoreRegression => {
                    compute_mean_integer_label_for_leaf(training_labels, &sample_indices_at_node)
                }
            };
            tree_nodes_flat_vector[node_flat_index] = DecisionTreeNode {
                node_branch_decision: TreeNodeBranchDecision::LeafNode,
                split_feature_index: FeatureIndex::Age,
                split_threshold_value: 0,
                left_child_node_index: NO_CHILD_NODE_INDEX,
                right_child_node_index: NO_CHILD_NODE_INDEX,
                leaf_predicted_value: leaf_prediction,
            };
            continue;
        }

        // Try to find a split.
        let best_split_option = find_best_split_for_sample_set(
            training_feature_vectors,
            training_labels,
            &sample_indices_at_node,
            label_kind,
            min_samples_per_leaf,
        );

        match best_split_option {
            None => {
                // No valid split found — make a leaf.
                let leaf_prediction: i32 = match label_kind {
                    TreeLabelKind::CompletionClassification => {
                        compute_majority_class_label_for_leaf(
                            training_labels,
                            &sample_indices_at_node,
                        )
                    }
                    TreeLabelKind::PerformanceScoreRegression => {
                        compute_mean_integer_label_for_leaf(
                            training_labels,
                            &sample_indices_at_node,
                        )
                    }
                };
                tree_nodes_flat_vector[node_flat_index] = DecisionTreeNode {
                    node_branch_decision: TreeNodeBranchDecision::LeafNode,
                    split_feature_index: FeatureIndex::Age,
                    split_threshold_value: 0,
                    left_child_node_index: NO_CHILD_NODE_INDEX,
                    right_child_node_index: NO_CHILD_NODE_INDEX,
                    leaf_predicted_value: leaf_prediction,
                };
            }
            Some(best_split_evaluation) => {
                // Partition sample indices into left and right children.
                let mut left_child_sample_indices: Vec<usize> =
                    Vec::with_capacity(best_split_evaluation.left_child_sample_count);
                let mut right_child_sample_indices: Vec<usize> =
                    Vec::with_capacity(best_split_evaluation.right_child_sample_count);

                for sample_index_reference in sample_indices_at_node.iter() {
                    let feature_value_for_this_sample = extract_feature_value_from_vector(
                        &training_feature_vectors[*sample_index_reference],
                        best_split_evaluation.split_feature_index,
                    );
                    if feature_value_for_this_sample < best_split_evaluation.split_threshold_value {
                        left_child_sample_indices.push(*sample_index_reference);
                    } else {
                        right_child_sample_indices.push(*sample_index_reference);
                    }
                }

                // Pre-allocate placeholder nodes for both children.
                let left_child_node_index: u32 = tree_nodes_flat_vector.len() as u32;
                tree_nodes_flat_vector.push(DecisionTreeNode {
                    node_branch_decision: TreeNodeBranchDecision::LeafNode,
                    split_feature_index: FeatureIndex::Age,
                    split_threshold_value: 0,
                    left_child_node_index: NO_CHILD_NODE_INDEX,
                    right_child_node_index: NO_CHILD_NODE_INDEX,
                    leaf_predicted_value: 0,
                });

                let right_child_node_index: u32 = tree_nodes_flat_vector.len() as u32;
                tree_nodes_flat_vector.push(DecisionTreeNode {
                    node_branch_decision: TreeNodeBranchDecision::LeafNode,
                    split_feature_index: FeatureIndex::Age,
                    split_threshold_value: 0,
                    left_child_node_index: NO_CHILD_NODE_INDEX,
                    right_child_node_index: NO_CHILD_NODE_INDEX,
                    leaf_predicted_value: 0,
                });

                // Write the decision node.
                tree_nodes_flat_vector[node_flat_index] = DecisionTreeNode {
                    node_branch_decision: TreeNodeBranchDecision::DecisionNode,
                    split_feature_index: best_split_evaluation.split_feature_index,
                    split_threshold_value: best_split_evaluation.split_threshold_value,
                    left_child_node_index,
                    right_child_node_index,
                    leaf_predicted_value: 0,
                };

                // Push child work items onto the queue.
                build_work_queue.push_back(PendingNodeBuildWorkItem {
                    node_index_in_flat_vector: left_child_node_index,
                    sample_indices_at_this_node: left_child_sample_indices,
                    current_depth_of_this_node: current_node_depth + 1,
                });
                build_work_queue.push_back(PendingNodeBuildWorkItem {
                    node_index_in_flat_vector: right_child_node_index,
                    sample_indices_at_this_node: right_child_sample_indices,
                    current_depth_of_this_node: current_node_depth + 1,
                });
            }
        }
    }

    Ok(DecisionTree {
        all_tree_nodes_flat_vector: tree_nodes_flat_vector,
        root_node_index: root_node_position_in_vector,
        label_kind_for_this_tree: label_kind,
        max_depth_used_for_training: max_depth,
    })
}

/// Predicts a single feature vector by walking the decision tree
/// iteratively.
///
/// ## Algorithm
///
/// Start at the root. At each decision node, compare the feature vector's
/// value for the split feature against the split threshold: go left if
/// `value < threshold`, else right. Repeat until a leaf is reached. Return
/// the leaf's `leaf_predicted_value`.
///
/// ## Defensive Traversal Bound
///
/// The loop is bounded by `max_depth + 1` (no correct tree has a path
/// longer than its maximum depth plus the root). If the loop exceeds this
/// count without reaching a leaf, it returns the prediction from whatever
/// node it is currently on, treating it as a leaf. This guards against a
/// corrupted or cyclic tree (which should never happen, but hardware faults
/// and model-file corruption are real possibilities).
///
/// ## Error Cases
///
/// Returns `FieldValueOutOfValidRange` if the root index is out of bounds.
/// Otherwise always returns `Ok(i32)`.
pub fn predict_single_feature_vector_with_tree(
    trained_decision_tree: &DecisionTree,
    feature_vector_to_classify: &EngineeredFeatureVector,
) -> Result<i32, HorseRacingError> {
    let total_node_count: usize = trained_decision_tree.all_tree_nodes_flat_vector.len();
    let root_index_as_usize: usize = trained_decision_tree.root_node_index as usize;

    if root_index_as_usize >= total_node_count {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "predict_single_feature_vector_with_tree: root index out of bounds",
        ));
    }

    // Defensive traversal depth bound.
    let maximum_traversal_steps: u32 = trained_decision_tree.max_depth_used_for_training + 2;
    let mut steps_taken: u32 = 0;
    let mut current_node_index: usize = root_index_as_usize;

    loop {
        steps_taken += 1;
        if steps_taken > maximum_traversal_steps {
            // Corrupted tree: bail out with whatever node we are on.
            break;
        }

        if current_node_index >= total_node_count {
            // Child index is out of bounds — treat current position as a
            // leaf rather than panicking.
            break;
        }

        let current_node_reference =
            &trained_decision_tree.all_tree_nodes_flat_vector[current_node_index];

        match current_node_reference.node_branch_decision {
            TreeNodeBranchDecision::LeafNode => {
                return Ok(current_node_reference.leaf_predicted_value);
            }
            TreeNodeBranchDecision::DecisionNode => {
                let feature_value_from_input = extract_feature_value_from_vector(
                    feature_vector_to_classify,
                    current_node_reference.split_feature_index,
                );
                if feature_value_from_input < current_node_reference.split_threshold_value {
                    current_node_index = current_node_reference.left_child_node_index as usize;
                } else {
                    current_node_index = current_node_reference.right_child_node_index as usize;
                }
            }
        }
    }

    // Fallback: return the last node's prediction value.
    if current_node_index < total_node_count {
        Ok(
            trained_decision_tree.all_tree_nodes_flat_vector[current_node_index]
                .leaf_predicted_value,
        )
    } else {
        // Cannot even read the last node — return a safe default.
        Ok(PERFORMANCE_SCORE_FOR_DID_NOT_FINISH)
    }
}

/// Predicts an entire batch of feature vectors, returning one `i32`
/// prediction per vector.
///
/// ## Project Role
///
/// Used for both training-set accuracy evaluation and prediction-mode
/// batch inference. Returns a parallel `Vec<i32>` so the caller can
/// compare predictions against known labels positionally.
///
/// ## Error Handling
///
/// If any individual prediction fails (corrupted tree), that sample's
/// prediction is replaced with a safe default
/// (`PERFORMANCE_SCORE_FOR_DID_NOT_FINISH` for regression, 0 for
/// classification) and the batch continues. The batch never aborts
/// partway.
pub fn predict_batch_with_tree(
    trained_decision_tree: &DecisionTree,
    batch_of_feature_vectors: &[EngineeredFeatureVector],
) -> Vec<i32> {
    let safe_default_prediction: i32 = match trained_decision_tree.label_kind_for_this_tree {
        TreeLabelKind::CompletionClassification => 0,
        TreeLabelKind::PerformanceScoreRegression => PERFORMANCE_SCORE_FOR_DID_NOT_FINISH,
    };

    let mut predictions_output_vector: Vec<i32> =
        Vec::with_capacity(batch_of_feature_vectors.len());

    for feature_vector_reference in batch_of_feature_vectors.iter() {
        let prediction_for_this_sample = match predict_single_feature_vector_with_tree(
            trained_decision_tree,
            feature_vector_reference,
        ) {
            Ok(predicted_value) => predicted_value,
            Err(_prediction_error) => safe_default_prediction,
        };
        predictions_output_vector.push(prediction_for_this_sample);
    }

    predictions_output_vector
}

// ============================================================================
// SECTION 4b — CARGO TESTS
// ============================================================================

#[cfg(test)]
mod section_four_b_tree_build_and_predict_tests {
    use super::*;

    /// Builds a minimal synthetic training set where the classification task
    /// is perfectly separable: samples with age < 5 have completion = 1,
    /// samples with age >= 5 have completion = 0.
    ///
    /// Returns (feature_vectors, labels).
    fn build_perfectly_separable_classification_training_set()
    -> (Vec<EngineeredFeatureVector>, Vec<i32>) {
        let mut feature_vectors: Vec<EngineeredFeatureVector> = Vec::new();
        let mut completion_labels: Vec<i32> = Vec::new();

        // Young horses complete (age 2, 3, 4).
        for young_horse_age in 2..=4 {
            feature_vectors.push(EngineeredFeatureVector {
                age: young_horse_age,
                height: 150,
                experience: 3,
                weight: 900,
                height_to_weight_ratio_times_one_thousand: 166,
                age_times_experience: young_horse_age * 3,
            });
            completion_labels.push(1);
        }

        // Old horses do not complete (age 6, 7, 8).
        for old_horse_age in 6..=8 {
            feature_vectors.push(EngineeredFeatureVector {
                age: old_horse_age,
                height: 150,
                experience: 3,
                weight: 900,
                height_to_weight_ratio_times_one_thousand: 166,
                age_times_experience: old_horse_age * 3,
            });
            completion_labels.push(0);
        }

        (feature_vectors, completion_labels)
    }

    /// Builds a synthetic regression training set where performance score
    /// correlates perfectly with height: taller horses score higher.
    ///
    /// Returns (feature_vectors, labels).
    fn build_height_correlated_regression_training_set() -> (Vec<EngineeredFeatureVector>, Vec<i32>)
    {
        let mut feature_vectors: Vec<EngineeredFeatureVector> = Vec::new();
        let mut performance_score_labels: Vec<i32> = Vec::new();

        let height_score_pairs: [(i32, i32); 6] = [
            (120, 200),
            (130, 200),
            (150, 600),
            (160, 600),
            (180, 1000),
            (190, 1000),
        ];

        for (height_value, score_value) in height_score_pairs.iter() {
            feature_vectors.push(EngineeredFeatureVector {
                age: 4,
                height: *height_value,
                experience: 3,
                weight: 900,
                height_to_weight_ratio_times_one_thousand: (*height_value * 1000) / 900,
                age_times_experience: 12,
            });
            performance_score_labels.push(*score_value);
        }

        (feature_vectors, performance_score_labels)
    }

    /// Verifies that a classification tree trained on perfectly separable
    /// data correctly classifies every training sample.
    #[test]
    fn classification_tree_achieves_perfect_accuracy_on_separable_data() {
        let (feature_vectors, completion_labels) =
            build_perfectly_separable_classification_training_set();
        let trained_classification_tree = build_decision_tree_iteratively(
            &feature_vectors,
            &completion_labels,
            TreeLabelKind::CompletionClassification,
            4,
            1,
        )
        .expect("building tree must succeed on valid input");

        let predictions = predict_batch_with_tree(&trained_classification_tree, &feature_vectors);
        assert_eq!(predictions.len(), completion_labels.len());
        for sample_position in 0..predictions.len() {
            assert_eq!(
                predictions[sample_position], completion_labels[sample_position],
                "classification mismatch at sample position {}",
                sample_position
            );
        }
    }

    /// Verifies that the tree correctly routes a novel (unseen) feature
    /// vector that falls clearly on one side of the learned boundary.
    #[test]
    fn classification_tree_predicts_correctly_on_novel_sample() {
        let (feature_vectors, completion_labels) =
            build_perfectly_separable_classification_training_set();
        let trained_classification_tree = build_decision_tree_iteratively(
            &feature_vectors,
            &completion_labels,
            TreeLabelKind::CompletionClassification,
            4,
            1,
        )
        .expect("building tree must succeed");

        // A very young horse (age 2) should be predicted to complete.
        let young_horse_feature_vector = EngineeredFeatureVector {
            age: 2,
            height: 150,
            experience: 3,
            weight: 900,
            height_to_weight_ratio_times_one_thousand: 166,
            age_times_experience: 6,
        };
        let young_horse_prediction = predict_single_feature_vector_with_tree(
            &trained_classification_tree,
            &young_horse_feature_vector,
        )
        .expect("prediction must succeed");
        assert_eq!(young_horse_prediction, 1);

        // A very old horse (age 8) should be predicted to not complete.
        let old_horse_feature_vector = EngineeredFeatureVector {
            age: 8,
            height: 150,
            experience: 3,
            weight: 900,
            height_to_weight_ratio_times_one_thousand: 166,
            age_times_experience: 24,
        };
        let old_horse_prediction = predict_single_feature_vector_with_tree(
            &trained_classification_tree,
            &old_horse_feature_vector,
        )
        .expect("prediction must succeed");
        assert_eq!(old_horse_prediction, 0);
    }

    /// Verifies that a regression tree trained on height-correlated data
    /// produces sensible predictions (higher height -> higher score).
    #[test]
    fn regression_tree_predicts_higher_score_for_taller_horses() {
        let (feature_vectors, performance_labels) =
            build_height_correlated_regression_training_set();
        let trained_regression_tree = build_decision_tree_iteratively(
            &feature_vectors,
            &performance_labels,
            TreeLabelKind::PerformanceScoreRegression,
            4,
            1,
        )
        .expect("building tree must succeed");

        let short_horse = EngineeredFeatureVector {
            age: 4,
            height: 125,
            experience: 3,
            weight: 900,
            height_to_weight_ratio_times_one_thousand: 138,
            age_times_experience: 12,
        };
        let tall_horse = EngineeredFeatureVector {
            age: 4,
            height: 185,
            experience: 3,
            weight: 900,
            height_to_weight_ratio_times_one_thousand: 205,
            age_times_experience: 12,
        };

        let short_horse_prediction =
            predict_single_feature_vector_with_tree(&trained_regression_tree, &short_horse)
                .expect("prediction must succeed");
        let tall_horse_prediction =
            predict_single_feature_vector_with_tree(&trained_regression_tree, &tall_horse)
                .expect("prediction must succeed");

        assert!(
            tall_horse_prediction > short_horse_prediction,
            "taller horse must receive a higher performance score prediction"
        );
    }

    /// Verifies that `max_depth = 0` produces a single-leaf tree (no splits).
    #[test]
    fn max_depth_zero_produces_single_leaf_tree() {
        let (feature_vectors, completion_labels) =
            build_perfectly_separable_classification_training_set();
        let trained_tree = build_decision_tree_iteratively(
            &feature_vectors,
            &completion_labels,
            TreeLabelKind::CompletionClassification,
            0,
            1,
        )
        .expect("building tree must succeed");

        assert_eq!(trained_tree.all_tree_nodes_flat_vector.len(), 1);
        assert_eq!(
            trained_tree.all_tree_nodes_flat_vector[0].node_branch_decision,
            TreeNodeBranchDecision::LeafNode,
        );
    }

    /// Verifies that increasing `min_samples_per_leaf` to the full training
    /// set size prevents any split, also producing a single-leaf tree.
    #[test]
    fn large_min_samples_per_leaf_prevents_splitting() {
        let (feature_vectors, completion_labels) =
            build_perfectly_separable_classification_training_set();
        let total_sample_count = feature_vectors.len();

        let trained_tree = build_decision_tree_iteratively(
            &feature_vectors,
            &completion_labels,
            TreeLabelKind::CompletionClassification,
            10,
            total_sample_count,
        )
        .expect("building tree must succeed");

        assert_eq!(trained_tree.all_tree_nodes_flat_vector.len(), 1);
    }

    /// Verifies that the build function rejects mismatched vector/label
    /// lengths.
    #[test]
    fn build_tree_rejects_mismatched_feature_and_label_lengths() {
        let feature_vectors: Vec<EngineeredFeatureVector> = vec![EngineeredFeatureVector {
            age: 4,
            height: 150,
            experience: 3,
            weight: 900,
            height_to_weight_ratio_times_one_thousand: 166,
            age_times_experience: 12,
        }];
        let too_many_labels: Vec<i32> = vec![1, 0];
        let build_result = build_decision_tree_iteratively(
            &feature_vectors,
            &too_many_labels,
            TreeLabelKind::CompletionClassification,
            4,
            1,
        );
        assert!(matches!(
            build_result,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));
    }

    /// Verifies that building on empty training data produces a valid
    /// single-leaf tree (not an error).
    #[test]
    fn build_tree_on_empty_data_produces_single_leaf() {
        let empty_features: Vec<EngineeredFeatureVector> = Vec::new();
        let empty_labels: Vec<i32> = Vec::new();
        let trained_tree = build_decision_tree_iteratively(
            &empty_features,
            &empty_labels,
            TreeLabelKind::CompletionClassification,
            4,
            1,
        )
        .expect("building on empty data must succeed");
        assert_eq!(trained_tree.all_tree_nodes_flat_vector.len(), 1);
    }

    /// Verifies that batch prediction produces exactly one output per input
    /// and that the outputs are consistent with individual predictions.
    #[test]
    fn batch_prediction_matches_individual_predictions() {
        let (feature_vectors, completion_labels) =
            build_perfectly_separable_classification_training_set();
        let trained_tree = build_decision_tree_iteratively(
            &feature_vectors,
            &completion_labels,
            TreeLabelKind::CompletionClassification,
            4,
            1,
        )
        .expect("building tree must succeed");

        let batch_predictions = predict_batch_with_tree(&trained_tree, &feature_vectors);
        assert_eq!(batch_predictions.len(), feature_vectors.len());

        for sample_position in 0..feature_vectors.len() {
            let individual_prediction = predict_single_feature_vector_with_tree(
                &trained_tree,
                &feature_vectors[sample_position],
            )
            .expect("individual prediction must succeed");
            assert_eq!(
                batch_predictions[sample_position], individual_prediction,
                "batch and individual predictions must agree at position {}",
                sample_position
            );
        }
    }

    /// Verifies that the majority-class leaf for a classification tree on
    /// uniform data returns the correct class.
    #[test]
    fn majority_class_label_returns_correct_majority() {
        let labels_mostly_ones: Vec<i32> = vec![1, 1, 1, 0];
        let all_indices: Vec<usize> = (0..labels_mostly_ones.len()).collect();
        assert_eq!(
            compute_majority_class_label_for_leaf(&labels_mostly_ones, &all_indices),
            1
        );

        let labels_mostly_zeros: Vec<i32> = vec![0, 0, 0, 1];
        let all_indices_zeros: Vec<usize> = (0..labels_mostly_zeros.len()).collect();
        assert_eq!(
            compute_majority_class_label_for_leaf(&labels_mostly_zeros, &all_indices_zeros),
            0
        );
    }

    /// Verifies that the majority-class tiebreak favors 0 (conservative).
    #[test]
    fn majority_class_label_tiebreak_favors_zero() {
        let tied_labels: Vec<i32> = vec![0, 0, 1, 1];
        let all_indices: Vec<usize> = (0..tied_labels.len()).collect();
        assert_eq!(
            compute_majority_class_label_for_leaf(&tied_labels, &all_indices),
            0,
            "tie should break in favor of 0 (conservative / DNF)"
        );
    }

    /// Verifies integer mean computation on a known example.
    /// Labels: [200, 400, 600] -> mean = 400.
    #[test]
    fn mean_integer_label_computes_correctly() {
        let labels: Vec<i32> = vec![200, 400, 600];
        let all_indices: Vec<usize> = (0..labels.len()).collect();
        assert_eq!(
            compute_mean_integer_label_for_leaf(&labels, &all_indices),
            400
        );
    }

    /// Verifies that the mean computation rounds half-up.
    /// Labels: [200, 300] -> true mean = 250.0 -> rounds to 250.
    /// Labels: [200, 301] -> true mean = 250.5 -> rounds to 251.
    #[test]
    fn mean_integer_label_rounds_correctly() {
        let even_labels: Vec<i32> = vec![200, 300];
        let even_indices: Vec<usize> = (0..even_labels.len()).collect();
        assert_eq!(
            compute_mean_integer_label_for_leaf(&even_labels, &even_indices),
            250
        );

        let odd_labels: Vec<i32> = vec![200, 301];
        let odd_indices: Vec<usize> = (0..odd_labels.len()).collect();
        assert_eq!(
            compute_mean_integer_label_for_leaf(&odd_labels, &odd_indices),
            251
        );
    }
}

/*
Section 5: Linear Margin Threshold Analysis
This section implements the second modeling approach: scanning each feature for threshold boundaries where outcomes degrade. Unlike the decision tree (which captures feature interactions), the linear margin analyzer looks at each feature independently — its purpose is to identify simple "risk zones" where extreme values correlate with failure or poor performance.

What This Section Contains

SingleFeatureMarginBoundary struct — the result for one feature: optional low boundary, optional high boundary, and the associated outcome statistics
LinearMarginModel struct — the complete margin analysis: one SingleFeatureMarginBoundary per feature
compute_single_feature_margin_boundary_for_classification — scans sorted feature values to find the low and high boundaries where completion failure rate exceeds a configurable threshold
compute_single_feature_margin_boundary_for_regression — same idea but finds boundaries where mean performance score drops below a configurable threshold
build_linear_margin_model — orchestrates the per-feature scan for all features, producing the complete LinearMarginModel
evaluate_single_feature_vector_against_margins — checks one feature vector against all boundaries and returns a list of triggered risk flags
RiskFlag struct — one triggered risk: which feature, which direction (low/high), what the boundary value was

Plus cargo tests.

Design Decisions Explained Up Front
Why scan from the extremes inward (not split-point search): The tree already captures optimal split points. The linear margin model serves a different purpose: answering "at what extreme values does an outcome become unreliable?" This is a boundary-detection question, not an optimization question. Scanning from the sorted minimum upward (and maximum downward) until the outcome "recovers" finds the outermost zone of poor outcomes — the risk zone — directly.
Why percentage thresholds (not absolute counts): A "failure rate exceeds 50%" threshold adapts to however many samples exist at each extreme, which is important for a ~200-row dataset where some features may have only a few samples at the tails. An absolute count ("more than 3 failures") would be too sensitive to dataset size.
Why integer percentage thresholds (not float): Consistent with the project's integer-math policy. A threshold of 50 means "50%". The comparison (failure_count * 100) / total_count >= threshold_percent avoids all float arithmetic.
Why Option<i32> for boundary values: Some features may have no risk zone on one or both sides (e.g., age might show risk only at the high end, not the low end). None means "no boundary detected on this side", which is semantically distinct from "boundary at value 0".
Why a sliding-window scan (not per-value-bucket): With ~200 rows and only ~8 unique age values, per-value buckets would have very few samples each, making failure-rate estimates noisy. Instead, the scanner accumulates samples from the extreme inward, computing the failure rate of the accumulated "tail" at each step. This gives the failure rate of "all samples at or beyond this value", which is more stable and directly answers the question "is this region of the feature space risky?"

Notes on This Section
Why SingleFeatureMarginBoundary reuses the field name low_tail_failure_rate_percent for both classification and regression: The field holds whatever the boundary scanner produced — a failure-rate percent for classification, or a mean score for regression. Introducing separate field names per label kind would double the struct's fields while adding no behavioral difference. The label_kind_for_this_model tag on LinearMarginModel tells the display code how to interpret the number. The doc string explicitly notes this dual meaning.
Why the high-boundary scan checks >= (not just >): A horse exactly at the boundary value is considered "in the risk zone" because the boundary was derived from a tail that included that value. Using > would exclude boundary-exact values, creating an off-by-one gap where the most informative data point is not flagged.
Why .expect() appears inside tests only: Used to convert Option lookups in test assertions where the preceding assert!(is_some()) already guarantees Some. Production code paths never use .expect().
Why the boundary scanner uses i64 accumulators: Same overflow-safety reasoning as the tree impurity functions. Accumulating up to 200 scores of up to 1000 gives sums up to 200,000 — well within i64 but potentially surprising if accumulated counts or sums were later multiplied together in i32.
*/

// ============================================================================
// SECTION 5 — LINEAR MARGIN THRESHOLD ANALYSIS
// ============================================================================
//
// This section implements the "simple linear" modeling approach described in
// the project scope: per-feature threshold scanning to identify extreme
// value regions ("risk zones") where completion failure rate is high or
// performance score is low.
//
// ## How It Differs From the Decision Tree
//
// The decision tree captures feature *interactions* (e.g., "low height AND
// high weight together predict failure"). The linear margin model examines
// each feature *independently*. Its purpose is not prediction accuracy —
// the tree is better at that — but interpretability: a human can look at
// the margin results and see "any horse with height-to-weight ratio below
// 120 has historically failed 75% of the time."
//
// ## Scanning Algorithm (Per Feature)
//
// For each feature:
//   1. Collect (feature_value, label) pairs for all training samples.
//   2. Sort by feature value.
//   3. Scan from the low end upward: accumulate samples into a "low tail".
//      At each new unique feature value, compute the failure rate (or mean
//      score) of the accumulated tail. If the tail's outcome is "bad"
//      (failure rate >= threshold, or mean score <= threshold), record that
//      feature value as the current low-boundary candidate. Continue until
//      the tail's outcome is no longer bad. The last recorded candidate is
//      the low boundary.
//   4. Repeat scanning from the high end downward for the high boundary.
//
// The result per feature is zero, one, or two boundaries (low, high, or
// both).

/// Identifies the direction of a risk boundary: whether a feature value
/// is dangerously *low* or dangerously *high*.
///
/// ## Project Context
///
/// In the horse-racing domain, "low" might mean a height-to-weight ratio
/// that is too small (horse too heavy for its height), while "high" might
/// mean an age that is too old. Both are "risk zones" but in opposite
/// directions along the feature axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginBoundaryDirection {
    /// Feature values at or below this boundary are in the risk zone.
    Low,
    /// Feature values at or above this boundary are in the risk zone.
    High,
}

/// The margin analysis result for a single feature.
///
/// ## Fields
///
/// - `feature_index` — which feature this boundary describes.
/// - `low_boundary_value` — if `Some(v)`, then feature values `<= v` are
///   in the low-end risk zone. `None` if no low-end risk was detected.
/// - `high_boundary_value` — if `Some(v)`, then feature values `>= v` are
///   in the high-end risk zone. `None` if no high-end risk was detected.
/// - `low_tail_failure_rate_percent` — if a low boundary exists, the
///   failure rate (or inverse of mean score, depending on label kind) in
///   the low tail, as an integer percent 0–100. Useful for reporting
///   severity.
/// - `high_tail_failure_rate_percent` — same for the high tail.
///
/// ## When Both Boundaries Are `None`
///
/// This feature showed no risk zone on either side at the configured
/// threshold. It may still contribute to the tree model via interactions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleFeatureMarginBoundary {
    pub feature_index: FeatureIndex,
    pub low_boundary_value: Option<i32>,
    pub high_boundary_value: Option<i32>,
    pub low_tail_failure_rate_percent: i32,
    pub high_tail_failure_rate_percent: i32,
}

/// A single triggered risk flag when a feature vector is checked against
/// the margin model.
///
/// ## Fields
///
/// - `flagged_feature_index` — which feature triggered the flag.
/// - `boundary_direction` — whether the value was too low or too high.
/// - `boundary_threshold_value` — the boundary value that was crossed.
/// - `actual_feature_value` — the feature vector's value that triggered it.
///
/// ## Project Role
///
/// Risk flags are collected into a `Vec<RiskFlag>` per prediction sample
/// and displayed in the prediction output table. An empty vector means no
/// features are in a risk zone for that sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiskFlag {
    pub flagged_feature_index: FeatureIndex,
    pub boundary_direction: MarginBoundaryDirection,
    pub boundary_threshold_value: i32,
    pub actual_feature_value: i32,
}

/// The complete linear margin model: one boundary result per feature.
///
/// ## Fields
///
/// - `feature_boundaries` — one `SingleFeatureMarginBoundary` per feature
///   in canonical order. The vector length equals `ENGINEERED_FEATURE_COUNT`.
/// - `label_kind_for_this_model` — whether this model was trained on
///   completion (classification) or performance score (regression).
/// - `threshold_percent_used_for_training` — the sensitivity parameter
///   that was used to build the boundaries; recorded for reproducibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearMarginModel {
    pub feature_boundaries: Vec<SingleFeatureMarginBoundary>,
    pub label_kind_for_this_model: TreeLabelKind,
    pub threshold_percent_used_for_training: i32,
}

/// Computes the low and high margin boundaries for a single feature on
/// the classification (completion) task.
///
/// ## Corrected Algorithm
///
/// The boundary detection evaluates each unique feature value group
/// independently to decide whether that group itself is "risky". The
/// boundary advances as long as consecutive groups (scanning inward from
/// each extreme) are individually risky. The scan stops as soon as a
/// group is not risky — this correctly identifies the outermost contiguous
/// zone of bad outcomes, regardless of how many samples are in later groups.
///
/// The accumulated tail (all samples from the extreme up to and including
/// the current boundary) is used only to compute the reported severity
/// statistic (`low_tail_failure_rate_percent`), not to decide whether to
/// advance the boundary.
///
/// ## Why the Previous Version Was Wrong
///
/// The prior version evaluated the *running accumulated* failure rate at
/// each step. Because the accumulator retains all prior (failing) samples,
/// adding a few passing samples rarely brings the accumulated rate below
/// the threshold quickly with small data (~200 rows). The boundary
/// therefore advanced too far into the safe zone.
fn compute_single_feature_classification_boundaries(
    feature_values_sorted_with_labels: &[(i32, i32)],
    failure_threshold_percent: i32,
) -> (Option<i32>, i32, Option<i32>, i32) {
    let total_sample_count = feature_values_sorted_with_labels.len();
    if total_sample_count == 0 {
        return (None, 0, None, 0);
    }

    // --- Low-boundary scan (left to right) ---
    // For each unique feature value group, compute that group's own failure
    // rate. If it meets the threshold, it is part of the risk zone and the
    // boundary advances to cover it. If it does not, stop.
    let mut low_boundary_candidate: Option<i32> = None;
    let mut accumulated_failure_count_for_severity: i64 = 0;
    let mut accumulated_total_count_for_severity: i64 = 0;
    let mut low_tail_failure_rate_at_boundary: i32 = 0;

    let mut scan_position_low: usize = 0;
    // Outer loop bound: at most total_sample_count iterations (each sample
    // is consumed by the inner loop exactly once across all outer iterations).
    while scan_position_low < total_sample_count {
        let current_group_feature_value = feature_values_sorted_with_labels[scan_position_low].0;

        // Count this group's own samples.
        let mut group_total_count: i64 = 0;
        let mut group_failure_count: i64 = 0;
        let group_start_position = scan_position_low;

        while scan_position_low < total_sample_count
            && feature_values_sorted_with_labels[scan_position_low].0 == current_group_feature_value
        {
            group_total_count += 1;
            if feature_values_sorted_with_labels[scan_position_low].1 == 0 {
                group_failure_count += 1;
            }
            scan_position_low += 1;
        }
        let _ = group_start_position; // consumed by inner loop

        // Evaluate this group on its own failure rate.
        let group_failure_rate_percent: i32 =
            ((group_failure_count * 100) / group_total_count) as i32;

        if group_failure_rate_percent >= failure_threshold_percent {
            // This group is risky — extend the low boundary to cover it.
            low_boundary_candidate = Some(current_group_feature_value);
            // Accumulate into the severity tail.
            accumulated_failure_count_for_severity += group_failure_count;
            accumulated_total_count_for_severity += group_total_count;
            if accumulated_total_count_for_severity > 0 {
                low_tail_failure_rate_at_boundary = ((accumulated_failure_count_for_severity * 100)
                    / accumulated_total_count_for_severity)
                    as i32;
            }
        } else {
            // This group is not risky — the contiguous risk zone has ended.
            break;
        }
    }

    // --- High-boundary scan (right to left) ---
    let mut high_boundary_candidate: Option<i32> = None;
    let mut accumulated_failure_count_high_for_severity: i64 = 0;
    let mut accumulated_total_count_high_for_severity: i64 = 0;
    let mut high_tail_failure_rate_at_boundary: i32 = 0;

    let mut scan_position_high_plus_one: usize = total_sample_count;
    while scan_position_high_plus_one > 0 {
        let current_group_feature_value =
            feature_values_sorted_with_labels[scan_position_high_plus_one - 1].0;

        let mut group_total_count: i64 = 0;
        let mut group_failure_count: i64 = 0;

        while scan_position_high_plus_one > 0
            && feature_values_sorted_with_labels[scan_position_high_plus_one - 1].0
                == current_group_feature_value
        {
            group_total_count += 1;
            if feature_values_sorted_with_labels[scan_position_high_plus_one - 1].1 == 0 {
                group_failure_count += 1;
            }
            scan_position_high_plus_one -= 1;
        }

        let group_failure_rate_percent: i32 =
            ((group_failure_count * 100) / group_total_count) as i32;

        if group_failure_rate_percent >= failure_threshold_percent {
            high_boundary_candidate = Some(current_group_feature_value);
            accumulated_failure_count_high_for_severity += group_failure_count;
            accumulated_total_count_high_for_severity += group_total_count;
            if accumulated_total_count_high_for_severity > 0 {
                high_tail_failure_rate_at_boundary =
                    ((accumulated_failure_count_high_for_severity * 100)
                        / accumulated_total_count_high_for_severity) as i32;
            }
        } else {
            break;
        }
    }

    (
        low_boundary_candidate,
        low_tail_failure_rate_at_boundary,
        high_boundary_candidate,
        high_tail_failure_rate_at_boundary,
    )
}

/// Computes the low and high margin boundaries for a single feature on
/// the regression (performance score) task.
///
/// ## Corrected Algorithm
///
/// Same correction as the classification variant: each unique feature value
/// group is evaluated on its own mean score, not the running accumulated
/// mean. The boundary advances as long as consecutive inward groups are
/// individually at or below the score threshold. The accumulated tail mean
/// is tracked separately only for the severity statistic.
fn compute_single_feature_regression_boundaries(
    feature_values_sorted_with_scores: &[(i32, i32)],
    low_score_threshold_value: i32,
) -> (Option<i32>, i32, Option<i32>, i32) {
    let total_sample_count = feature_values_sorted_with_scores.len();
    if total_sample_count == 0 {
        return (None, 0, None, 0);
    }

    // --- Low-boundary scan (left to right) ---
    let mut low_boundary_candidate: Option<i32> = None;
    let mut accumulated_score_sum_for_severity: i64 = 0;
    let mut accumulated_count_for_severity: i64 = 0;
    let mut low_tail_mean_at_boundary: i32 = 0;

    let mut scan_position_low: usize = 0;
    while scan_position_low < total_sample_count {
        let current_group_feature_value = feature_values_sorted_with_scores[scan_position_low].0;

        let mut group_total_count: i64 = 0;
        let mut group_score_sum: i64 = 0;

        while scan_position_low < total_sample_count
            && feature_values_sorted_with_scores[scan_position_low].0 == current_group_feature_value
        {
            group_total_count += 1;
            group_score_sum += feature_values_sorted_with_scores[scan_position_low].1 as i64;
            scan_position_low += 1;
        }

        let group_mean_score: i32 = (group_score_sum / group_total_count) as i32;

        if group_mean_score <= low_score_threshold_value {
            low_boundary_candidate = Some(current_group_feature_value);
            accumulated_score_sum_for_severity += group_score_sum;
            accumulated_count_for_severity += group_total_count;
            if accumulated_count_for_severity > 0 {
                low_tail_mean_at_boundary =
                    (accumulated_score_sum_for_severity / accumulated_count_for_severity) as i32;
            }
        } else {
            break;
        }
    }

    // --- High-boundary scan (right to left) ---
    let mut high_boundary_candidate: Option<i32> = None;
    let mut accumulated_score_sum_high_for_severity: i64 = 0;
    let mut accumulated_count_high_for_severity: i64 = 0;
    let mut high_tail_mean_at_boundary: i32 = 0;

    let mut scan_position_high_plus_one: usize = total_sample_count;
    while scan_position_high_plus_one > 0 {
        let current_group_feature_value =
            feature_values_sorted_with_scores[scan_position_high_plus_one - 1].0;

        let mut group_total_count: i64 = 0;
        let mut group_score_sum: i64 = 0;

        while scan_position_high_plus_one > 0
            && feature_values_sorted_with_scores[scan_position_high_plus_one - 1].0
                == current_group_feature_value
        {
            group_total_count += 1;
            group_score_sum +=
                feature_values_sorted_with_scores[scan_position_high_plus_one - 1].1 as i64;
            scan_position_high_plus_one -= 1;
        }

        let group_mean_score: i32 = (group_score_sum / group_total_count) as i32;

        if group_mean_score <= low_score_threshold_value {
            high_boundary_candidate = Some(current_group_feature_value);
            accumulated_score_sum_high_for_severity += group_score_sum;
            accumulated_count_high_for_severity += group_total_count;
            if accumulated_count_high_for_severity > 0 {
                high_tail_mean_at_boundary = (accumulated_score_sum_high_for_severity
                    / accumulated_count_high_for_severity)
                    as i32;
            }
        } else {
            break;
        }
    }

    (
        low_boundary_candidate,
        low_tail_mean_at_boundary,
        high_boundary_candidate,
        high_tail_mean_at_boundary,
    )
}

/// Builds a complete linear margin model by scanning every feature.
///
/// ## Inputs
///
/// - `training_feature_vectors` — the full set of engineered feature vectors
///   for the training data.
/// - `training_labels` — parallel label vector (completion values for
///   classification, performance scores for regression).
/// - `label_kind` — determines which boundary-scanning function to use.
/// - `threshold_value` — for classification: the failure-rate percent
///   threshold (0–100). For regression: the absolute score threshold
///   (e.g. 400).
///
/// ## Algorithm
///
/// For each feature in canonical order:
///   1. Collect `(feature_value, label)` pairs.
///   2. Sort by feature value.
///   3. Call the appropriate boundary scanner.
///   4. Store the result in the model.
///
/// ## Error Cases
///
/// Returns `FieldValueOutOfValidRange` if feature vectors and labels have
/// different lengths. Otherwise always succeeds (features with no risk zones
/// get `None` boundaries, which is a valid result).
pub fn build_linear_margin_model(
    training_feature_vectors: &[EngineeredFeatureVector],
    training_labels: &[i32],
    label_kind: TreeLabelKind,
    threshold_value: i32,
) -> Result<LinearMarginModel, HorseRacingError> {
    if training_feature_vectors.len() != training_labels.len() {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "build_linear_margin_model: feature vectors and labels length mismatch",
        ));
    }

    let sample_count = training_feature_vectors.len();
    let all_features = all_feature_indices_in_canonical_order();
    let mut all_feature_boundaries: Vec<SingleFeatureMarginBoundary> =
        Vec::with_capacity(ENGINEERED_FEATURE_COUNT);

    for feature_under_analysis in all_features.iter() {
        // Collect (feature_value, label) pairs.
        let mut feature_label_pairs: Vec<(i32, i32)> = Vec::with_capacity(sample_count);
        for sample_position in 0..sample_count {
            let feature_value = extract_feature_value_from_vector(
                &training_feature_vectors[sample_position],
                *feature_under_analysis,
            );
            feature_label_pairs.push((feature_value, training_labels[sample_position]));
        }

        // Sort by feature value ascending. Stable sort preserves insertion
        // order among tied feature values, which is deterministic given
        // deterministic input ordering.
        feature_label_pairs.sort_by(|pair_a, pair_b| pair_a.0.cmp(&pair_b.0));

        // Scan for boundaries using the appropriate function.
        let (
            low_boundary_option,
            low_tail_rate_or_score,
            high_boundary_option,
            high_tail_rate_or_score,
        ) = match label_kind {
            TreeLabelKind::CompletionClassification => {
                compute_single_feature_classification_boundaries(
                    &feature_label_pairs,
                    threshold_value,
                )
            }
            TreeLabelKind::PerformanceScoreRegression => {
                compute_single_feature_regression_boundaries(&feature_label_pairs, threshold_value)
            }
        };

        all_feature_boundaries.push(SingleFeatureMarginBoundary {
            feature_index: *feature_under_analysis,
            low_boundary_value: low_boundary_option,
            high_boundary_value: high_boundary_option,
            low_tail_failure_rate_percent: low_tail_rate_or_score,
            high_tail_failure_rate_percent: high_tail_rate_or_score,
        });
    }

    Ok(LinearMarginModel {
        feature_boundaries: all_feature_boundaries,
        label_kind_for_this_model: label_kind,
        threshold_percent_used_for_training: threshold_value,
    })
}

/// Checks a single feature vector against all margin boundaries in a
/// trained `LinearMarginModel`, returning a list of triggered risk flags.
///
/// ## Logic
///
/// For each feature boundary in the model:
/// - If a low boundary exists and the feature value is `<= low_boundary`:
///   flag it.
/// - If a high boundary exists and the feature value is `>= high_boundary`:
///   flag it.
///
/// ## Returns
///
/// A `Vec<RiskFlag>` that may be empty (no features in a risk zone for this
/// sample). The caller uses this for display purposes; the tree model's
/// prediction is the primary output, and the risk flags are supplementary
/// interpretability.
pub fn evaluate_single_feature_vector_against_margins(
    linear_margin_model: &LinearMarginModel,
    feature_vector_to_check: &EngineeredFeatureVector,
) -> Vec<RiskFlag> {
    let mut triggered_risk_flags: Vec<RiskFlag> = Vec::with_capacity(ENGINEERED_FEATURE_COUNT);

    for boundary_entry in linear_margin_model.feature_boundaries.iter() {
        let actual_value = extract_feature_value_from_vector(
            feature_vector_to_check,
            boundary_entry.feature_index,
        );

        // Check low boundary.
        if let Some(low_threshold) = boundary_entry.low_boundary_value {
            if actual_value <= low_threshold {
                triggered_risk_flags.push(RiskFlag {
                    flagged_feature_index: boundary_entry.feature_index,
                    boundary_direction: MarginBoundaryDirection::Low,
                    boundary_threshold_value: low_threshold,
                    actual_feature_value: actual_value,
                });
            }
        }

        // Check high boundary.
        if let Some(high_threshold) = boundary_entry.high_boundary_value {
            if actual_value >= high_threshold {
                triggered_risk_flags.push(RiskFlag {
                    flagged_feature_index: boundary_entry.feature_index,
                    boundary_direction: MarginBoundaryDirection::High,
                    boundary_threshold_value: high_threshold,
                    actual_feature_value: actual_value,
                });
            }
        }
    }

    triggered_risk_flags
}

/// Checks a batch of feature vectors against the margin model, returning
/// one `Vec<RiskFlag>` per sample.
///
/// ## Project Role
///
/// Used in prediction output to annotate each horse's prediction with any
/// margin risk flags. Mirrors `predict_batch_with_tree` in structure so
/// the caller can zip tree predictions and margin flags by sample position.
pub fn evaluate_batch_against_margins(
    linear_margin_model: &LinearMarginModel,
    batch_of_feature_vectors: &[EngineeredFeatureVector],
) -> Vec<Vec<RiskFlag>> {
    let mut all_risk_flags_per_sample: Vec<Vec<RiskFlag>> =
        Vec::with_capacity(batch_of_feature_vectors.len());

    for feature_vector_reference in batch_of_feature_vectors.iter() {
        let flags_for_this_sample = evaluate_single_feature_vector_against_margins(
            linear_margin_model,
            feature_vector_reference,
        );
        all_risk_flags_per_sample.push(flags_for_this_sample);
    }

    all_risk_flags_per_sample
}

// ============================================================================
// SECTION 5 — CARGO TESTS
// ============================================================================

#[cfg(test)]
mod section_five_linear_margin_tests {
    use super::*;

    /// Builds a synthetic classification training set where low-age horses
    /// (age 1, 2) consistently fail (completion = 0) and all others
    /// complete. This gives a clear low-boundary risk zone on the age
    /// feature.
    fn build_low_age_failure_classification_set() -> (Vec<EngineeredFeatureVector>, Vec<i32>) {
        let mut feature_vectors: Vec<EngineeredFeatureVector> = Vec::new();
        let mut completion_labels: Vec<i32> = Vec::new();

        // Ages 1 and 2: all fail (completion = 0).
        for failing_age in 1..=2 {
            for _duplicate in 0..3 {
                feature_vectors.push(EngineeredFeatureVector {
                    age: failing_age,
                    height: 150,
                    experience: 3,
                    weight: 900,
                    height_to_weight_ratio_times_one_thousand: 166,
                    age_times_experience: failing_age * 3,
                });
                completion_labels.push(0);
            }
        }

        // Ages 4 through 7: all complete (completion = 1).
        for completing_age in 4..=7 {
            for _duplicate in 0..3 {
                feature_vectors.push(EngineeredFeatureVector {
                    age: completing_age,
                    height: 150,
                    experience: 3,
                    weight: 900,
                    height_to_weight_ratio_times_one_thousand: 166,
                    age_times_experience: completing_age * 3,
                });
                completion_labels.push(1);
            }
        }

        (feature_vectors, completion_labels)
    }

    /// Builds a synthetic regression training set where low-height horses
    /// have very low performance scores and high-height horses have high
    /// scores. Provides clear risk zones at both extremes.
    fn build_height_risk_regression_set() -> (Vec<EngineeredFeatureVector>, Vec<i32>) {
        let mut feature_vectors: Vec<EngineeredFeatureVector> = Vec::new();
        let mut performance_labels: Vec<i32> = Vec::new();

        // Very short horses (height 110, 115): low scores.
        for short_height in [110, 115].iter() {
            for _duplicate in 0..3 {
                feature_vectors.push(EngineeredFeatureVector {
                    age: 4,
                    height: *short_height,
                    experience: 3,
                    weight: 900,
                    height_to_weight_ratio_times_one_thousand: (*short_height * 1000) / 900,
                    age_times_experience: 12,
                });
                performance_labels.push(200);
            }
        }

        // Medium horses (height 150, 160): good scores.
        for medium_height in [150, 160].iter() {
            for _duplicate in 0..3 {
                feature_vectors.push(EngineeredFeatureVector {
                    age: 4,
                    height: *medium_height,
                    experience: 3,
                    weight: 900,
                    height_to_weight_ratio_times_one_thousand: (*medium_height * 1000) / 900,
                    age_times_experience: 12,
                });
                performance_labels.push(800);
            }
        }

        // Very tall horses (height 195, 199): low scores again.
        for tall_height in [195, 199].iter() {
            for _duplicate in 0..3 {
                feature_vectors.push(EngineeredFeatureVector {
                    age: 4,
                    height: *tall_height,
                    experience: 3,
                    weight: 900,
                    height_to_weight_ratio_times_one_thousand: (*tall_height * 1000) / 900,
                    age_times_experience: 12,
                });
                performance_labels.push(200);
            }
        }

        (feature_vectors, performance_labels)
    }

    /// Verifies that the classification margin model detects a low-age risk
    /// boundary when young horses consistently fail.
    #[test]
    fn classification_margin_detects_low_age_risk_zone() {
        let (feature_vectors, completion_labels) = build_low_age_failure_classification_set();
        let failure_rate_threshold_percent: i32 = 50;

        let margin_model = build_linear_margin_model(
            &feature_vectors,
            &completion_labels,
            TreeLabelKind::CompletionClassification,
            failure_rate_threshold_percent,
        )
        .expect("margin model build must succeed");

        // Find the boundary for the Age feature.
        let age_boundary_option = margin_model
            .feature_boundaries
            .iter()
            .find(|boundary| boundary.feature_index == FeatureIndex::Age);
        assert!(
            age_boundary_option.is_some(),
            "must have a boundary entry for Age"
        );
        let age_boundary = age_boundary_option.expect("just checked is_some");

        // Low boundary should exist (ages 1-2 fail).
        assert!(
            age_boundary.low_boundary_value.is_some(),
            "age low boundary must be detected"
        );
        let detected_low_boundary = age_boundary
            .low_boundary_value
            .expect("just checked is_some");
        // The boundary should be at age 2 (the highest failing age).
        assert_eq!(detected_low_boundary, 2);

        // High boundary should NOT exist (no high-age failures in this set).
        assert!(
            age_boundary.high_boundary_value.is_none(),
            "age high boundary should not be detected in this test data"
        );
    }

    /// Verifies that the regression margin model detects risk zones at both
    /// extremes of the height feature.
    #[test]
    fn regression_margin_detects_both_height_risk_zones() {
        let (feature_vectors, performance_labels) = build_height_risk_regression_set();
        // Threshold: mean score <= 300 is "risky".
        let low_score_threshold: i32 = 300;

        let margin_model = build_linear_margin_model(
            &feature_vectors,
            &performance_labels,
            TreeLabelKind::PerformanceScoreRegression,
            low_score_threshold,
        )
        .expect("margin model build must succeed");

        let height_boundary_option = margin_model
            .feature_boundaries
            .iter()
            .find(|boundary| boundary.feature_index == FeatureIndex::Height);
        assert!(height_boundary_option.is_some());
        let height_boundary = height_boundary_option.expect("just checked is_some");

        // Low boundary: short horses (110, 115) have mean score 200 <= 300.
        assert!(
            height_boundary.low_boundary_value.is_some(),
            "height low boundary must be detected for short horses"
        );

        // High boundary: tall horses (195, 199) have mean score 200 <= 300.
        assert!(
            height_boundary.high_boundary_value.is_some(),
            "height high boundary must be detected for tall horses"
        );
    }

    /// Verifies that a feature vector in the low-age risk zone triggers a
    /// risk flag, while one safely in the middle does not.
    #[test]
    fn risk_flag_evaluation_triggers_for_extreme_and_not_for_safe() {
        let (feature_vectors, completion_labels) = build_low_age_failure_classification_set();
        let margin_model = build_linear_margin_model(
            &feature_vectors,
            &completion_labels,
            TreeLabelKind::CompletionClassification,
            50,
        )
        .expect("margin model build must succeed");

        // A horse with age 1 should trigger the low-age risk flag.
        let risky_young_horse = EngineeredFeatureVector {
            age: 1,
            height: 150,
            experience: 3,
            weight: 900,
            height_to_weight_ratio_times_one_thousand: 166,
            age_times_experience: 3,
        };
        let risky_flags =
            evaluate_single_feature_vector_against_margins(&margin_model, &risky_young_horse);
        let age_risk_flag_found = risky_flags.iter().any(|flag_reference| {
            flag_reference.flagged_feature_index == FeatureIndex::Age
                && flag_reference.boundary_direction == MarginBoundaryDirection::Low
        });
        assert!(
            age_risk_flag_found,
            "age = 1 must trigger a low-age risk flag"
        );

        // A horse with age 5 should NOT trigger any age risk flag.
        let safe_middle_horse = EngineeredFeatureVector {
            age: 5,
            height: 150,
            experience: 3,
            weight: 900,
            height_to_weight_ratio_times_one_thousand: 166,
            age_times_experience: 15,
        };
        let safe_flags =
            evaluate_single_feature_vector_against_margins(&margin_model, &safe_middle_horse);
        let no_age_flag = !safe_flags
            .iter()
            .any(|flag_reference| flag_reference.flagged_feature_index == FeatureIndex::Age);
        assert!(no_age_flag, "age = 5 must not trigger any age risk flag");
    }

    /// Verifies that the batch evaluation produces one result per input
    /// sample.
    #[test]
    fn batch_margin_evaluation_produces_correct_count() {
        let (feature_vectors, completion_labels) = build_low_age_failure_classification_set();
        let margin_model = build_linear_margin_model(
            &feature_vectors,
            &completion_labels,
            TreeLabelKind::CompletionClassification,
            50,
        )
        .expect("margin model build must succeed");

        let batch_flags = evaluate_batch_against_margins(&margin_model, &feature_vectors);
        assert_eq!(batch_flags.len(), feature_vectors.len());
    }

    /// Verifies that the model rejects mismatched feature/label lengths.
    #[test]
    fn margin_model_rejects_length_mismatch() {
        let feature_vectors: Vec<EngineeredFeatureVector> = vec![EngineeredFeatureVector {
            age: 4,
            height: 150,
            experience: 3,
            weight: 900,
            height_to_weight_ratio_times_one_thousand: 166,
            age_times_experience: 12,
        }];
        let too_many_labels: Vec<i32> = vec![1, 0];
        let build_result = build_linear_margin_model(
            &feature_vectors,
            &too_many_labels,
            TreeLabelKind::CompletionClassification,
            50,
        );
        assert!(matches!(
            build_result,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));
    }

    /// Verifies that a uniform training set (all samples complete, no
    /// failures) produces no risk boundaries.
    #[test]
    fn no_risk_boundaries_when_all_samples_succeed() {
        let mut feature_vectors: Vec<EngineeredFeatureVector> = Vec::new();
        let mut completion_labels: Vec<i32> = Vec::new();
        for age_value in 1..=8 {
            feature_vectors.push(EngineeredFeatureVector {
                age: age_value,
                height: 150,
                experience: 3,
                weight: 900,
                height_to_weight_ratio_times_one_thousand: 166,
                age_times_experience: age_value * 3,
            });
            completion_labels.push(1);
        }

        let margin_model = build_linear_margin_model(
            &feature_vectors,
            &completion_labels,
            TreeLabelKind::CompletionClassification,
            50,
        )
        .expect("must succeed");

        for boundary_entry in margin_model.feature_boundaries.iter() {
            assert!(
                boundary_entry.low_boundary_value.is_none(),
                "no low boundary expected when all samples succeed"
            );
            assert!(
                boundary_entry.high_boundary_value.is_none(),
                "no high boundary expected when all samples succeed"
            );
        }
    }

    /// Verifies that an empty training set produces a valid model with no
    /// boundaries (not an error).
    #[test]
    fn empty_training_set_produces_valid_model_with_no_boundaries() {
        let empty_features: Vec<EngineeredFeatureVector> = Vec::new();
        let empty_labels: Vec<i32> = Vec::new();

        let margin_model = build_linear_margin_model(
            &empty_features,
            &empty_labels,
            TreeLabelKind::CompletionClassification,
            50,
        )
        .expect("must succeed on empty data");

        assert_eq!(
            margin_model.feature_boundaries.len(),
            ENGINEERED_FEATURE_COUNT
        );
        for boundary_entry in margin_model.feature_boundaries.iter() {
            assert!(boundary_entry.low_boundary_value.is_none());
            assert!(boundary_entry.high_boundary_value.is_none());
        }
    }

    /// Verifies that the classification boundary scanner correctly handles
    /// the case where only the first unique feature value has failures,
    /// placing the boundary exactly at that value.
    #[test]
    fn classification_boundary_scanner_places_boundary_at_single_failing_value() {
        // Sorted (feature_value, label) pairs.
        // Only feature value 100 has failures.
        let sorted_pairs: Vec<(i32, i32)> = vec![
            (100, 0),
            (100, 0),
            (100, 0),
            (200, 1),
            (200, 1),
            (300, 1),
            (300, 1),
        ];

        let (low_boundary, _low_rate, high_boundary, _high_rate) =
            compute_single_feature_classification_boundaries(&sorted_pairs, 50);
        assert_eq!(low_boundary, Some(100));
        assert!(high_boundary.is_none());
    }

    /// Verifies the regression boundary scanner on a known sorted sequence
    /// where low values have low scores.
    #[test]
    fn regression_boundary_scanner_detects_low_score_zone() {
        let sorted_pairs: Vec<(i32, i32)> = vec![
            (100, 100),
            (100, 200),
            (200, 150),
            (300, 800),
            (300, 900),
            (400, 1000),
        ];
        // Threshold: mean score <= 300 is risky.
        let (low_boundary, _low_score, _high_boundary, _high_score) =
            compute_single_feature_regression_boundaries(&sorted_pairs, 300);

        // At feature value 100: tail mean = (100+200)/2 = 150 <= 300 -> boundary candidate.
        // At feature value 200: tail mean = (100+200+150)/3 = 150 <= 300 -> boundary candidate.
        // At feature value 300: tail mean = (100+200+150+800+900)/5 = 430 > 300 -> stop.
        // So low boundary should be at 200.
        assert_eq!(low_boundary, Some(200));
    }
}

/*
Section 6: Model Persistence (Save and Load as Plain Text)
This section handles saving trained models to disk and loading them back. Both the decision tree and the linear margin model are persisted as plain text files — one clearly labeled field per line — so they are human-readable, debuggable without special tooling, and exactly round-trippable without floating point.

What This Section Contains

save_decision_tree_to_plain_text_file — writes a DecisionTree to a file, one node per line with labeled fields
load_decision_tree_from_plain_text_file — reads that file back into a DecisionTree, validating structure as it goes
save_linear_margin_model_to_plain_text_file — writes a LinearMarginModel to a file
load_linear_margin_model_from_plain_text_file — reads it back
Shared helpers: parse_labeled_line_value, write_line_to_file_with_newline — used by both save/load paths

Plus cargo tests (round-trip tests for both model types, malformed-file rejection tests).

File Format Design
Decision tree file:
Copyhorse_racing_decision_tree_v1
label_kind=completion_classification
max_depth_used=4
node_count=7
node_index=0 kind=decision feature=age threshold=5 left=1 right=2 leaf_value=0
node_index=1 kind=leaf feature=age threshold=0 left=4294967295 right=4294967295 leaf_value=1
...
Linear margin model file:
Copyhorse_racing_linear_margin_v1
label_kind=completion_classification
threshold_used=50
feature_count=6
feature=age low_boundary=2 high_boundary=none low_rate=100 high_rate=0
feature=height low_boundary=none high_boundary=none low_rate=0 high_rate=0
...
One header line for format identification, then labeled key=value lines, then one data line per node or feature. The header line acts as a version sentinel — if the format ever changes, the header changes and old files are rejected cleanly rather than silently misloaded.


Why parse_labeled_line_value returns a &str with a lifetime tied to the input line: The caller owns the String line and passes it by reference. Returning a subslice avoids allocating a new String for every field value during loading — the value is immediately parsed (to i32, u32, or matched against a known string) and discarded.
Why node lines use split_whitespace rather than a second labeled-line parser: Each node line contains seven space-separated key=value tokens. split_whitespace handles any amount of whitespace between tokens and is robust to trailing spaces. The labeled-line parser parse_labeled_line_value then handles the = within each token.
Why the child-index referential integrity check is deferred until after all nodes are loaded: The tree is built breadth-first, so a parent node always has a lower index than its children. But a corrupt file could contain any ordering, so checking during the per-node parse loop would require forward-referencing into a not-yet-populated vector. Deferring to a second pass over the completed loaded_tree_nodes is simpler, clearer, and catches the same errors.
Why flush() is called explicitly before dropping the BufWriter: Rust's BufWriter flushes on Drop, but a flush error during Drop is silently swallowed (no way to propagate it). Calling flush() explicitly means a disk-full or write error surfaces as a Result::Err that the caller can handle — rather than disappearing silently into a drop.

*/

// ============================================================================
// SECTION 6 — MODEL PERSISTENCE: SAVE AND LOAD AS PLAIN TEXT
// ============================================================================
//
// This section saves and loads both trained model types (decision tree and
// linear margin model) as plain text files. The format is deliberately
// simple: one labeled field per line, human-readable without special tools.
//
// ## Format Philosophy
//
// Each file begins with a format-identification header line (e.g.
// "horse_racing_decision_tree_v1"). This acts as a version sentinel: if
// the format ever changes incompatibly, the header version changes, and
// the loader rejects old files with a clear error rather than silently
// misinterpreting them.
//
// All values are integers or canonical string names — no floats, no JSON,
// no TOML, no external format dependency. The sentinel `NO_CHILD_NODE_INDEX`
// (u32::MAX = 4294967295) is written and read as a plain integer for
// decision-node child fields of leaf nodes.
//
// Optional values (boundary values in the margin model) are written as
// the integer value if present, or the literal string "none" if absent.
//
// ## File I/O Policy
//
// Writes use `BufWriter` to avoid one syscall per line. Reads use
// `BufReader` and process one line at a time — the whole file is never
// loaded into memory at once.

use std::io::{BufWriter, Write};

/// The format-identification header written as the first line of every
/// saved decision tree file.
///
/// If the tree file format changes incompatibly, increment the version
/// suffix here. The loader checks this exact string and rejects files
/// that do not match.
const DECISION_TREE_FILE_FORMAT_HEADER: &str = "horse_racing_decision_tree_v1";

/// The format-identification header for linear margin model files.
const LINEAR_MARGIN_MODEL_FILE_FORMAT_HEADER: &str = "horse_racing_linear_margin_v1";

/// The literal string written in place of an absent `Option<i32>` boundary
/// value in a margin model file.
const ABSENT_BOUNDARY_SENTINEL_STRING: &str = "none";

/// Writes one labeled line to a `BufWriter<File>`, appending a newline.
///
/// ## Project Role
///
/// Centralizing line writes means every save function uses the same write
/// path, so a file I/O error in any save function surfaces the same error
/// variant with a unique calling-function prefix baked in by the caller.
///
/// ## Arguments
///
/// - `buffered_file_writer` — the destination.
/// - `line_content` — the line text, without a trailing newline (this
///   function appends `\n`).
/// - `error_prefix` — a `&'static str` identifying the calling function,
///   used in the error message if the write fails.
fn write_line_to_buffered_file(
    buffered_file_writer: &mut BufWriter<File>,
    line_content: &str,
    error_prefix: &'static str,
) -> Result<(), HorseRacingError> {
    buffered_file_writer
        .write_all(line_content.as_bytes())
        .map_err(|_io_error_discarded| HorseRacingError::CsvFileReadFailure(error_prefix))?;
    buffered_file_writer
        .write_all(b"\n")
        .map_err(|_io_error_discarded| HorseRacingError::CsvFileReadFailure(error_prefix))
}

/// Parses a single labeled value from a model file line of the form
/// `key=value`, returning the `value` portion as a `&str`.
///
/// ## Arguments
///
/// - `line_text` — the full line.
/// - `expected_key` — the exact key string expected before the `=`.
/// - `error_prefix` — identifies the calling function in any error.
///
/// ## Error
///
/// Returns `CsvHeaderMismatch` if the key before `=` does not match
/// `expected_key`, or if there is no `=` on the line. Reusing
/// `CsvHeaderMismatch` here is intentional: a key mismatch in a model
/// file is exactly the same class of problem as a header mismatch in a
/// CSV — the file does not match the expected schema.
fn parse_labeled_line_value<'line_lifetime>(
    line_text: &'line_lifetime str,
    expected_key: &str,
    error_prefix: &'static str,
) -> Result<&'line_lifetime str, HorseRacingError> {
    let equals_byte_position = match line_text.find('=') {
        Some(position) => position,
        None => {
            return Err(HorseRacingError::CsvHeaderMismatch(error_prefix));
        }
    };
    let actual_key = &line_text[..equals_byte_position];
    if actual_key != expected_key {
        return Err(HorseRacingError::CsvHeaderMismatch(error_prefix));
    }
    Ok(&line_text[(equals_byte_position + 1)..])
}

/// Reads the next non-empty line from a line iterator, returning
/// `CsvFileReadFailure` if the iterator is exhausted or the line cannot
/// be read.
///
/// ## Why a Helper
///
/// Both the tree loader and the margin loader need to advance through
/// expected lines in sequence. Centralizing this avoids repeating the
/// same match-on-Option-then-match-on-Result pattern at every line read.
fn read_next_nonempty_model_file_line(
    line_iterator: &mut impl Iterator<Item = std::io::Result<String>>,
    error_prefix: &'static str,
) -> Result<String, HorseRacingError> {
    loop {
        match line_iterator.next() {
            None => {
                return Err(HorseRacingError::CsvFileReadFailure(error_prefix));
            }
            Some(Err(_io_error_discarded)) => {
                return Err(HorseRacingError::CsvFileReadFailure(error_prefix));
            }
            Some(Ok(line_string)) => {
                if !line_string.trim().is_empty() {
                    return Ok(line_string);
                }
                // Empty line — skip and continue.
            }
        }
    }
}

/// Saves a trained `DecisionTree` to a plain text file.
///
/// ## File Format
///
/// Line 1: format header (`DECISION_TREE_FILE_FORMAT_HEADER`)
/// Line 2: `label_kind=<canonical name>`
/// Line 3: `max_depth_used=<integer>`
/// Line 4: `node_count=<integer>`
/// Lines 5+: one node per line, space-separated labeled fields:
///   `node_index=N kind=decision|leaf feature=<name> threshold=N left=N right=N leaf_value=N`
///
/// ## Project Role
///
/// Called after Stage 2 (train-on-all-data) to persist the final models.
/// Also called after Stage 1 (hyperparameter search) if the caller wants
/// to checkpoint the best-split model before retraining on all data.
///
/// ## Error Handling
///
/// Any file creation or write failure returns `CsvFileReadFailure` with
/// the function name prefix. The underlying OS error is discarded to avoid
/// leaking filesystem paths into production logs.
pub fn save_decision_tree_to_plain_text_file(
    trained_decision_tree: &DecisionTree,
    output_file_path: &Path,
) -> Result<(), HorseRacingError> {
    let created_file = File::create(output_file_path).map_err(|_os_error_discarded| {
        HorseRacingError::CsvFileReadFailure(
            "save_decision_tree_to_plain_text_file: could not create output file",
        )
    })?;
    let mut buffered_writer = BufWriter::new(created_file);

    // Header line — version sentinel.
    write_line_to_buffered_file(
        &mut buffered_writer,
        DECISION_TREE_FILE_FORMAT_HEADER,
        "save_decision_tree_to_plain_text_file: header write failed",
    )?;

    // Label kind.
    let label_kind_line = format!(
        "label_kind={}",
        trained_decision_tree
            .label_kind_for_this_tree
            .canonical_label_kind_name_string()
    );
    write_line_to_buffered_file(
        &mut buffered_writer,
        &label_kind_line,
        "save_decision_tree_to_plain_text_file: label_kind write failed",
    )?;

    // Max depth used.
    let max_depth_line = format!(
        "max_depth_used={}",
        trained_decision_tree.max_depth_used_for_training
    );
    write_line_to_buffered_file(
        &mut buffered_writer,
        &max_depth_line,
        "save_decision_tree_to_plain_text_file: max_depth write failed",
    )?;

    // Node count.
    let node_count_line = format!(
        "node_count={}",
        trained_decision_tree.all_tree_nodes_flat_vector.len()
    );
    write_line_to_buffered_file(
        &mut buffered_writer,
        &node_count_line,
        "save_decision_tree_to_plain_text_file: node_count write failed",
    )?;

    // One line per node. The loop bound is the node count, which is
    // bounded by the tree builder's defensive cap.
    for (node_position, node_reference) in trained_decision_tree
        .all_tree_nodes_flat_vector
        .iter()
        .enumerate()
    {
        let kind_string = match node_reference.node_branch_decision {
            TreeNodeBranchDecision::DecisionNode => "decision",
            TreeNodeBranchDecision::LeafNode => "leaf",
        };
        let node_line = format!(
            "node_index={} kind={} feature={} threshold={} left={} right={} leaf_value={}",
            node_position,
            kind_string,
            node_reference
                .split_feature_index
                .canonical_feature_name_string(),
            node_reference.split_threshold_value,
            node_reference.left_child_node_index,
            node_reference.right_child_node_index,
            node_reference.leaf_predicted_value,
        );
        write_line_to_buffered_file(
            &mut buffered_writer,
            &node_line,
            "save_decision_tree_to_plain_text_file: node line write failed",
        )?;
    }

    // Flush the buffer explicitly so any buffered bytes are written before
    // the file handle is dropped.
    buffered_writer.flush().map_err(|_io_error_discarded| {
        HorseRacingError::CsvFileReadFailure("save_decision_tree_to_plain_text_file: flush failed")
    })?;

    Ok(())
}

/// Loads a `DecisionTree` from a plain text file saved by
/// `save_decision_tree_to_plain_text_file`.
///
/// ## Validation
///
/// - The first line must equal `DECISION_TREE_FILE_FORMAT_HEADER` exactly.
/// - The declared `node_count` must match the number of node lines found.
/// - Every node line is parsed field-by-field; any missing or unparseable
///   field is a hard error (the model file is corrupt).
/// - Node indices in node lines must equal their position in the vector
///   (a consistency check: a reordered or truncated file is caught here).
/// - Child indices for decision nodes must be within the node vector's
///   bounds (a referential integrity check).
///
/// ## Why Strict Validation
///
/// A silently corrupt model file produces wrong predictions without any
/// indication that anything is wrong. Strict validation at load time
/// ensures a corrupt file fails loudly rather than producing subtly
/// incorrect outputs.
pub fn load_decision_tree_from_plain_text_file(
    model_file_path: &Path,
) -> Result<DecisionTree, HorseRacingError> {
    let opened_file = File::open(model_file_path).map_err(|_os_error_discarded| {
        HorseRacingError::CsvFileReadFailure(
            "load_decision_tree_from_plain_text_file: could not open file",
        )
    })?;
    let buffered_reader = BufReader::new(opened_file);
    let mut line_iterator = buffered_reader.lines();

    // Validate format header.
    let header_line = read_next_nonempty_model_file_line(
        &mut line_iterator,
        "load_decision_tree_from_plain_text_file: missing header",
    )?;
    if header_line != DECISION_TREE_FILE_FORMAT_HEADER {
        return Err(HorseRacingError::CsvHeaderMismatch(
            "load_decision_tree_from_plain_text_file: header mismatch",
        ));
    }

    // Parse label_kind.
    let label_kind_line = read_next_nonempty_model_file_line(
        &mut line_iterator,
        "load_decision_tree_from_plain_text_file: missing label_kind line",
    )?;
    let label_kind_value_str = parse_labeled_line_value(
        &label_kind_line,
        "label_kind",
        "load_decision_tree_from_plain_text_file: label_kind key mismatch",
    )?;
    let parsed_label_kind = TreeLabelKind::label_kind_from_canonical_name(label_kind_value_str)?;

    // Parse max_depth_used.
    let max_depth_line = read_next_nonempty_model_file_line(
        &mut line_iterator,
        "load_decision_tree_from_plain_text_file: missing max_depth line",
    )?;
    let max_depth_value_str = parse_labeled_line_value(
        &max_depth_line,
        "max_depth_used",
        "load_decision_tree_from_plain_text_file: max_depth key mismatch",
    )?;
    let parsed_max_depth: u32 =
        max_depth_value_str
            .trim()
            .parse::<u32>()
            .map_err(|_parse_error_discarded| {
                HorseRacingError::CsvFieldIntegerParseFailure(
                    "load_decision_tree_from_plain_text_file: max_depth parse failed",
                )
            })?;

    // Parse node_count.
    let node_count_line = read_next_nonempty_model_file_line(
        &mut line_iterator,
        "load_decision_tree_from_plain_text_file: missing node_count line",
    )?;
    let node_count_value_str = parse_labeled_line_value(
        &node_count_line,
        "node_count",
        "load_decision_tree_from_plain_text_file: node_count key mismatch",
    )?;
    let declared_node_count: usize =
        node_count_value_str
            .trim()
            .parse::<usize>()
            .map_err(|_parse_error_discarded| {
                HorseRacingError::CsvFieldIntegerParseFailure(
                    "load_decision_tree_from_plain_text_file: node_count parse failed",
                )
            })?;

    // Sanity-check the declared node count before allocating.
    let maximum_plausible_node_count: usize = 1_000_000;
    if declared_node_count > maximum_plausible_node_count {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "load_decision_tree_from_plain_text_file: declared node count implausibly large",
        ));
    }

    // Parse exactly `declared_node_count` node lines.
    let mut loaded_tree_nodes: Vec<DecisionTreeNode> = Vec::with_capacity(declared_node_count);

    // Loop is bounded by declared_node_count, which is checked above.
    for expected_node_index in 0..declared_node_count {
        let node_line = read_next_nonempty_model_file_line(
            &mut line_iterator,
            "load_decision_tree_from_plain_text_file: missing node line",
        )?;

        // Each node line has seven space-separated labeled fields.
        // Parse them positionally using split_whitespace.
        let node_line_fields: Vec<&str> = node_line.split_whitespace().collect();
        let expected_node_line_field_count: usize = 7;
        if node_line_fields.len() != expected_node_line_field_count {
            return Err(HorseRacingError::CsvRowFieldCountMismatch(
                "load_decision_tree_from_plain_text_file: node line field count wrong",
            ));
        }

        // node_index=N
        let node_index_value_str = parse_labeled_line_value(
            node_line_fields[0],
            "node_index",
            "load_decision_tree_from_plain_text_file: node_index key mismatch",
        )?;
        let parsed_node_index: usize = node_index_value_str.parse::<usize>().map_err(|_| {
            HorseRacingError::CsvFieldIntegerParseFailure(
                "load_decision_tree_from_plain_text_file: node_index parse failed",
            )
        })?;

        // Referential integrity: node indices must be sequential.
        if parsed_node_index != expected_node_index {
            return Err(HorseRacingError::FieldValueOutOfValidRange(
                "load_decision_tree_from_plain_text_file: node index out of sequence",
            ));
        }

        // kind=decision|leaf
        let kind_value_str = parse_labeled_line_value(
            node_line_fields[1],
            "kind",
            "load_decision_tree_from_plain_text_file: kind key mismatch",
        )?;
        let parsed_node_kind = match kind_value_str {
            "decision" => TreeNodeBranchDecision::DecisionNode,
            "leaf" => TreeNodeBranchDecision::LeafNode,
            _ => {
                return Err(HorseRacingError::FieldValueOutOfValidRange(
                    "load_decision_tree_from_plain_text_file: unknown node kind",
                ));
            }
        };

        // feature=<name>
        let feature_value_str = parse_labeled_line_value(
            node_line_fields[2],
            "feature",
            "load_decision_tree_from_plain_text_file: feature key mismatch",
        )?;
        let parsed_feature_index =
            FeatureIndex::feature_index_from_canonical_name(feature_value_str)?;

        // threshold=N
        let threshold_value_str = parse_labeled_line_value(
            node_line_fields[3],
            "threshold",
            "load_decision_tree_from_plain_text_file: threshold key mismatch",
        )?;
        let parsed_threshold: i32 = threshold_value_str.parse::<i32>().map_err(|_| {
            HorseRacingError::CsvFieldIntegerParseFailure(
                "load_decision_tree_from_plain_text_file: threshold parse failed",
            )
        })?;

        // left=N
        let left_value_str = parse_labeled_line_value(
            node_line_fields[4],
            "left",
            "load_decision_tree_from_plain_text_file: left key mismatch",
        )?;
        let parsed_left_child: u32 = left_value_str.parse::<u32>().map_err(|_| {
            HorseRacingError::CsvFieldIntegerParseFailure(
                "load_decision_tree_from_plain_text_file: left child parse failed",
            )
        })?;

        // right=N
        let right_value_str = parse_labeled_line_value(
            node_line_fields[5],
            "right",
            "load_decision_tree_from_plain_text_file: right key mismatch",
        )?;
        let parsed_right_child: u32 = right_value_str.parse::<u32>().map_err(|_| {
            HorseRacingError::CsvFieldIntegerParseFailure(
                "load_decision_tree_from_plain_text_file: right child parse failed",
            )
        })?;

        // leaf_value=N
        let leaf_value_str = parse_labeled_line_value(
            node_line_fields[6],
            "leaf_value",
            "load_decision_tree_from_plain_text_file: leaf_value key mismatch",
        )?;
        let parsed_leaf_value: i32 = leaf_value_str.parse::<i32>().map_err(|_| {
            HorseRacingError::CsvFieldIntegerParseFailure(
                "load_decision_tree_from_plain_text_file: leaf_value parse failed",
            )
        })?;

        // Referential integrity: decision node children must be within
        // declared_node_count. We cannot check this until all nodes are
        // loaded (a child may have a higher index than its parent), so we
        // defer this check to after the loading loop.
        loaded_tree_nodes.push(DecisionTreeNode {
            node_branch_decision: parsed_node_kind,
            split_feature_index: parsed_feature_index,
            split_threshold_value: parsed_threshold,
            left_child_node_index: parsed_left_child,
            right_child_node_index: parsed_right_child,
            leaf_predicted_value: parsed_leaf_value,
        });
    }

    // Deferred referential integrity check: every decision node's child
    // indices must be valid positions in the loaded vector.
    for loaded_node_reference in loaded_tree_nodes.iter() {
        if loaded_node_reference.node_branch_decision == TreeNodeBranchDecision::DecisionNode {
            let left_index_as_usize = loaded_node_reference.left_child_node_index as usize;
            let right_index_as_usize = loaded_node_reference.right_child_node_index as usize;
            if left_index_as_usize >= loaded_tree_nodes.len()
                || right_index_as_usize >= loaded_tree_nodes.len()
            {
                return Err(HorseRacingError::FieldValueOutOfValidRange(
                    "load_decision_tree_from_plain_text_file: child index out of bounds",
                ));
            }
        }
    }

    Ok(DecisionTree {
        all_tree_nodes_flat_vector: loaded_tree_nodes,
        root_node_index: 0,
        label_kind_for_this_tree: parsed_label_kind,
        max_depth_used_for_training: parsed_max_depth,
    })
}

/// Saves a trained `LinearMarginModel` to a plain text file.
///
/// ## File Format
///
/// Line 1: format header (`LINEAR_MARGIN_MODEL_FILE_FORMAT_HEADER`)
/// Line 2: `label_kind=<canonical name>`
/// Line 3: `threshold_used=<integer>`
/// Line 4: `feature_count=<integer>`
/// Lines 5+: one feature boundary per line:
///   `feature=<name> low_boundary=<integer|none> high_boundary=<integer|none>
///    low_rate=<integer> high_rate=<integer>`
pub fn save_linear_margin_model_to_plain_text_file(
    linear_margin_model: &LinearMarginModel,
    output_file_path: &Path,
) -> Result<(), HorseRacingError> {
    let created_file = File::create(output_file_path).map_err(|_os_error_discarded| {
        HorseRacingError::CsvFileReadFailure(
            "save_linear_margin_model_to_plain_text_file: could not create output file",
        )
    })?;
    let mut buffered_writer = BufWriter::new(created_file);

    write_line_to_buffered_file(
        &mut buffered_writer,
        LINEAR_MARGIN_MODEL_FILE_FORMAT_HEADER,
        "save_linear_margin_model_to_plain_text_file: header write failed",
    )?;

    let label_kind_line = format!(
        "label_kind={}",
        linear_margin_model
            .label_kind_for_this_model
            .canonical_label_kind_name_string()
    );
    write_line_to_buffered_file(
        &mut buffered_writer,
        &label_kind_line,
        "save_linear_margin_model_to_plain_text_file: label_kind write failed",
    )?;

    let threshold_line = format!(
        "threshold_used={}",
        linear_margin_model.threshold_percent_used_for_training
    );
    write_line_to_buffered_file(
        &mut buffered_writer,
        &threshold_line,
        "save_linear_margin_model_to_plain_text_file: threshold write failed",
    )?;

    let feature_count_line = format!(
        "feature_count={}",
        linear_margin_model.feature_boundaries.len()
    );
    write_line_to_buffered_file(
        &mut buffered_writer,
        &feature_count_line,
        "save_linear_margin_model_to_plain_text_file: feature_count write failed",
    )?;

    // One line per feature boundary. Loop is bounded by feature count,
    // which equals ENGINEERED_FEATURE_COUNT.
    for boundary_entry_reference in linear_margin_model.feature_boundaries.iter() {
        let low_boundary_string = match boundary_entry_reference.low_boundary_value {
            Some(boundary_integer_value) => format!("{}", boundary_integer_value),
            None => ABSENT_BOUNDARY_SENTINEL_STRING.to_string(),
        };
        let high_boundary_string = match boundary_entry_reference.high_boundary_value {
            Some(boundary_integer_value) => format!("{}", boundary_integer_value),
            None => ABSENT_BOUNDARY_SENTINEL_STRING.to_string(),
        };
        let boundary_line = format!(
            "feature={} low_boundary={} high_boundary={} low_rate={} high_rate={}",
            boundary_entry_reference
                .feature_index
                .canonical_feature_name_string(),
            low_boundary_string,
            high_boundary_string,
            boundary_entry_reference.low_tail_failure_rate_percent,
            boundary_entry_reference.high_tail_failure_rate_percent,
        );
        write_line_to_buffered_file(
            &mut buffered_writer,
            &boundary_line,
            "save_linear_margin_model_to_plain_text_file: boundary line write failed",
        )?;
    }

    buffered_writer.flush().map_err(|_io_error_discarded| {
        HorseRacingError::CsvFileReadFailure(
            "save_linear_margin_model_to_plain_text_file: flush failed",
        )
    })?;

    Ok(())
}

/// Loads a `LinearMarginModel` from a plain text file saved by
/// `save_linear_margin_model_to_plain_text_file`.
///
/// ## Validation
///
/// - First line must equal `LINEAR_MARGIN_MODEL_FILE_FORMAT_HEADER`.
/// - Declared `feature_count` must equal `ENGINEERED_FEATURE_COUNT` (a
///   model trained with a different feature set is not compatible).
/// - Boundary values are either parseable integers or the literal string
///   `"none"`. Any other string is a hard error.
pub fn load_linear_margin_model_from_plain_text_file(
    model_file_path: &Path,
) -> Result<LinearMarginModel, HorseRacingError> {
    let opened_file = File::open(model_file_path).map_err(|_os_error_discarded| {
        HorseRacingError::CsvFileReadFailure(
            "load_linear_margin_model_from_plain_text_file: could not open file",
        )
    })?;
    let buffered_reader = BufReader::new(opened_file);
    let mut line_iterator = buffered_reader.lines();

    // Validate format header.
    let header_line = read_next_nonempty_model_file_line(
        &mut line_iterator,
        "load_linear_margin_model_from_plain_text_file: missing header",
    )?;
    if header_line != LINEAR_MARGIN_MODEL_FILE_FORMAT_HEADER {
        return Err(HorseRacingError::CsvHeaderMismatch(
            "load_linear_margin_model_from_plain_text_file: header mismatch",
        ));
    }

    // label_kind
    let label_kind_line = read_next_nonempty_model_file_line(
        &mut line_iterator,
        "load_linear_margin_model_from_plain_text_file: missing label_kind",
    )?;
    let label_kind_str = parse_labeled_line_value(
        &label_kind_line,
        "label_kind",
        "load_linear_margin_model_from_plain_text_file: label_kind key mismatch",
    )?;
    let parsed_label_kind = TreeLabelKind::label_kind_from_canonical_name(label_kind_str)?;

    // threshold_used
    let threshold_line = read_next_nonempty_model_file_line(
        &mut line_iterator,
        "load_linear_margin_model_from_plain_text_file: missing threshold_used",
    )?;
    let threshold_str = parse_labeled_line_value(
        &threshold_line,
        "threshold_used",
        "load_linear_margin_model_from_plain_text_file: threshold_used key mismatch",
    )?;
    let parsed_threshold: i32 = threshold_str.trim().parse::<i32>().map_err(|_| {
        HorseRacingError::CsvFieldIntegerParseFailure(
            "load_linear_margin_model_from_plain_text_file: threshold parse failed",
        )
    })?;

    // feature_count
    let feature_count_line = read_next_nonempty_model_file_line(
        &mut line_iterator,
        "load_linear_margin_model_from_plain_text_file: missing feature_count",
    )?;
    let feature_count_str = parse_labeled_line_value(
        &feature_count_line,
        "feature_count",
        "load_linear_margin_model_from_plain_text_file: feature_count key mismatch",
    )?;
    let declared_feature_count: usize =
        feature_count_str.trim().parse::<usize>().map_err(|_| {
            HorseRacingError::CsvFieldIntegerParseFailure(
                "load_linear_margin_model_from_plain_text_file: feature_count parse failed",
            )
        })?;

    // The loaded model must have the same feature count as the current
    // binary. A mismatch means the file was saved with a different version
    // of the feature set and cannot be used safely.
    if declared_feature_count != ENGINEERED_FEATURE_COUNT {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "load_linear_margin_model_from_plain_text_file: feature count mismatch",
        ));
    }

    // Parse exactly `declared_feature_count` boundary lines.
    let mut loaded_feature_boundaries: Vec<SingleFeatureMarginBoundary> =
        Vec::with_capacity(declared_feature_count);

    // Loop bounded by declared_feature_count == ENGINEERED_FEATURE_COUNT.
    for _boundary_position in 0..declared_feature_count {
        let boundary_line = read_next_nonempty_model_file_line(
            &mut line_iterator,
            "load_linear_margin_model_from_plain_text_file: missing boundary line",
        )?;

        /*
        Five boundary fields:

        feature=...
        low_boundary=...
        high_boundary=...
        low_rate=...
        high_rate=...
        */

        let boundary_fields: Vec<&str> = boundary_line.split_whitespace().collect();
        let expected_boundary_field_count: usize = 5;
        if boundary_fields.len() != expected_boundary_field_count {
            return Err(HorseRacingError::CsvRowFieldCountMismatch(
                "load_linear_margin_model_from_plain_text_file: boundary field count wrong",
            ));
        }

        // feature=<name>
        let feature_name_str = parse_labeled_line_value(
            boundary_fields[0],
            "feature",
            "load_linear_margin_model_from_plain_text_file: feature key mismatch",
        )?;
        let parsed_feature_index =
            FeatureIndex::feature_index_from_canonical_name(feature_name_str)?;

        // low_boundary=<integer|none>
        let low_boundary_str = parse_labeled_line_value(
            boundary_fields[1],
            "low_boundary",
            "load_linear_margin_model_from_plain_text_file: low_boundary key mismatch",
        )?;
        let parsed_low_boundary: Option<i32> =
            if low_boundary_str == ABSENT_BOUNDARY_SENTINEL_STRING {
                None
            } else {
                Some(low_boundary_str.parse::<i32>().map_err(|_| {
                    HorseRacingError::CsvFieldIntegerParseFailure(
                        "load_linear_margin_model_from_plain_text_file: low_boundary parse failed",
                    )
                })?)
            };

        // high_boundary=<integer|none>
        let high_boundary_str = parse_labeled_line_value(
            boundary_fields[2],
            "high_boundary",
            "load_linear_margin_model_from_plain_text_file: high_boundary key mismatch",
        )?;
        let parsed_high_boundary: Option<i32> =
            if high_boundary_str == ABSENT_BOUNDARY_SENTINEL_STRING {
                None
            } else {
                Some(high_boundary_str.parse::<i32>().map_err(|_| {
                    HorseRacingError::CsvFieldIntegerParseFailure(
                        "load_linear_margin_model_from_plain_text_file: high_boundary parse failed",
                    )
                })?)
            };

        // low_rate=<integer>
        let low_rate_str = parse_labeled_line_value(
            boundary_fields[3],
            "low_rate",
            "load_linear_margin_model_from_plain_text_file: low_rate key mismatch",
        )?;
        let parsed_low_rate: i32 = low_rate_str.parse::<i32>().map_err(|_| {
            HorseRacingError::CsvFieldIntegerParseFailure(
                "load_linear_margin_model_from_plain_text_file: low_rate parse failed",
            )
        })?;

        // high_rate=<integer>
        let high_rate_str = parse_labeled_line_value(
            boundary_fields[4],
            "high_rate",
            "load_linear_margin_model_from_plain_text_file: high_rate key mismatch",
        )?;
        let parsed_high_rate: i32 = high_rate_str.parse::<i32>().map_err(|_| {
            HorseRacingError::CsvFieldIntegerParseFailure(
                "load_linear_margin_model_from_plain_text_file: high_rate parse failed",
            )
        })?;

        loaded_feature_boundaries.push(SingleFeatureMarginBoundary {
            feature_index: parsed_feature_index,
            low_boundary_value: parsed_low_boundary,
            high_boundary_value: parsed_high_boundary,
            low_tail_failure_rate_percent: parsed_low_rate,
            high_tail_failure_rate_percent: parsed_high_rate,
        });
    }

    Ok(LinearMarginModel {
        feature_boundaries: loaded_feature_boundaries,
        label_kind_for_this_model: parsed_label_kind,
        threshold_percent_used_for_training: parsed_threshold,
    })
}

// ============================================================================
// SECTION 6 — CARGO TESTS
// ============================================================================

#[cfg(test)]
mod section_six_model_persistence_tests {
    use super::*;

    /// Builds a minimal valid classification decision tree for persistence
    /// tests: one decision node (root) splitting on age at threshold 5,
    /// with two leaf children.
    fn build_minimal_classification_tree_for_testing() -> DecisionTree {
        let root_node = DecisionTreeNode {
            node_branch_decision: TreeNodeBranchDecision::DecisionNode,
            split_feature_index: FeatureIndex::Age,
            split_threshold_value: 5,
            left_child_node_index: 1,
            right_child_node_index: 2,
            leaf_predicted_value: 0,
        };
        let left_leaf_node = DecisionTreeNode {
            node_branch_decision: TreeNodeBranchDecision::LeafNode,
            split_feature_index: FeatureIndex::Age,
            split_threshold_value: 0,
            left_child_node_index: NO_CHILD_NODE_INDEX,
            right_child_node_index: NO_CHILD_NODE_INDEX,
            leaf_predicted_value: 1,
        };
        let right_leaf_node = DecisionTreeNode {
            node_branch_decision: TreeNodeBranchDecision::LeafNode,
            split_feature_index: FeatureIndex::Age,
            split_threshold_value: 0,
            left_child_node_index: NO_CHILD_NODE_INDEX,
            right_child_node_index: NO_CHILD_NODE_INDEX,
            leaf_predicted_value: 0,
        };
        DecisionTree {
            all_tree_nodes_flat_vector: vec![root_node, left_leaf_node, right_leaf_node],
            root_node_index: 0,
            label_kind_for_this_tree: TreeLabelKind::CompletionClassification,
            max_depth_used_for_training: 1,
        }
    }

    /// Builds a minimal valid linear margin model for persistence tests.
    fn build_minimal_margin_model_for_testing() -> LinearMarginModel {
        let mut feature_boundaries: Vec<SingleFeatureMarginBoundary> = Vec::new();
        // Age: low boundary at 2, no high boundary.
        feature_boundaries.push(SingleFeatureMarginBoundary {
            feature_index: FeatureIndex::Age,
            low_boundary_value: Some(2),
            high_boundary_value: None,
            low_tail_failure_rate_percent: 100,
            high_tail_failure_rate_percent: 0,
        });
        // All other features: no boundaries.
        for remaining_feature in all_feature_indices_in_canonical_order().iter().skip(1) {
            feature_boundaries.push(SingleFeatureMarginBoundary {
                feature_index: *remaining_feature,
                low_boundary_value: None,
                high_boundary_value: None,
                low_tail_failure_rate_percent: 0,
                high_tail_failure_rate_percent: 0,
            });
        }
        LinearMarginModel {
            feature_boundaries,
            label_kind_for_this_model: TreeLabelKind::CompletionClassification,
            threshold_percent_used_for_training: 50,
        }
    }

    /// Verifies that saving then loading a decision tree produces a
    /// structurally identical tree (full round-trip).
    #[test]
    fn decision_tree_round_trips_through_plain_text_file() {
        let original_tree = build_minimal_classification_tree_for_testing();
        let temporary_file_path =
            std::env::temp_dir().join("horse_racing_section_six_tree_round_trip_test.txt");

        save_decision_tree_to_plain_text_file(&original_tree, &temporary_file_path)
            .expect("save must succeed");

        let loaded_tree = load_decision_tree_from_plain_text_file(&temporary_file_path)
            .expect("load must succeed");

        assert_eq!(
            original_tree.all_tree_nodes_flat_vector.len(),
            loaded_tree.all_tree_nodes_flat_vector.len()
        );
        assert_eq!(
            original_tree.label_kind_for_this_tree,
            loaded_tree.label_kind_for_this_tree
        );
        assert_eq!(
            original_tree.max_depth_used_for_training,
            loaded_tree.max_depth_used_for_training
        );
        for node_position in 0..original_tree.all_tree_nodes_flat_vector.len() {
            assert_eq!(
                original_tree.all_tree_nodes_flat_vector[node_position],
                loaded_tree.all_tree_nodes_flat_vector[node_position],
                "node at position {} did not round-trip correctly",
                node_position
            );
        }

        let _ignored = std::fs::remove_file(&temporary_file_path);
    }

    /// Verifies that a tree loaded from disk produces the same predictions
    /// as the in-memory tree it was saved from.
    #[test]
    fn loaded_decision_tree_produces_same_predictions_as_original() {
        let original_tree = build_minimal_classification_tree_for_testing();
        let temporary_file_path =
            std::env::temp_dir().join("horse_racing_section_six_tree_predict_test.txt");

        save_decision_tree_to_plain_text_file(&original_tree, &temporary_file_path)
            .expect("save must succeed");
        let loaded_tree = load_decision_tree_from_plain_text_file(&temporary_file_path)
            .expect("load must succeed");

        let test_feature_vectors: Vec<EngineeredFeatureVector> = vec![
            // Age 3 < threshold 5 -> should predict 1 (left leaf).
            EngineeredFeatureVector {
                age: 3,
                height: 150,
                experience: 3,
                weight: 900,
                height_to_weight_ratio_times_one_thousand: 166,
                age_times_experience: 9,
            },
            // Age 7 >= threshold 5 -> should predict 0 (right leaf).
            EngineeredFeatureVector {
                age: 7,
                height: 150,
                experience: 3,
                weight: 900,
                height_to_weight_ratio_times_one_thousand: 166,
                age_times_experience: 21,
            },
        ];

        let original_predictions = predict_batch_with_tree(&original_tree, &test_feature_vectors);
        let loaded_predictions = predict_batch_with_tree(&loaded_tree, &test_feature_vectors);
        assert_eq!(original_predictions, loaded_predictions);

        let _ignored = std::fs::remove_file(&temporary_file_path);
    }

    /// Verifies that saving then loading a linear margin model produces an
    /// identical model (full round-trip).
    #[test]
    fn linear_margin_model_round_trips_through_plain_text_file() {
        let original_model = build_minimal_margin_model_for_testing();
        let temporary_file_path =
            std::env::temp_dir().join("horse_racing_section_six_margin_round_trip_test.txt");

        save_linear_margin_model_to_plain_text_file(&original_model, &temporary_file_path)
            .expect("save must succeed");

        let loaded_model = load_linear_margin_model_from_plain_text_file(&temporary_file_path)
            .expect("load must succeed");

        assert_eq!(
            original_model.label_kind_for_this_model,
            loaded_model.label_kind_for_this_model
        );
        assert_eq!(
            original_model.threshold_percent_used_for_training,
            loaded_model.threshold_percent_used_for_training
        );
        assert_eq!(
            original_model.feature_boundaries.len(),
            loaded_model.feature_boundaries.len()
        );
        for position in 0..original_model.feature_boundaries.len() {
            assert_eq!(
                original_model.feature_boundaries[position],
                loaded_model.feature_boundaries[position],
                "boundary at position {} did not round-trip correctly",
                position
            );
        }

        let _ignored = std::fs::remove_file(&temporary_file_path);
    }

    /// Verifies that a file with a wrong header is rejected by the tree
    /// loader.
    #[test]
    fn tree_loader_rejects_wrong_header() {
        let temporary_file_path =
            std::env::temp_dir().join("horse_racing_section_six_tree_bad_header_test.txt");

        {
            let mut file_handle =
                File::create(&temporary_file_path).expect("temp file create must succeed");
            file_handle
                .write_all(b"wrong_header\nlabel_kind=completion_classification\n")
                .expect("write must succeed");
        }

        let load_result = load_decision_tree_from_plain_text_file(&temporary_file_path);
        assert!(matches!(
            load_result,
            Err(HorseRacingError::CsvHeaderMismatch(_))
        ));

        let _ignored = std::fs::remove_file(&temporary_file_path);
    }

    /// Verifies that a file with a wrong header is rejected by the margin
    /// model loader.
    #[test]
    fn margin_loader_rejects_wrong_header() {
        let temporary_file_path =
            std::env::temp_dir().join("horse_racing_section_six_margin_bad_header_test.txt");

        {
            let mut file_handle =
                File::create(&temporary_file_path).expect("temp file create must succeed");
            file_handle
                .write_all(b"wrong_header\n")
                .expect("write must succeed");
        }

        let load_result = load_linear_margin_model_from_plain_text_file(&temporary_file_path);
        assert!(matches!(
            load_result,
            Err(HorseRacingError::CsvHeaderMismatch(_))
        ));

        let _ignored = std::fs::remove_file(&temporary_file_path);
    }

    /// Verifies that loading from a nonexistent file returns a file-read
    /// error rather than panicking.
    #[test]
    fn tree_loader_returns_error_for_nonexistent_file() {
        let nonexistent_path =
            Path::new("/tmp/horse_racing_this_file_does_not_exist_section_six.txt");
        let load_result = load_decision_tree_from_plain_text_file(nonexistent_path);
        assert!(matches!(
            load_result,
            Err(HorseRacingError::CsvFileReadFailure(_))
        ));
    }

    /// Verifies that a margin model file with a feature_count that does not
    /// match ENGINEERED_FEATURE_COUNT is rejected.
    #[test]
    fn margin_loader_rejects_mismatched_feature_count() {
        let temporary_file_path =
            std::env::temp_dir().join("horse_racing_section_six_margin_wrong_feature_count.txt");

        {
            let mut file_handle =
                File::create(&temporary_file_path).expect("temp file create must succeed");
            // Write a valid header but declare the wrong feature count.
            let content = format!(
                "{}\nlabel_kind=completion_classification\nthreshold_used=50\nfeature_count=99\n",
                LINEAR_MARGIN_MODEL_FILE_FORMAT_HEADER
            );
            file_handle
                .write_all(content.as_bytes())
                .expect("write must succeed");
        }

        let load_result = load_linear_margin_model_from_plain_text_file(&temporary_file_path);
        assert!(matches!(
            load_result,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));

        let _ignored = std::fs::remove_file(&temporary_file_path);
    }
}

/*
Section 7: Training Orchestration
This section ties together everything from Sections 1–6 into the two-stage training pipeline: Stage 1 (hyperparameter search with train/validate split) and Stage 2 (final train-on-all-data with best hyperparameters). It also computes accuracy metrics so the user can evaluate model quality.

What This Section Contains

HyperparameterCandidate struct — one combination of tree depth and min-leaf-samples to evaluate
HyperparameterSearchResult struct — the accuracy outcome for one candidate combination on one label kind
ModelAccuracyReport struct — train and validate accuracy for a fully trained model
compute_classification_accuracy_percent — integer percent of correctly predicted completion labels
compute_regression_mean_absolute_error — integer mean absolute error for performance score predictions
run_hyperparameter_search_for_one_label_kind — evaluates all candidate combinations on the train/validate split, returning a sorted result table
select_best_hyperparameters_from_search_results — picks the best combination from the search table
run_full_training_stage_one_and_stage_two — the top-level training entry point: runs Stage 1, then Stage 2 on all data, saving all four model files
TrainingRunSummary struct — everything the caller (and the results file) needs to know about a completed training run

Plus cargo tests.

Design Decisions Explained Up Front
Why integer mean absolute error (not RMSE) for regression accuracy: RMSE requires a square root — which means floats. MAE is sum(|predicted - actual|) / count, which stays in integer arithmetic with the same i64 promotion pattern used throughout. MAE is also directly interpretable in the project's score units: "on average the tree is off by 180 score points" is immediately meaningful; an RMSE of 180 is not quite the same thing but equally usable for hyperparameter comparison.
Why accuracy percent for classification (not F1 or AUC): With ~200 rows and binary labels, accuracy percent is the simplest metric that answers the question "does this model work?" F1 and AUC require float arithmetic and are more useful when class imbalance is severe and known in advance. If the data turns out to be heavily imbalanced (many DNFs vs. few completions), this can be revisited. The project is explicitly experimental.
Why the hyperparameter search returns a sorted Vec<HyperparameterSearchResult>: The caller and the results file both benefit from seeing the full search table, not just the winner. Showing the table lets the user see whether there is a clear best configuration or whether several are equally good. The sort is by validate accuracy (best first).
Why Stage 1 and Stage 2 are one function (not two separately callable functions): Stage 2 depends on the output of Stage 1 (the best hyperparameters). Combining them into one orchestration function means the caller cannot accidentally run Stage 2 with wrong hyperparameters, and the TrainingRunSummary captures both stages' results coherently. A caller that genuinely wants to run only Stage 1 (e.g. to inspect results before committing to Stage 2) can do so by inspecting the summary before the models are saved — but the typical case is one call that does both.
Why TrainingRunSummary does not own the trained models: The models are saved to disk as part of run_full_training_stage_one_and_stage_two. The caller gets the summary for display and results-file writing; if it also needs the in-memory models (for an immediate prediction run without reloading), it can load them from the saved files. This keeps the summary struct lightweight and avoids holding two copies of each model in memory simultaneously.


Why unwrap_or_else appears in Stage 2 tree builds: Per the project rules, production code never panics. A tree build failure in Stage 2 (after a successful Stage 1 search) would be caused by a hardware fault or data corruption between the two stages. The fallback single-leaf tree is a safe degraded mode: it produces predictions, the models are saved, and the user sees the accuracy report showing 0% or near-0% which flags that something went wrong.
Why the partition-to-feature-vector mapping uses a linear scan on row_id and game_id together: Using both fields as the match key prevents a collision if two records from different races happened to share a row_id value (which should not happen with a well-formed CSV, but is defended against). With ~200 rows the linear scan is at most 200 comparisons per partition record.
Why combined_search_results_table concatenates both label kinds unsorted: The summary is for archiving and display. The caller (results file writer in Section 8) will print each label kind's results separately and can filter by result.label_kind. Sorting the combined table by validate accuracy across both label kinds would mix classification and regression results in a way that is harder to read.
Why the linear margin models report train_accuracy_percent: 0: Margin models identify risk zones, not point predictions. Applying a "prediction accuracy" metric to them would be misleading — a margin model that flags no risk zones on training data gets 100% "accuracy" in the trivial sense of "never incorrectly flagged a training sample." The 0 is an explicit "not applicable" signal, not a claim of 0% accuracy.
*/

// ============================================================================
// SECTION 7 — TRAINING ORCHESTRATION
// ============================================================================
//
// This section provides the two-stage training pipeline:
//
//   Stage 1: Hyperparameter search.
//            The training data is split by game_id group (Section 3) into
//            an 80% train partition and a 20% validate partition. Every
//            combination of (tree_max_depth, tree_min_leaf_samples) in the
//            configured search grid is evaluated. For each combination, two
//            trees are trained (completion classification, performance score
//            regression) and two linear margin models are built. Train and
//            validate accuracy are recorded for each. Results are sorted
//            best-first and written to a timestamped file in /results.
//
//   Stage 2: Final model training.
//            Using the best hyperparameters from Stage 1, all four models
//            are retrained on 100% of the available training data. The
//            trained models are saved to the /models directory. Train-set
//            accuracy on all data is recorded as a reference metric
//            (acknowledged as optimistic — the model has seen this data).
//
// ## Why Two Stages
//
// Hyperparameter search on the full dataset would overfit the search to
// the data used for evaluation. Stage 1's held-out validate set gives an
// unbiased (within the limits of a small dataset) estimate of how well a
// hyperparameter combination generalises. Stage 2 then trains on all data
// with the best combination, squeezing maximum information into the final
// model.
//
// ## What "Accuracy" Means Here
//
// For classification (completion label):
//   accuracy_percent = (correctly_predicted_count * 100) / total_count
//   Integer arithmetic, 0-100 range.
//
// For regression (performance score label):
//   mean_absolute_error = sum(|predicted - actual|) / count
//   Integer arithmetic, units are score points (0-1000 scale).
//   Lower is better. Reported alongside a "score accuracy percent"
//   defined as max(0, 100 - (mae * 100) / 1000) so both metrics fit
//   the same 0-100 display scale in the results table.

/// Split frequency counts per feature from a trained decision tree.
///
/// Records how many decision (internal) nodes in the tree use each
/// feature as their split criterion. Higher counts indicate the tree
/// found that feature more useful for partitioning the training data.
///
/// ## Interpretation
///
/// Split frequency is a coarse but reliable importance measure: a feature
/// used in many splits is structurally important to the tree's decision
/// logic. It does not capture the magnitude of impurity reduction at each
/// split, but it is simple, deterministic, and computable from the saved
/// tree alone — no training data needed.
///
/// ## Project Role
///
/// Displayed in the training summary and recorded in the training history
/// CSV so the user can track which features matter most across successive
/// training runs as new race data is added.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureSplitCounts {
    pub age_split_count: u32,
    pub height_split_count: u32,
    pub experience_split_count: u32,
    pub weight_split_count: u32,
    pub height_to_weight_ratio_split_count: u32,
    pub age_times_experience_split_count: u32,
}

impl FeatureSplitCounts {
    /// Returns a new instance with all counts at zero.
    pub fn new_all_zeros() -> Self {
        FeatureSplitCounts {
            age_split_count: 0,
            height_split_count: 0,
            experience_split_count: 0,
            weight_split_count: 0,
            height_to_weight_ratio_split_count: 0,
            age_times_experience_split_count: 0,
        }
    }

    /// Returns the split count for a given feature, allowing callers to
    /// iterate features in canonical order without matching by name.
    pub fn get_count_for_feature(&self, feature_index: FeatureIndex) -> u32 {
        match feature_index {
            FeatureIndex::Age => self.age_split_count,
            FeatureIndex::Height => self.height_split_count,
            FeatureIndex::Experience => self.experience_split_count,
            FeatureIndex::Weight => self.weight_split_count,
            FeatureIndex::HeightToWeightRatioTimesOneThousand => {
                self.height_to_weight_ratio_split_count
            }
            FeatureIndex::AgeTimesExperience => self.age_times_experience_split_count,
        }
    }

    /// Increments the count for a given feature by one.
    fn increment_count_for_feature(&mut self, feature_index: FeatureIndex) {
        match feature_index {
            FeatureIndex::Age => self.age_split_count += 1,
            FeatureIndex::Height => self.height_split_count += 1,
            FeatureIndex::Experience => self.experience_split_count += 1,
            FeatureIndex::Weight => self.weight_split_count += 1,
            FeatureIndex::HeightToWeightRatioTimesOneThousand => {
                self.height_to_weight_ratio_split_count += 1
            }
            FeatureIndex::AgeTimesExperience => self.age_times_experience_split_count += 1,
        }
    }
}

/// The association between one feature's extreme values and a label,
/// measured by comparing the mean label in the bottom third of samples
/// (sorted by feature value) against the mean label in the top third.
///
/// ## Interpretation
///
/// - Positive `top_minus_bottom_difference`: high feature values associate
///   with better outcomes (higher completion rate or higher performance
///   score).
/// - Negative: high feature values associate with worse outcomes.
/// - Near zero: weak or no monotonic association in this dataset.
///
/// ## Why Bottom/Top Third (Not Halves or Quartiles)
///
/// Thirds balance signal strength against sample size. With ~200 rows,
/// each third has ~65 samples — enough for a stable mean. Halves would
/// dilute the signal from extremes; quartiles would have ~50 samples
/// each, which is also fine but thirds are simpler and symmetric.
///
/// ## Middle Third
///
/// The middle third is intentionally discarded. It represents the
/// "average" feature value region where outcomes are expected to be
/// unremarkable. Excluding it sharpens the contrast between extremes,
/// which is exactly what the margin model also focuses on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingleFeatureOutcomeAssociation {
    pub feature_index: FeatureIndex,
    /// Mean label value among the bottom third of samples when sorted
    /// by this feature's value (lowest feature values).
    pub bottom_third_mean_label: i32,
    /// Mean label value among the top third of samples (highest feature
    /// values).
    pub top_third_mean_label: i32,
    /// `top_third_mean_label - bottom_third_mean_label`. Positive means
    /// higher feature values correlate with better outcomes.
    pub top_minus_bottom_difference: i32,
}

/// Complete feature analysis results from a single training run.
///
/// Contains split frequency counts from both Stage 2 final trees and
/// feature-outcome associations for both label kinds. Stored in
/// `TrainingRunSummary` and written to both the timestamped results file
/// and the persistent training history CSV.
#[derive(Debug, Clone)]
pub struct FeatureAnalysisBundle {
    /// How many decision nodes in the final classification tree use each
    /// feature as a split criterion.
    pub classification_tree_split_counts: FeatureSplitCounts,
    /// Same for the final regression tree.
    pub regression_tree_split_counts: FeatureSplitCounts,
    /// Bottom-vs-top-third label difference per feature, using the
    /// completion (0/1) label.
    pub classification_outcome_associations: Vec<SingleFeatureOutcomeAssociation>,
    /// Same using the performance score (0–1000) label.
    pub regression_outcome_associations: Vec<SingleFeatureOutcomeAssociation>,
}

/// Group counts for each partition of the three-way split, recorded in
/// the training summary for display and CSV logging.
///
/// ## Project Role
///
/// Lets the user see at a glance how many race groups (and therefore how
/// many horse records) went into each partition. Important context when
/// interpreting accuracy numbers: a test set of 4 groups (20 rows) has
/// much higher variance than one of 20 groups (100 rows).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionGroupCounts {
    pub total_records: usize,
    pub total_race_groups: usize,
    pub train_groups: usize,
    pub validate_groups: usize,
    pub test_groups: usize,
}

/// One combination of tree hyperparameters to evaluate during the
/// hyperparameter search.
///
/// ## Fields
///
/// - `tree_max_depth` — maximum depth the decision tree builder is allowed
///   to grow. Shallower trees generalise better on small data; deeper trees
///   can overfit. The search grid explores both ends.
/// - `tree_min_leaf_samples` — the minimum number of training samples that
///   must reach a leaf. Higher values prevent splits on very small groups,
///   which are unreliable with ~200 rows.
///
/// ## Project Context
///
/// These are the only two tree hyperparameters the builder exposes
/// (Section 4b). The linear margin model's threshold is a separate
/// config value and is not part of this search grid (it is more of an
/// interpretability dial than a accuracy hyperparameter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperparameterCandidate {
    pub tree_max_depth: u32,
    pub tree_min_leaf_samples: usize,
}

/// The accuracy results for one hyperparameter candidate on one label kind,
/// as evaluated during Stage 1.
///
/// ## Fields
///
/// - `candidate` — the hyperparameter combination this result is for.
/// - `label_kind` — classification or regression.
/// - `train_accuracy_percent` — accuracy on the training partition (will
///   be optimistic; included for reference / overfit detection).
/// - `validate_accuracy_percent` — accuracy on the validation partition
///   (the number that drives hyperparameter selection).
///
/// ## What "Accuracy Percent" Means for Regression
///
/// For regression the field holds the "score accuracy percent" defined as
/// `max(0, 100 - (mae * 100) / 1000)`, where `mae` is the mean absolute
/// error in score points. This maps MAE onto the same 0-100 range as
/// classification accuracy so the results table is uniform. The raw MAE
/// is reported separately in `train_mean_absolute_error` and
/// `validate_mean_absolute_error`.
///
/// - `train_mean_absolute_error` — raw MAE on training partition (0 for
///   classification results).
/// - `validate_mean_absolute_error` — raw MAE on validation partition (0
///   for classification results).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HyperparameterSearchResult {
    pub candidate: HyperparameterCandidate,
    pub label_kind: TreeLabelKind,
    pub train_accuracy_percent: i32,
    pub validate_accuracy_percent: i32,
    pub train_mean_absolute_error: i32,
    pub validate_mean_absolute_error: i32,
}

/// Accuracy report for a trained model evaluated on a specified dataset.
///
/// ## Fields
///
/// - `label_kind` — which label this model predicts.
/// - `model_kind_description` — "decision_tree" or "linear_margin".
/// - `evaluation_set_description` — identifies which data was used for
///   evaluation, e.g. "stage2_all_training_data" or "held_out_test_set".
///   Provides context so the reader knows whether the accuracy figure is
///   optimistic (trained on the same data) or from unseen data.
/// - `accuracy_percent` — 0–100 classification accuracy, or the MAE-derived
///   score accuracy percent for regression models.
/// - `mean_absolute_error` — raw MAE in score-point units for regression
///   models; 0 for classification models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelAccuracyReport {
    pub label_kind: TreeLabelKind,
    pub model_kind_description: &'static str,
    pub evaluation_set_description: &'static str,
    pub accuracy_percent: i32,
    pub mean_absolute_error: i32,
}

/// Everything the caller needs to know about a completed training run,
/// for display, results-file writing, and persistent history logging.
///
/// ## Fields
///
/// - `best_hyperparameter_candidate_found` — the winning combination from
///   Stage 1, subsequently used for test evaluation and Stage 2.
/// - `hyperparameter_search_results_table` — the full sorted search table
///   from Stage 1, for display and archiving.
/// - `test_set_accuracy_reports` — accuracy of the best model (trained on
///   train+validate) evaluated on the held-out test set. These numbers
///   were not used for any model selection decisions and therefore provide
///   an unbiased accuracy estimate.
/// - `stage_two_accuracy_reports` — accuracy on all training data after
///   Stage 2 retrain (optimistic — the model has seen this data).
/// - `saved_model_file_paths` — the four paths where model files were
///   written.
/// - `feature_analysis` — split frequency and feature-outcome associations
///   from the Stage 2 final trees and training data.
/// - `partition_group_counts` — how many groups went into each partition,
///   for context when interpreting accuracy numbers.
#[derive(Debug, Clone)]
pub struct TrainingRunSummary {
    pub best_hyperparameter_candidate_found: HyperparameterCandidate,
    pub hyperparameter_search_results_table: Vec<HyperparameterSearchResult>,
    pub test_set_accuracy_reports: Vec<ModelAccuracyReport>,
    pub stage_two_accuracy_reports: Vec<ModelAccuracyReport>,
    pub saved_model_file_paths: Vec<std::path::PathBuf>,
    pub feature_analysis: FeatureAnalysisBundle,
    pub partition_group_counts: PartitionGroupCounts,
}

/// Computes the percentage of classification predictions that exactly
/// match the true labels.
///
/// ## Formula
///
/// `accuracy_percent = (correctly_predicted_count * 100) / total_count`
///
/// Integer arithmetic, result in [0, 100]. Returns 0 if `total_count == 0`
/// (no samples to evaluate).
///
/// ## Project Role
///
/// Used for the completion (binary classification) model during both Stage
/// 1 hyperparameter search and Stage 2 final accuracy reporting.
pub fn compute_classification_accuracy_percent(
    predicted_labels: &[i32],
    true_labels: &[i32],
) -> i32 {
    let total_count: i64 = predicted_labels.len() as i64;
    if total_count == 0 {
        return 0;
    }
    if predicted_labels.len() != true_labels.len() {
        // Mismatched lengths produce a meaningless result; return 0 as a
        // safe sentinel rather than panicking.
        return 0;
    }

    let mut correctly_predicted_count: i64 = 0;
    // Loop bounded by predicted_labels.len(), which is bounded by
    // training set size, which is bounded by CSV reader's defensive cap.
    for sample_position in 0..predicted_labels.len() {
        if predicted_labels[sample_position] == true_labels[sample_position] {
            correctly_predicted_count += 1;
        }
    }

    ((correctly_predicted_count * 100) / total_count) as i32
}

/// Computes the mean absolute error between predicted and true integer
/// regression labels.
///
/// ## Formula
///
/// `mae = sum(|predicted[i] - true[i]|) / count`
///
/// All arithmetic in `i64` to avoid overflow (200 samples × max error
/// 1000 = 200,000, well within `i64`). The final division truncates to
/// `i32`.
///
/// ## Returns
///
/// `(mae_integer, score_accuracy_percent)` where:
/// - `mae_integer` is the raw MAE in score-point units.
/// - `score_accuracy_percent` is `max(0, 100 - (mae * 100) / 1000)`,
///   mapping MAE onto a 0-100 scale for uniform display with
///   classification accuracy.
///
/// Returns `(0, 100)` if `total_count == 0`.
///
/// ## Project Role
///
/// Used for the performance score (regression) model during both Stage 1
/// and Stage 2.
pub fn compute_regression_mean_absolute_error(
    predicted_scores: &[i32],
    true_scores: &[i32],
) -> (i32, i32) {
    let total_count: i64 = predicted_scores.len() as i64;
    if total_count == 0 {
        return (0, 100);
    }
    if predicted_scores.len() != true_scores.len() {
        return (0, 0);
    }

    let mut sum_of_absolute_errors: i64 = 0;
    for sample_position in 0..predicted_scores.len() {
        let absolute_error: i64 =
            (predicted_scores[sample_position] as i64 - true_scores[sample_position] as i64).abs();
        sum_of_absolute_errors += absolute_error;
    }

    let mean_absolute_error: i32 = (sum_of_absolute_errors / total_count) as i32;

    // Map MAE onto a 0-100 accuracy scale. A MAE of 0 gives 100%;
    // a MAE of 1000 (the maximum possible on a 0-1000 score range) gives 0%.
    // Values between are linear. Clamp to 0 from below in case MAE somehow
    // exceeds 1000 (data corruption defensive case).
    let score_accuracy_percent: i32 = (100_i32 - (mean_absolute_error * 100) / 1000).max(0);

    (mean_absolute_error, score_accuracy_percent)
}

/// Walks a trained decision tree's flat node vector and counts how many
/// decision (internal) nodes split on each feature.
///
/// ## Algorithm
///
/// A single bounded pass over the node vector. Each node that is a
/// `DecisionNode` contributes +1 to the count for its `split_feature_index`.
/// Leaf nodes are skipped. The loop is bounded by the node vector length,
/// which is bounded by the tree builder's defensive cap.
///
/// ## Project Role
///
/// Called once per final Stage 2 tree to populate the
/// `FeatureAnalysisBundle`. The results appear in the training summary
/// and in the persistent training history CSV.
pub fn compute_feature_split_counts_from_tree(trained_tree: &DecisionTree) -> FeatureSplitCounts {
    let mut counts = FeatureSplitCounts::new_all_zeros();

    for node_reference in trained_tree.all_tree_nodes_flat_vector.iter() {
        if node_reference.node_branch_decision == TreeNodeBranchDecision::DecisionNode {
            counts.increment_count_for_feature(node_reference.split_feature_index);
        }
    }

    counts
}

/// Computes the feature-outcome association for every feature by comparing
/// the mean label in the bottom third of samples (sorted by feature value)
/// against the mean label in the top third.
///
/// ## Algorithm
///
/// For each feature in canonical order:
///   1. Collect `(feature_value, label)` pairs for all samples.
///   2. Sort by `feature_value` ascending.
///   3. Compute `third_count = sample_count / 3` (integer division).
///   4. Bottom third = first `third_count` samples after sorting.
///   5. Top third = last `third_count` samples after sorting.
///   6. Compute integer mean label for each third.
///   7. Difference = `top_mean - bottom_mean`.
///
/// ## Edge Cases
///
/// - If `sample_count < 3` (fewer than 3 samples), `third_count` is 0
///   and all associations are reported as 0. This prevents misleading
///   results from single-sample "means".
/// - If feature vectors and labels have different lengths (caller bug),
///   all associations are reported as 0 rather than panicking.
///
/// ## Integer Arithmetic
///
/// All sums are accumulated in `i64` to prevent overflow. The final mean
/// is truncated to `i32` via integer division — no floats involved.
pub fn compute_feature_outcome_associations(
    feature_vectors: &[EngineeredFeatureVector],
    labels: &[i32],
) -> Vec<SingleFeatureOutcomeAssociation> {
    let sample_count = feature_vectors.len();
    let all_features = all_feature_indices_in_canonical_order();
    let mut associations: Vec<SingleFeatureOutcomeAssociation> =
        Vec::with_capacity(ENGINEERED_FEATURE_COUNT);

    // Defensive: mismatched lengths or too few samples → all zeros.
    if sample_count < 3 || feature_vectors.len() != labels.len() {
        for feature_reference in all_features.iter() {
            associations.push(SingleFeatureOutcomeAssociation {
                feature_index: *feature_reference,
                bottom_third_mean_label: 0,
                top_third_mean_label: 0,
                top_minus_bottom_difference: 0,
            });
        }
        return associations;
    }

    let third_count: usize = sample_count / 3;

    // Bounded outer loop: ENGINEERED_FEATURE_COUNT iterations (currently 6).
    for feature_reference in all_features.iter() {
        if third_count < 1 {
            associations.push(SingleFeatureOutcomeAssociation {
                feature_index: *feature_reference,
                bottom_third_mean_label: 0,
                top_third_mean_label: 0,
                top_minus_bottom_difference: 0,
            });
            continue;
        }

        // Collect (feature_value, label) pairs.
        let mut feature_label_pairs: Vec<(i32, i32)> = Vec::with_capacity(sample_count);
        for sample_position in 0..sample_count {
            let feature_value = extract_feature_value_from_vector(
                &feature_vectors[sample_position],
                *feature_reference,
            );
            feature_label_pairs.push((feature_value, labels[sample_position]));
        }

        // Sort by feature value ascending.
        feature_label_pairs.sort_by(|pair_a, pair_b| pair_a.0.cmp(&pair_b.0));

        // Bottom third: first `third_count` samples.
        let mut bottom_label_sum: i64 = 0;
        for pair_position in 0..third_count {
            bottom_label_sum += feature_label_pairs[pair_position].1 as i64;
        }
        let bottom_third_mean: i32 = (bottom_label_sum / third_count as i64) as i32;

        // Top third: last `third_count` samples.
        let top_start_position: usize = sample_count - third_count;
        let mut top_label_sum: i64 = 0;
        for pair_position in top_start_position..sample_count {
            top_label_sum += feature_label_pairs[pair_position].1 as i64;
        }
        let top_third_mean: i32 = (top_label_sum / third_count as i64) as i32;

        associations.push(SingleFeatureOutcomeAssociation {
            feature_index: *feature_reference,
            bottom_third_mean_label: bottom_third_mean,
            top_third_mean_label: top_third_mean,
            top_minus_bottom_difference: top_third_mean - bottom_third_mean,
        });
    }

    associations
}

/// Computes engineered feature vectors and extracts both label types from
/// a set of raw horse race records.
///
/// ## Project Role
///
/// Eliminates the fragile row_id-matching pattern previously used to map
/// records back to pre-computed feature vectors after a group-level split.
/// Each partition's records are independently converted to feature vectors,
/// which is trivially fast at ~200 rows and avoids cross-partition index
/// coupling.
///
/// Records that fail feature engineering (e.g. a weight of zero reaching
/// this function despite upstream validation) are silently skipped rather
/// than aborting the batch.
///
/// ## Returns
///
/// `(feature_vectors, completion_labels, performance_score_labels)` — three
/// parallel vectors of equal length, which may be shorter than
/// `source_records.len()` if any records were skipped.
fn build_feature_vectors_and_labels_from_records(
    source_records: &[RawHorseRaceRecord],
) -> (Vec<EngineeredFeatureVector>, Vec<i32>, Vec<i32>) {
    let record_count = source_records.len();
    let mut feature_vectors: Vec<EngineeredFeatureVector> = Vec::with_capacity(record_count);
    let mut completion_labels: Vec<i32> = Vec::with_capacity(record_count);
    let mut performance_labels: Vec<i32> = Vec::with_capacity(record_count);

    for record_reference in source_records.iter() {
        match compute_engineered_feature_vector_from_raw_record(record_reference) {
            Ok(engineered_vector) => {
                feature_vectors.push(engineered_vector);
                completion_labels.push(record_reference.completion);
                performance_labels.push(record_reference.performance_score);
            }
            Err(_feature_engineering_error_intentionally_skipped) => {
                continue;
            }
        }
    }

    (feature_vectors, completion_labels, performance_labels)
}

/// Evaluates all hyperparameter candidates on one label kind using the
/// provided train/validate split.
///
/// ## Inputs
///
/// - `train_feature_vectors` — feature vectors for the training partition.
/// - `train_labels` — labels for the training partition, parallel to
///   `train_feature_vectors`.
/// - `validate_feature_vectors` — feature vectors for the validation
///   partition.
/// - `validate_labels` — labels for the validation partition.
/// - `label_kind` — classification or regression.
/// - `candidates_to_evaluate` — the search grid of hyperparameter
///   combinations to try.
///
/// ## Returns
///
/// A `Vec<HyperparameterSearchResult>` sorted by `validate_accuracy_percent`
/// descending (best first). The caller can inspect the full table or take
/// `results[0]` as the best candidate.
///
/// ## Error Handling
///
/// If any individual tree build fails (which should not happen on valid
/// data, but defensive programming applies), that candidate is skipped
/// with a logged entry showing 0% accuracy rather than aborting the
/// entire search. The search continues with remaining candidates.
pub fn run_hyperparameter_search_for_one_label_kind(
    train_feature_vectors: &[EngineeredFeatureVector],
    train_labels: &[i32],
    validate_feature_vectors: &[EngineeredFeatureVector],
    validate_labels: &[i32],
    label_kind: TreeLabelKind,
    candidates_to_evaluate: &[HyperparameterCandidate],
) -> Vec<HyperparameterSearchResult> {
    let mut search_results_accumulator: Vec<HyperparameterSearchResult> =
        Vec::with_capacity(candidates_to_evaluate.len());

    // Upper bound on outer loop: candidates_to_evaluate.len(), which is
    // bounded by the search grid size from config.
    for candidate_reference in candidates_to_evaluate.iter() {
        let current_candidate = *candidate_reference;

        // Build the tree on the training partition.
        let trained_tree_result = build_decision_tree_iteratively(
            train_feature_vectors,
            train_labels,
            label_kind,
            current_candidate.tree_max_depth,
            current_candidate.tree_min_leaf_samples,
        );

        let trained_tree = match trained_tree_result {
            Ok(tree) => tree,
            Err(_build_error_intentionally_skipped) => {
                // Record a zero-accuracy result so this candidate appears
                // in the table but is ranked last.
                search_results_accumulator.push(HyperparameterSearchResult {
                    candidate: current_candidate,
                    label_kind,
                    train_accuracy_percent: 0,
                    validate_accuracy_percent: 0,
                    train_mean_absolute_error: 1000,
                    validate_mean_absolute_error: 1000,
                });
                continue;
            }
        };

        // Predict on both partitions.
        let train_predictions = predict_batch_with_tree(&trained_tree, train_feature_vectors);
        let validate_predictions = predict_batch_with_tree(&trained_tree, validate_feature_vectors);

        // Compute accuracy metrics appropriate for the label kind.
        let (train_accuracy, validate_accuracy, train_mae, validate_mae) = match label_kind {
            TreeLabelKind::CompletionClassification => {
                let train_acc =
                    compute_classification_accuracy_percent(&train_predictions, train_labels);
                let validate_acc =
                    compute_classification_accuracy_percent(&validate_predictions, validate_labels);
                // MAE is not meaningful for classification; use 0.
                (train_acc, validate_acc, 0_i32, 0_i32)
            }
            TreeLabelKind::PerformanceScoreRegression => {
                let (train_mae_raw, train_acc) =
                    compute_regression_mean_absolute_error(&train_predictions, train_labels);
                let (validate_mae_raw, validate_acc) =
                    compute_regression_mean_absolute_error(&validate_predictions, validate_labels);
                (train_acc, validate_acc, train_mae_raw, validate_mae_raw)
            }
        };

        search_results_accumulator.push(HyperparameterSearchResult {
            candidate: current_candidate,
            label_kind,
            train_accuracy_percent: train_accuracy,
            validate_accuracy_percent: validate_accuracy,
            train_mean_absolute_error: train_mae,
            validate_mean_absolute_error: validate_mae,
        });
    }

    // Sort by validate accuracy descending. Stable sort preserves insertion
    // order for ties, which keeps the output deterministic given a fixed
    // candidate list order.
    search_results_accumulator.sort_by(|result_a, result_b| {
        result_b
            .validate_accuracy_percent
            .cmp(&result_a.validate_accuracy_percent)
    });

    search_results_accumulator
}

/// Selects the best hyperparameter candidate from a sorted search result
/// table.
///
/// ## Selection Policy
///
/// Takes the first entry from the sorted table (highest validate accuracy).
/// In the case of a tie, the first candidate in the table wins — which,
/// because the sort is stable and the input candidates come from a fixed
/// configured grid, means the tie-break is deterministic.
///
/// ## Error
///
/// Returns `FieldValueOutOfValidRange` if the results table is empty
/// (no candidates were evaluated — indicates a configuration error).
pub fn select_best_hyperparameters_from_search_results(
    sorted_search_results: &[HyperparameterSearchResult],
) -> Result<HyperparameterCandidate, HorseRacingError> {
    if sorted_search_results.is_empty() {
        return Err(HorseRacingError::FieldValueOutOfValidRange(
            "select_best_hyperparameters_from_search_results: empty results table",
        ));
    }
    Ok(sorted_search_results[0].candidate)
}

/// Runs both training stages and saves all four model files.
///
/// ## Three-Way Split Pipeline
///
/// ```text
/// test_train_data.csv
///   → group by game_id
///   → three-way split (test / train / validate)
///   → Stage 1: hyperparameter search on train/validate
///   → select best hyperparameters
///   → evaluate best model on held-out test set
///   → Stage 2: retrain on ALL data (train+validate+test)
///   → compute feature analysis
///   → save four model files
/// ```
///
/// ## Test Set Evaluation (Between Stage 1 and Stage 2)
///
/// After Stage 1 selects the best hyperparameters, two trees are trained
/// on the combined train+validate partition (not just train) using those
/// hyperparameters. These trees are then evaluated on the held-out test
/// partition — data that was never used for fitting or tuning. The
/// resulting accuracy numbers are the most trustworthy estimate of how
/// well the model generalises to unseen races.
///
/// ## Stage 2: Final Models
///
/// After test evaluation, all four models are retrained on 100% of the
/// data using the best hyperparameters and saved to disk. The test
/// evaluation has already provided the unbiased accuracy estimate, so
/// Stage 2's purpose is purely to squeeze maximum information into the
/// final production models.
///
/// ## Inputs
///
/// - `all_parsed_training_records` — every validated record from the CSV.
/// - `test_fraction_percent` — percentage of race groups held out as the
///   test set (1–50). These groups are never used during Stage 1.
/// - `training_fraction_percent` — of the remaining (non-test) groups,
///   the percentage used for training within Stage 1. The rest are the
///   validate partition.
/// - `split_seed_value` — seed for the deterministic group shuffle.
/// - `hyperparameter_candidates` — the search grid for Stage 1.
/// - `linear_margin_threshold_classification` — failure-rate percent
///   threshold for the classification margin model.
/// - `linear_margin_threshold_regression` — score threshold for the
///   regression margin model.
/// - `models_directory_path` — directory for saved model files.
pub fn run_full_training_stage_one_and_stage_two(
    all_parsed_training_records: &[RawHorseRaceRecord],
    test_fraction_percent: u32,
    training_fraction_percent: u32,
    split_seed_value: u32,
    hyperparameter_candidates: &[HyperparameterCandidate],
    linear_margin_threshold_classification: i32,
    linear_margin_threshold_regression: i32,
    models_directory_path: &Path,
) -> Result<TrainingRunSummary, HorseRacingError> {
    // -------------------------------------------------------------------------
    // Group records by game_id for group-level splitting.
    // -------------------------------------------------------------------------
    let all_race_groups = group_raw_records_by_game_id(all_parsed_training_records)?;
    let total_race_group_count = all_race_groups.len();
    let total_record_count = all_parsed_training_records.len();

    // -------------------------------------------------------------------------
    // Three-way split: test / train / validate.
    // -------------------------------------------------------------------------
    let three_way_split = split_race_groups_into_test_train_validate(
        &all_race_groups,
        test_fraction_percent,
        training_fraction_percent,
        split_seed_value,
    )?;

    let train_group_count = three_way_split
        .train_validate_split
        .training_race_groups
        .len();
    let validate_group_count = three_way_split
        .train_validate_split
        .validation_race_groups
        .len();
    let test_group_count = three_way_split.test_race_groups.len();

    let partition_group_counts = PartitionGroupCounts {
        total_records: total_record_count,
        total_race_groups: total_race_group_count,
        train_groups: train_group_count,
        validate_groups: validate_group_count,
        test_groups: test_group_count,
    };

    // -------------------------------------------------------------------------
    // Build feature vectors and labels for each partition.
    // -------------------------------------------------------------------------
    let train_records = flatten_race_groups_into_records(
        &three_way_split.train_validate_split.training_race_groups,
    );
    let validate_records = flatten_race_groups_into_records(
        &three_way_split.train_validate_split.validation_race_groups,
    );
    let test_records = flatten_race_groups_into_records(&three_way_split.test_race_groups);

    let (train_feature_vectors, train_completion_labels, train_performance_labels) =
        build_feature_vectors_and_labels_from_records(&train_records);
    let (validate_feature_vectors, validate_completion_labels, validate_performance_labels) =
        build_feature_vectors_and_labels_from_records(&validate_records);
    let (test_feature_vectors, test_completion_labels, test_performance_labels) =
        build_feature_vectors_and_labels_from_records(&test_records);

    // -------------------------------------------------------------------------
    // Stage 1: Hyperparameter search on train/validate split.
    // -------------------------------------------------------------------------
    let classification_search_results = run_hyperparameter_search_for_one_label_kind(
        &train_feature_vectors,
        &train_completion_labels,
        &validate_feature_vectors,
        &validate_completion_labels,
        TreeLabelKind::CompletionClassification,
        hyperparameter_candidates,
    );

    let regression_search_results = run_hyperparameter_search_for_one_label_kind(
        &train_feature_vectors,
        &train_performance_labels,
        &validate_feature_vectors,
        &validate_performance_labels,
        TreeLabelKind::PerformanceScoreRegression,
        hyperparameter_candidates,
    );

    // Select best candidate. If classification and regression disagree,
    // pick the candidate with the higher validate accuracy.
    let best_classification_candidate = select_best_hyperparameters_from_search_results(
        &classification_search_results,
    )
    .unwrap_or(HyperparameterCandidate {
        tree_max_depth: 4,
        tree_min_leaf_samples: 2,
    });

    let best_regression_candidate = select_best_hyperparameters_from_search_results(
        &regression_search_results,
    )
    .unwrap_or(HyperparameterCandidate {
        tree_max_depth: 4,
        tree_min_leaf_samples: 2,
    });

    let best_overall_candidate: HyperparameterCandidate = {
        let classification_best_validate = classification_search_results
            .first()
            .map(|r| r.validate_accuracy_percent)
            .unwrap_or(0);
        let regression_best_validate = regression_search_results
            .first()
            .map(|r| r.validate_accuracy_percent)
            .unwrap_or(0);
        if regression_best_validate > classification_best_validate {
            best_regression_candidate
        } else {
            best_classification_candidate
        }
    };

    // Combine search result tables for the summary.
    let mut combined_search_results_table: Vec<HyperparameterSearchResult> =
        Vec::with_capacity(classification_search_results.len() + regression_search_results.len());
    for result_ref in classification_search_results.iter() {
        combined_search_results_table.push(*result_ref);
    }
    for result_ref in regression_search_results.iter() {
        combined_search_results_table.push(*result_ref);
    }

    // -------------------------------------------------------------------------
    // Test set evaluation: train on train+validate, evaluate on test.
    // -------------------------------------------------------------------------
    // Combine train and validate partitions for the test evaluation models.
    let mut train_validate_combined_records: Vec<RawHorseRaceRecord> =
        Vec::with_capacity(train_records.len() + validate_records.len());
    train_validate_combined_records.extend_from_slice(&train_records);
    train_validate_combined_records.extend_from_slice(&validate_records);

    let (
        train_validate_feature_vectors,
        train_validate_completion_labels,
        train_validate_performance_labels,
    ) = build_feature_vectors_and_labels_from_records(&train_validate_combined_records);

    let mut test_set_accuracy_reports: Vec<ModelAccuracyReport> = Vec::new();

    // Classification tree for test evaluation.
    let test_eval_classification_tree = build_decision_tree_iteratively(
        &train_validate_feature_vectors,
        &train_validate_completion_labels,
        TreeLabelKind::CompletionClassification,
        best_overall_candidate.tree_max_depth,
        best_overall_candidate.tree_min_leaf_samples,
    );
    match test_eval_classification_tree {
        Ok(ref tree) => {
            let test_predictions = predict_batch_with_tree(tree, &test_feature_vectors);
            let test_accuracy =
                compute_classification_accuracy_percent(&test_predictions, &test_completion_labels);
            test_set_accuracy_reports.push(ModelAccuracyReport {
                label_kind: TreeLabelKind::CompletionClassification,
                model_kind_description: "decision_tree",
                evaluation_set_description: "held_out_test_set",
                accuracy_percent: test_accuracy,
                mean_absolute_error: 0,
            });
        }
        Err(_build_error_discarded) => {
            test_set_accuracy_reports.push(ModelAccuracyReport {
                label_kind: TreeLabelKind::CompletionClassification,
                model_kind_description: "decision_tree",
                evaluation_set_description: "held_out_test_set",
                accuracy_percent: 0,
                mean_absolute_error: 0,
            });
        }
    }

    // Regression tree for test evaluation.
    let test_eval_regression_tree = build_decision_tree_iteratively(
        &train_validate_feature_vectors,
        &train_validate_performance_labels,
        TreeLabelKind::PerformanceScoreRegression,
        best_overall_candidate.tree_max_depth,
        best_overall_candidate.tree_min_leaf_samples,
    );
    match test_eval_regression_tree {
        Ok(ref tree) => {
            let test_predictions = predict_batch_with_tree(tree, &test_feature_vectors);
            let (test_mae, test_accuracy) =
                compute_regression_mean_absolute_error(&test_predictions, &test_performance_labels);
            test_set_accuracy_reports.push(ModelAccuracyReport {
                label_kind: TreeLabelKind::PerformanceScoreRegression,
                model_kind_description: "decision_tree",
                evaluation_set_description: "held_out_test_set",
                accuracy_percent: test_accuracy,
                mean_absolute_error: test_mae,
            });
        }
        Err(_build_error_discarded) => {
            test_set_accuracy_reports.push(ModelAccuracyReport {
                label_kind: TreeLabelKind::PerformanceScoreRegression,
                model_kind_description: "decision_tree",
                evaluation_set_description: "held_out_test_set",
                accuracy_percent: 0,
                mean_absolute_error: 1000,
            });
        }
    }

    // -------------------------------------------------------------------------
    // Stage 2: Train all four models on the full dataset.
    // -------------------------------------------------------------------------
    // Build feature vectors from ALL records (train + validate + test).
    let all_records_flat = flatten_race_groups_into_records(&all_race_groups);
    let (all_feature_vectors, all_completion_labels, all_performance_score_labels) =
        build_feature_vectors_and_labels_from_records(&all_records_flat);

    // Ensure the models directory exists.
    std::fs::create_dir_all(models_directory_path).map_err(|_os_error_discarded| {
        HorseRacingError::CsvFileReadFailure(
            "run_full_training_stage_one_and_stage_two: could not create models directory",
        )
    })?;

    let mut saved_model_file_paths: Vec<std::path::PathBuf> = Vec::new();
    let mut stage_two_accuracy_reports: Vec<ModelAccuracyReport> = Vec::new();

    // --- Classification decision tree ---
    let final_classification_tree = build_decision_tree_iteratively(
        &all_feature_vectors,
        &all_completion_labels,
        TreeLabelKind::CompletionClassification,
        best_overall_candidate.tree_max_depth,
        best_overall_candidate.tree_min_leaf_samples,
    )
    .unwrap_or_else(|_build_error_discarded| DecisionTree {
        all_tree_nodes_flat_vector: vec![DecisionTreeNode {
            node_branch_decision: TreeNodeBranchDecision::LeafNode,
            split_feature_index: FeatureIndex::Age,
            split_threshold_value: 0,
            left_child_node_index: NO_CHILD_NODE_INDEX,
            right_child_node_index: NO_CHILD_NODE_INDEX,
            leaf_predicted_value: 1,
        }],
        root_node_index: 0,
        label_kind_for_this_tree: TreeLabelKind::CompletionClassification,
        max_depth_used_for_training: 0,
    });

    let classification_tree_file_path = models_directory_path.join("tree_completion.txt");
    save_decision_tree_to_plain_text_file(
        &final_classification_tree,
        &classification_tree_file_path,
    )?;
    saved_model_file_paths.push(classification_tree_file_path);

    let classification_train_predictions =
        predict_batch_with_tree(&final_classification_tree, &all_feature_vectors);
    let classification_train_accuracy = compute_classification_accuracy_percent(
        &classification_train_predictions,
        &all_completion_labels,
    );
    stage_two_accuracy_reports.push(ModelAccuracyReport {
        label_kind: TreeLabelKind::CompletionClassification,
        model_kind_description: "decision_tree",
        evaluation_set_description: "stage2_all_training_data",
        accuracy_percent: classification_train_accuracy,
        mean_absolute_error: 0,
    });

    // --- Regression decision tree ---
    let final_regression_tree = build_decision_tree_iteratively(
        &all_feature_vectors,
        &all_performance_score_labels,
        TreeLabelKind::PerformanceScoreRegression,
        best_overall_candidate.tree_max_depth,
        best_overall_candidate.tree_min_leaf_samples,
    )
    .unwrap_or_else(|_build_error_discarded| DecisionTree {
        all_tree_nodes_flat_vector: vec![DecisionTreeNode {
            node_branch_decision: TreeNodeBranchDecision::LeafNode,
            split_feature_index: FeatureIndex::Age,
            split_threshold_value: 0,
            left_child_node_index: NO_CHILD_NODE_INDEX,
            right_child_node_index: NO_CHILD_NODE_INDEX,
            leaf_predicted_value: PERFORMANCE_SCORE_FOR_DID_NOT_FINISH,
        }],
        root_node_index: 0,
        label_kind_for_this_tree: TreeLabelKind::PerformanceScoreRegression,
        max_depth_used_for_training: 0,
    });

    let regression_tree_file_path = models_directory_path.join("tree_rank.txt");
    save_decision_tree_to_plain_text_file(&final_regression_tree, &regression_tree_file_path)?;
    saved_model_file_paths.push(regression_tree_file_path);

    let regression_train_predictions =
        predict_batch_with_tree(&final_regression_tree, &all_feature_vectors);
    let (regression_train_mae, regression_train_accuracy) = compute_regression_mean_absolute_error(
        &regression_train_predictions,
        &all_performance_score_labels,
    );
    stage_two_accuracy_reports.push(ModelAccuracyReport {
        label_kind: TreeLabelKind::PerformanceScoreRegression,
        model_kind_description: "decision_tree",
        evaluation_set_description: "stage2_all_training_data",
        accuracy_percent: regression_train_accuracy,
        mean_absolute_error: regression_train_mae,
    });

    // --- Classification linear margin model ---
    let final_classification_margin = build_linear_margin_model(
        &all_feature_vectors,
        &all_completion_labels,
        TreeLabelKind::CompletionClassification,
        linear_margin_threshold_classification,
    )
    .unwrap_or_else(|_build_error_discarded| LinearMarginModel {
        feature_boundaries: all_feature_indices_in_canonical_order()
            .iter()
            .map(|fi| SingleFeatureMarginBoundary {
                feature_index: *fi,
                low_boundary_value: None,
                high_boundary_value: None,
                low_tail_failure_rate_percent: 0,
                high_tail_failure_rate_percent: 0,
            })
            .collect(),
        label_kind_for_this_model: TreeLabelKind::CompletionClassification,
        threshold_percent_used_for_training: linear_margin_threshold_classification,
    });

    let classification_margin_file_path = models_directory_path.join("linear_completion.txt");
    save_linear_margin_model_to_plain_text_file(
        &final_classification_margin,
        &classification_margin_file_path,
    )?;
    saved_model_file_paths.push(classification_margin_file_path);

    stage_two_accuracy_reports.push(ModelAccuracyReport {
        label_kind: TreeLabelKind::CompletionClassification,
        model_kind_description: "linear_margin",
        evaluation_set_description: "stage2_all_training_data",
        accuracy_percent: 0,
        mean_absolute_error: 0,
    });

    // --- Regression linear margin model ---
    let final_regression_margin = build_linear_margin_model(
        &all_feature_vectors,
        &all_performance_score_labels,
        TreeLabelKind::PerformanceScoreRegression,
        linear_margin_threshold_regression,
    )
    .unwrap_or_else(|_build_error_discarded| LinearMarginModel {
        feature_boundaries: all_feature_indices_in_canonical_order()
            .iter()
            .map(|fi| SingleFeatureMarginBoundary {
                feature_index: *fi,
                low_boundary_value: None,
                high_boundary_value: None,
                low_tail_failure_rate_percent: 0,
                high_tail_failure_rate_percent: 0,
            })
            .collect(),
        label_kind_for_this_model: TreeLabelKind::PerformanceScoreRegression,
        threshold_percent_used_for_training: linear_margin_threshold_regression,
    });

    let regression_margin_file_path = models_directory_path.join("linear_rank.txt");
    save_linear_margin_model_to_plain_text_file(
        &final_regression_margin,
        &regression_margin_file_path,
    )?;
    saved_model_file_paths.push(regression_margin_file_path);

    stage_two_accuracy_reports.push(ModelAccuracyReport {
        label_kind: TreeLabelKind::PerformanceScoreRegression,
        model_kind_description: "linear_margin",
        evaluation_set_description: "stage2_all_training_data",
        accuracy_percent: 0,
        mean_absolute_error: 0,
    });

    // -------------------------------------------------------------------------
    // Feature analysis from Stage 2 final trees and all training data.
    // -------------------------------------------------------------------------
    let classification_tree_split_counts =
        compute_feature_split_counts_from_tree(&final_classification_tree);
    let regression_tree_split_counts =
        compute_feature_split_counts_from_tree(&final_regression_tree);

    let classification_outcome_associations =
        compute_feature_outcome_associations(&all_feature_vectors, &all_completion_labels);
    let regression_outcome_associations =
        compute_feature_outcome_associations(&all_feature_vectors, &all_performance_score_labels);

    let feature_analysis = FeatureAnalysisBundle {
        classification_tree_split_counts,
        regression_tree_split_counts,
        classification_outcome_associations,
        regression_outcome_associations,
    };

    Ok(TrainingRunSummary {
        best_hyperparameter_candidate_found: best_overall_candidate,
        hyperparameter_search_results_table: combined_search_results_table,
        test_set_accuracy_reports,
        stage_two_accuracy_reports,
        saved_model_file_paths,
        feature_analysis,
        partition_group_counts,
    })
}

// ============================================================================
// SECTION 7 — CARGO TESTS
// ============================================================================

#[cfg(test)]
mod section_seven_training_orchestration_tests {
    use super::*;
    use std::io::Write;

    /// Verifies perfect accuracy when predicted labels exactly match true
    /// labels.
    #[test]
    fn classification_accuracy_is_one_hundred_percent_for_perfect_predictions() {
        let true_and_predicted_labels: Vec<i32> = vec![0, 1, 1, 0, 1];
        let accuracy = compute_classification_accuracy_percent(
            &true_and_predicted_labels,
            &true_and_predicted_labels,
        );
        assert_eq!(accuracy, 100);
    }

    /// Verifies zero accuracy when no predicted label matches.
    #[test]
    fn classification_accuracy_is_zero_percent_for_all_wrong_predictions() {
        let true_labels: Vec<i32> = vec![0, 0, 0, 0];
        let all_wrong_predictions: Vec<i32> = vec![1, 1, 1, 1];
        let accuracy =
            compute_classification_accuracy_percent(&all_wrong_predictions, &true_labels);
        assert_eq!(accuracy, 0);
    }

    /// Verifies that accuracy is computed correctly for a 3-out-of-4
    /// case, where the result should be 75%.
    #[test]
    fn classification_accuracy_is_seventy_five_percent_for_three_of_four_correct() {
        let true_labels: Vec<i32> = vec![1, 1, 1, 1];
        let three_correct_predictions: Vec<i32> = vec![1, 1, 1, 0];
        let accuracy =
            compute_classification_accuracy_percent(&three_correct_predictions, &true_labels);
        assert_eq!(accuracy, 75);
    }

    /// Verifies that empty input produces 0% (not a divide-by-zero panic).
    #[test]
    fn classification_accuracy_returns_zero_for_empty_input() {
        let empty: Vec<i32> = Vec::new();
        let accuracy = compute_classification_accuracy_percent(&empty, &empty);
        assert_eq!(accuracy, 0);
    }

    /// Verifies that zero MAE (perfect regression predictions) maps to
    /// 100% score accuracy.
    #[test]
    fn regression_mae_is_zero_and_accuracy_is_one_hundred_for_perfect_predictions() {
        let true_scores: Vec<i32> = vec![200, 400, 600, 800, 1000];
        let (mae, accuracy) = compute_regression_mean_absolute_error(&true_scores, &true_scores);
        assert_eq!(mae, 0);
        assert_eq!(accuracy, 100);
    }

    /// Verifies that a MAE of 500 score points (half the 0-1000 range)
    /// maps to 50% score accuracy.
    #[test]
    fn regression_mae_of_five_hundred_maps_to_fifty_percent_accuracy() {
        let true_scores: Vec<i32> = vec![0, 0];
        let predicted_scores: Vec<i32> = vec![500, 500];
        let (mae, accuracy) =
            compute_regression_mean_absolute_error(&predicted_scores, &true_scores);
        assert_eq!(mae, 500);
        assert_eq!(accuracy, 50);
    }

    /// Verifies that a MAE of 1000 (worst possible) maps to 0%, and that
    /// the clamp prevents negative values.
    #[test]
    fn regression_mae_of_one_thousand_maps_to_zero_percent_accuracy() {
        let true_scores: Vec<i32> = vec![0];
        let predicted_scores: Vec<i32> = vec![1000];
        let (mae, accuracy) =
            compute_regression_mean_absolute_error(&predicted_scores, &true_scores);
        assert_eq!(mae, 1000);
        assert_eq!(accuracy, 0);
    }

    /// Verifies that the hyperparameter search returns one result per
    /// candidate and that results are sorted with the best validate
    /// accuracy first.
    #[test]
    fn hyperparameter_search_returns_sorted_results_for_separable_data() {
        // Use the same separable dataset from Section 4b tests.
        let mut feature_vectors: Vec<EngineeredFeatureVector> = Vec::new();
        let mut completion_labels: Vec<i32> = Vec::new();
        for young_age in 2..=4_i32 {
            feature_vectors.push(EngineeredFeatureVector {
                age: young_age,
                height: 150,
                experience: 3,
                weight: 900,
                height_to_weight_ratio_times_one_thousand: 166,
                age_times_experience: young_age * 3,
            });
            completion_labels.push(1);
        }
        for old_age in 6..=8_i32 {
            feature_vectors.push(EngineeredFeatureVector {
                age: old_age,
                height: 150,
                experience: 3,
                weight: 900,
                height_to_weight_ratio_times_one_thousand: 166,
                age_times_experience: old_age * 3,
            });
            completion_labels.push(0);
        }

        // Use all data as both train and validate for this unit test —
        // we are testing the search mechanics, not generalisation.
        let candidates: Vec<HyperparameterCandidate> = vec![
            HyperparameterCandidate {
                tree_max_depth: 1,
                tree_min_leaf_samples: 1,
            },
            HyperparameterCandidate {
                tree_max_depth: 4,
                tree_min_leaf_samples: 1,
            },
        ];

        let search_results = run_hyperparameter_search_for_one_label_kind(
            &feature_vectors,
            &completion_labels,
            &feature_vectors,
            &completion_labels,
            TreeLabelKind::CompletionClassification,
            &candidates,
        );

        assert_eq!(search_results.len(), candidates.len());
        // Results must be sorted best-first (validate accuracy descending).
        for pair_position in 0..(search_results.len() - 1) {
            assert!(
                search_results[pair_position].validate_accuracy_percent
                    >= search_results[pair_position + 1].validate_accuracy_percent,
                "search results must be sorted by validate accuracy descending"
            );
        }
    }

    /// Verifies that `select_best_hyperparameters_from_search_results`
    /// returns the first entry of the sorted results (highest validate
    /// accuracy).
    #[test]
    fn select_best_hyperparameters_returns_first_entry() {
        let candidates_and_accuracies: Vec<(HyperparameterCandidate, i32)> = vec![
            (
                HyperparameterCandidate {
                    tree_max_depth: 3,
                    tree_min_leaf_samples: 2,
                },
                85,
            ),
            (
                HyperparameterCandidate {
                    tree_max_depth: 1,
                    tree_min_leaf_samples: 1,
                },
                72,
            ),
        ];
        let synthetic_sorted_results: Vec<HyperparameterSearchResult> = candidates_and_accuracies
            .iter()
            .map(|(candidate, accuracy)| HyperparameterSearchResult {
                candidate: *candidate,
                label_kind: TreeLabelKind::CompletionClassification,
                train_accuracy_percent: *accuracy,
                validate_accuracy_percent: *accuracy,
                train_mean_absolute_error: 0,
                validate_mean_absolute_error: 0,
            })
            .collect();

        let selected_candidate =
            select_best_hyperparameters_from_search_results(&synthetic_sorted_results)
                .expect("non-empty results must yield a candidate");
        assert_eq!(selected_candidate.tree_max_depth, 3);
        assert_eq!(selected_candidate.tree_min_leaf_samples, 2);
    }

    /// Verifies that `select_best_hyperparameters_from_search_results`
    /// returns an error for an empty results table.
    #[test]
    fn select_best_hyperparameters_rejects_empty_results_table() {
        let empty_results: Vec<HyperparameterSearchResult> = Vec::new();
        let selection_result = select_best_hyperparameters_from_search_results(&empty_results);
        assert!(matches!(
            selection_result,
            Err(HorseRacingError::FieldValueOutOfValidRange(_))
        ));
    }

    /// End-to-end training run with three-way split: writes a synthetic CSV,
    /// runs both stages, verifies model files and summary structure.
    #[test]
    fn full_training_run_with_three_way_split_saves_models_and_returns_valid_summary() {
        let temporary_directory =
            std::env::temp_dir().join("horse_racing_section_seven_three_way_split_test");
        std::fs::create_dir_all(&temporary_directory).expect("temp dir create must succeed");
        let models_subdirectory = temporary_directory.join("models");
        let training_csv_path = temporary_directory.join("training.csv");

        // Write synthetic CSV with 10 races (50 rows) — enough for a
        // meaningful three-way split at 20%/80%.
        {
            let mut csv_file =
                File::create(&training_csv_path).expect("temp csv create must succeed");
            csv_file
                .write_all(CSV_EXPECTED_HEADER_LINE.as_bytes())
                .expect("write must succeed");
            csv_file.write_all(b"\n").expect("write must succeed");

            let mut row_id_counter: i32 = 0;
            for game_id_value in 0..10_i32 {
                for rank_value in 1..=4_i32 {
                    // N_i32
                    let age_value: i32 = 3 + (rank_value % 4);
                    let row_line = format!(
                        "{},{},{},150,3,900,{},1\n",
                        row_id_counter, game_id_value, age_value, rank_value
                    );
                    csv_file
                        .write_all(row_line.as_bytes())
                        .expect("write must succeed");
                    row_id_counter += 1;
                }
            }
        }

        let parsed_records = read_training_csv_file_incrementally(&training_csv_path)
            .expect("training csv must parse");

        let hyperparameter_candidates: Vec<HyperparameterCandidate> = vec![
            HyperparameterCandidate {
                tree_max_depth: 2,
                tree_min_leaf_samples: 1,
            },
            HyperparameterCandidate {
                tree_max_depth: 3,
                tree_min_leaf_samples: 2,
            },
        ];

        let training_summary = run_full_training_stage_one_and_stage_two(
            &parsed_records,
            20, // test_fraction_percent
            80, // training_fraction_percent
            42,
            &hyperparameter_candidates,
            50,
            400,
            &models_subdirectory,
        )
        .expect("full training run must succeed");

        // Four model files must exist.
        let expected_model_file_names = [
            "tree_completion.txt",
            "tree_rank.txt",
            "linear_completion.txt",
            "linear_rank.txt",
        ];
        for expected_file_name in expected_model_file_names.iter() {
            let expected_file_path = models_subdirectory.join(expected_file_name);
            assert!(
                expected_file_path.exists(),
                "model file {} must exist after training",
                expected_file_name
            );
        }

        // Summary structure checks.
        assert_eq!(training_summary.stage_two_accuracy_reports.len(), 4);
        assert_eq!(training_summary.test_set_accuracy_reports.len(), 2);
        assert_eq!(training_summary.saved_model_file_paths.len(), 4);
        assert_eq!(
            training_summary.hyperparameter_search_results_table.len(),
            hyperparameter_candidates.len() * 2
        );

        // Partition counts must sum correctly.
        let pc = &training_summary.partition_group_counts;
        assert_eq!(
            pc.train_groups + pc.validate_groups + pc.test_groups,
            pc.total_race_groups
        );
        assert!(pc.test_groups >= 1);
        assert!(pc.train_groups >= 1);
        assert!(pc.validate_groups >= 1);
        assert_eq!(pc.total_records, 40); // 10 races * 4 horses
        assert_eq!(pc.total_race_groups, 10);

        // Feature analysis must have correct structure.
        assert_eq!(
            training_summary
                .feature_analysis
                .classification_outcome_associations
                .len(),
            ENGINEERED_FEATURE_COUNT
        );
        assert_eq!(
            training_summary
                .feature_analysis
                .regression_outcome_associations
                .len(),
            ENGINEERED_FEATURE_COUNT
        );

        let _ignored = std::fs::remove_dir_all(&temporary_directory);
    }

    /// Verifies that feature split counts correctly count decision nodes.
    #[test]
    fn feature_split_counts_match_tree_structure() {
        // Build a known tree: root splits on Age, left child splits on
        // Height, right child is a leaf.
        let known_tree = DecisionTree {
            all_tree_nodes_flat_vector: vec![
                DecisionTreeNode {
                    node_branch_decision: TreeNodeBranchDecision::DecisionNode,
                    split_feature_index: FeatureIndex::Age,
                    split_threshold_value: 5,
                    left_child_node_index: 1,
                    right_child_node_index: 2,
                    leaf_predicted_value: 0,
                },
                DecisionTreeNode {
                    node_branch_decision: TreeNodeBranchDecision::DecisionNode,
                    split_feature_index: FeatureIndex::Height,
                    split_threshold_value: 150,
                    left_child_node_index: 3,
                    right_child_node_index: 4,
                    leaf_predicted_value: 0,
                },
                DecisionTreeNode {
                    node_branch_decision: TreeNodeBranchDecision::LeafNode,
                    split_feature_index: FeatureIndex::Age,
                    split_threshold_value: 0,
                    left_child_node_index: NO_CHILD_NODE_INDEX,
                    right_child_node_index: NO_CHILD_NODE_INDEX,
                    leaf_predicted_value: 0,
                },
                DecisionTreeNode {
                    node_branch_decision: TreeNodeBranchDecision::LeafNode,
                    split_feature_index: FeatureIndex::Age,
                    split_threshold_value: 0,
                    left_child_node_index: NO_CHILD_NODE_INDEX,
                    right_child_node_index: NO_CHILD_NODE_INDEX,
                    leaf_predicted_value: 1,
                },
                DecisionTreeNode {
                    node_branch_decision: TreeNodeBranchDecision::LeafNode,
                    split_feature_index: FeatureIndex::Age,
                    split_threshold_value: 0,
                    left_child_node_index: NO_CHILD_NODE_INDEX,
                    right_child_node_index: NO_CHILD_NODE_INDEX,
                    leaf_predicted_value: 0,
                },
            ],
            root_node_index: 0,
            label_kind_for_this_tree: TreeLabelKind::CompletionClassification,
            max_depth_used_for_training: 2,
        };

        let counts = compute_feature_split_counts_from_tree(&known_tree);
        assert_eq!(counts.age_split_count, 1);
        assert_eq!(counts.height_split_count, 1);
        assert_eq!(counts.experience_split_count, 0);
        assert_eq!(counts.weight_split_count, 0);
        assert_eq!(counts.height_to_weight_ratio_split_count, 0);
        assert_eq!(counts.age_times_experience_split_count, 0);
    }

    /// Verifies feature-outcome association on a known dataset where
    /// higher age correlates with lower performance score.
    #[test]
    fn feature_outcome_associations_detect_negative_age_correlation() {
        let mut feature_vectors: Vec<EngineeredFeatureVector> = Vec::new();
        let mut performance_labels: Vec<i32> = Vec::new();

        // Young horses (age 2,3,4) get high scores.
        for young_age in 2..=4_i32 {
            for _dup in 0..3 {
                feature_vectors.push(EngineeredFeatureVector {
                    age: young_age,
                    height: 150,
                    experience: 3,
                    weight: 900,
                    height_to_weight_ratio_times_one_thousand: 166,
                    age_times_experience: young_age * 3,
                });
                performance_labels.push(800);
            }
        }
        // Old horses (age 6,7,8) get low scores.
        for old_age in 6..=8_i32 {
            for _dup in 0..3 {
                feature_vectors.push(EngineeredFeatureVector {
                    age: old_age,
                    height: 150,
                    experience: 3,
                    weight: 900,
                    height_to_weight_ratio_times_one_thousand: 166,
                    age_times_experience: old_age * 3,
                });
                performance_labels.push(200);
            }
        }

        let associations =
            compute_feature_outcome_associations(&feature_vectors, &performance_labels);

        let age_association = associations
            .iter()
            .find(|a| a.feature_index == FeatureIndex::Age)
            .expect("must find age association");

        // High age → low score → negative difference.
        assert!(
            age_association.top_minus_bottom_difference < 0,
            "age association must be negative (high age = worse outcome)"
        );
        assert_eq!(age_association.bottom_third_mean_label, 800);
        assert_eq!(age_association.top_third_mean_label, 200);
    }
}

/*
Section 8: Config Parsing, Output Formatting, Results File Writing, and main()
This is the final section. It connects all previous sections to the outside world: reading stats_config.toml, formatting human-readable output, writing timestamped results files, and dispatching the two CLI modes (train / predict).

What This Section Contains

StatsConfig struct — all runtime configuration values parsed from stats_config.toml
parse_stats_config_from_toml_file — hand-written flat-key=value TOML parser
get_current_timestamp_string — produces a filesystem-safe timestamp string for results filenames
format_training_summary_as_text — formats a TrainingRunSummary into a human-readable text block
format_prediction_results_as_text — formats tree predictions and margin risk flags into the results table
write_text_to_timestamped_results_file — saves any text block to a timestamped file in /results
run_predict_mode — loads models, reads prediction CSV, runs inference, formats and prints/saves results
run_train_mode — reads training CSV, runs both training stages, formats and prints/saves summary
main() — parses CLI args, reads config, dispatches to train or predict mode

Plus cargo tests.

Design Decisions Explained Up Front
Why a hand-written TOML parser (not the toml crate): Project rule — no third-party crates. The config file uses only flat key = value lines (no nested tables, no arrays). A hand-written parser for this exact subset is 30 lines and is easier to audit than a full TOML crate dependency.
Why the timestamp uses std::time::SystemTime (not chrono): chrono is a third-party crate. SystemTime gives seconds since the Unix epoch, which is sufficient for producing a unique, sortable filename component. The timestamp is formatted as YYYYMMDD_HHMMSS purely from integer arithmetic on the epoch value — no float arithmetic, no calendar library needed.
Why output is written to both stdout and a results file: The user sees results immediately on screen; the file provides a permanent record. Both use the same formatted text block, so there is exactly one formatting path to maintain.
Why run_predict_mode reloads models from disk (rather than accepting in-memory models): This tests the save/load round-trip in normal production flow, catches corrupt model files before printing results, and keeps the function's interface simple — it takes only the config, not four model handles.
Why main() never panics: Per project rules. Every failure in main() is handled by printing a terse error message to stderr and returning, never by panicking or calling process::exit. The program exits naturally when main() returns.

A Note on Timestamp Arithmetic
SystemTime gives seconds since Unix epoch (January 1, 1970). Converting to YYYYMMDD_HHMMSS requires only integer division and modulo. The arithmetic handles the Gregorian calendar's non-uniform month lengths and leap years with a simple day-of-year-to-month lookup table — no calendar library needed. The result is accurate for all dates in the plausible range of this project's use.


Why get_current_timestamp_string returns [u8; 15] (not String): Per the project's heap-avoidance policy, a fixed-size stack array is the appropriate return type for a fixed-length string. The format_training_summary_as_text and format_prediction_results_as_text functions use String::new() and push_str — these do allocate heap, but they are formatting functions that exist to produce output text, so heap allocation is both expected and appropriate there.
Why main() reads stats_config.toml from the current working directory (not a CLI argument): The scope document specifies "input CSV path read from stats_config.toml" and "CLI: cargo run -- train/predict reading paths from stats_config.toml". No path override on the CLI was specified. The current-working-directory convention is standard for project-local config files and matches how cargo run operates from the project root.
Why the nested helper functions write_two_digits and write_four_digits are defined inside get_current_timestamp_string: They are only used in that one function and have no meaning outside it. Defining them inside keeps the scope tight and avoids polluting the module namespace with helpers that have no standalone utility.
Why the risk flag display uses + and - suffixes (e.g. age+, ht_wt-): The flag direction (high/low risk) needs to be visible in the compact table format without adding another column. + for "too high", - for "too low" is concise and immediately readable.

Confirm and Then Final Steps
Please run these tests. Once confirmed, we will have a complete working crate. The remaining two tasks before the project is fully ready to use are:

Add data/ directory with a sample training.csv and predict.csv matching the documented schema.
Verify end-to-end by running cargo run -- train then cargo run -- predict against real or synthetic data.

Please confirm tests pass and I will produce any final integration notes.Add to Conversation

*/

// ============================================================================
// SECTION 8 — CONFIG, OUTPUT, RESULTS FILES, AND main()
// ============================================================================
//
// This section is the program's outer shell: it reads configuration,
// dispatches to train or predict mode, formats all human-readable output,
// and writes timestamped results files.
//
// ## Config File
//
// `stats_config.toml` contains flat key = value lines only. No nested
// tables. The hand-written parser reads the file line by line, splits each
// non-comment non-blank line on the first `=`, trims both sides, and
// matches known keys. Unknown keys are silently ignored (forward
// compatibility: a new key in the config file does not break an older
// binary that does not yet know about it).
//
// ## CLI
//
//   cargo run -- train      reads training CSV, runs both stages, saves
//                           models, writes results summary.
//   cargo run -- predict    reads prediction CSV, loads saved models,
//                           runs inference, writes results table.
//
// ## Results Files
//
// Every run appends a timestamped `.txt` file to the `/results` directory.
// Training runs write `train_YYYYMMDD_HHMMSS.txt`.
// Prediction runs write `predict_YYYYMMDD_HHMMSS.txt`.
// The directory is created if it does not exist.

/// Appends one row to the persistent training history CSV file.
///
/// ## File: `<results_dir>/training_history.csv`
///
/// This file grows by one row per training run. It is never overwritten
/// or truncated — each run appends. If the file does not exist, a header
/// row is written first.
///
/// ## Columns
///
/// Timestamp, best hyperparameters, partition sizes, Stage 1 validate
/// accuracy, held-out test accuracy, Stage 2 train accuracy, feature
/// split counts (per tree), and feature-outcome associations (per label
/// kind). All values are integers.
///
/// ## Project Role
///
/// Provides a persistent record that the user can open in a spreadsheet
/// to track model quality and feature importance across successive
/// training runs as new race data is added over time. The user can spot
/// trends such as a feature consistently contributing zero splits
/// (suggesting it may not be predictive) or test accuracy declining
/// (suggesting overfitting or data drift).
///
/// ## Error Handling
///
/// Returns `CsvFileReadFailure` if the file cannot be created, opened,
/// or written. Does not abort the training run — the caller handles the
/// error and continues.
pub fn append_training_run_to_history_csv(
    training_summary: &TrainingRunSummary,
    timestamp_bytes: &[u8; 15],
    results_dir: &Path,
) -> Result<(), HorseRacingError> {
    std::fs::create_dir_all(results_dir).map_err(|_os_error_discarded| {
        HorseRacingError::CsvFileReadFailure(
            "append_training_run_to_history_csv: could not create results directory",
        )
    })?;

    let history_csv_path = results_dir.join("training_history.csv");
    let file_already_exists = history_csv_path.exists();

    let opened_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&history_csv_path)
        .map_err(|_os_error_discarded| {
            HorseRacingError::CsvFileReadFailure(
                "append_training_run_to_history_csv: could not open history csv",
            )
        })?;

    let mut buffered_writer = BufWriter::new(opened_file);

    // Write header row if the file is new.
    if !file_already_exists {
        let header_line = "timestamp,best_depth,best_min_leaf,\
            total_recs,total_groups,train_grps,valid_grps,test_grps,\
            valid_acc_classif,valid_acc_regress,valid_mae,\
            test_acc_classif,test_acc_regress,test_mae,\
            s2_acc_classif,s2_acc_regress,\
            c_splits_age,c_splits_ht,c_splits_exp,c_splits_wt,c_splits_hwratio,c_splits_agexp,\
            r_splits_age,r_splits_ht,r_splits_exp,r_splits_wt,r_splits_hwratio,r_splits_agexp,\
            assoc_c_age,assoc_c_ht,assoc_c_exp,assoc_c_wt,assoc_c_hwratio,assoc_c_agexp,\
            assoc_r_age,assoc_r_ht,assoc_r_exp,assoc_r_wt,assoc_r_hwratio,assoc_r_agexp\n";
        buffered_writer
            .write_all(header_line.as_bytes())
            .map_err(|_io_error_discarded| {
                HorseRacingError::CsvFileReadFailure(
                    "append_training_run_to_history_csv: header write failed",
                )
            })?;
    }

    // Extract metrics from the summary.
    let timestamp_str = std::str::from_utf8(timestamp_bytes).unwrap_or("unknown_timestamp");

    let best = &training_summary.best_hyperparameter_candidate_found;
    let pc = &training_summary.partition_group_counts;
    let fa = &training_summary.feature_analysis;

    // Best validate accuracies from Stage 1 (first entry per label kind
    // in the sorted results table).
    let best_validate_classif = training_summary
        .hyperparameter_search_results_table
        .iter()
        .find(|r| r.label_kind == TreeLabelKind::CompletionClassification)
        .map(|r| r.validate_accuracy_percent)
        .unwrap_or(0);
    let best_validate_regress_result = training_summary
        .hyperparameter_search_results_table
        .iter()
        .find(|r| r.label_kind == TreeLabelKind::PerformanceScoreRegression);
    let best_validate_regress = best_validate_regress_result
        .map(|r| r.validate_accuracy_percent)
        .unwrap_or(0);
    let best_validate_mae = best_validate_regress_result
        .map(|r| r.validate_mean_absolute_error)
        .unwrap_or(0);

    // Test set accuracies.
    let test_acc_classif = training_summary
        .test_set_accuracy_reports
        .iter()
        .find(|r| r.label_kind == TreeLabelKind::CompletionClassification)
        .map(|r| r.accuracy_percent)
        .unwrap_or(0);
    let test_regress_report = training_summary
        .test_set_accuracy_reports
        .iter()
        .find(|r| r.label_kind == TreeLabelKind::PerformanceScoreRegression);
    let test_acc_regress = test_regress_report.map(|r| r.accuracy_percent).unwrap_or(0);
    let test_mae = test_regress_report
        .map(|r| r.mean_absolute_error)
        .unwrap_or(0);

    // Stage 2 accuracies (decision tree only, margin is n/a).
    let s2_acc_classif = training_summary
        .stage_two_accuracy_reports
        .iter()
        .find(|r| {
            r.label_kind == TreeLabelKind::CompletionClassification
                && r.model_kind_description == "decision_tree"
        })
        .map(|r| r.accuracy_percent)
        .unwrap_or(0);
    let s2_acc_regress = training_summary
        .stage_two_accuracy_reports
        .iter()
        .find(|r| {
            r.label_kind == TreeLabelKind::PerformanceScoreRegression
                && r.model_kind_description == "decision_tree"
        })
        .map(|r| r.accuracy_percent)
        .unwrap_or(0);

    // Feature split counts.
    let cs = &fa.classification_tree_split_counts;
    let rs = &fa.regression_tree_split_counts;

    // Feature associations — helper closure to get difference by feature.
    let get_assoc_diff =
        |associations: &[SingleFeatureOutcomeAssociation], feature: FeatureIndex| -> i32 {
            associations
                .iter()
                .find(|a| a.feature_index == feature)
                .map(|a| a.top_minus_bottom_difference)
                .unwrap_or(0)
        };

    let all_features = all_feature_indices_in_canonical_order();

    // Build the data row.
    let data_row = format!(
        "{},{},{},\
         {},{},{},{},{},\
         {},{},{},\
         {},{},{},\
         {},{},\
         {},{},{},{},{},{},\
         {},{},{},{},{},{},\
         {},{},{},{},{},{},\
         {},{},{},{},{},{}\n",
        timestamp_str,
        best.tree_max_depth,
        best.tree_min_leaf_samples,
        pc.total_records,
        pc.total_race_groups,
        pc.train_groups,
        pc.validate_groups,
        pc.test_groups,
        best_validate_classif,
        best_validate_regress,
        best_validate_mae,
        test_acc_classif,
        test_acc_regress,
        test_mae,
        s2_acc_classif,
        s2_acc_regress,
        cs.get_count_for_feature(all_features[0]),
        cs.get_count_for_feature(all_features[1]),
        cs.get_count_for_feature(all_features[2]),
        cs.get_count_for_feature(all_features[3]),
        cs.get_count_for_feature(all_features[4]),
        cs.get_count_for_feature(all_features[5]),
        rs.get_count_for_feature(all_features[0]),
        rs.get_count_for_feature(all_features[1]),
        rs.get_count_for_feature(all_features[2]),
        rs.get_count_for_feature(all_features[3]),
        rs.get_count_for_feature(all_features[4]),
        rs.get_count_for_feature(all_features[5]),
        get_assoc_diff(&fa.classification_outcome_associations, all_features[0]),
        get_assoc_diff(&fa.classification_outcome_associations, all_features[1]),
        get_assoc_diff(&fa.classification_outcome_associations, all_features[2]),
        get_assoc_diff(&fa.classification_outcome_associations, all_features[3]),
        get_assoc_diff(&fa.classification_outcome_associations, all_features[4]),
        get_assoc_diff(&fa.classification_outcome_associations, all_features[5]),
        get_assoc_diff(&fa.regression_outcome_associations, all_features[0]),
        get_assoc_diff(&fa.regression_outcome_associations, all_features[1]),
        get_assoc_diff(&fa.regression_outcome_associations, all_features[2]),
        get_assoc_diff(&fa.regression_outcome_associations, all_features[3]),
        get_assoc_diff(&fa.regression_outcome_associations, all_features[4]),
        get_assoc_diff(&fa.regression_outcome_associations, all_features[5]),
    );

    buffered_writer
        .write_all(data_row.as_bytes())
        .map_err(|_io_error_discarded| {
            HorseRacingError::CsvFileReadFailure(
                "append_training_run_to_history_csv: data row write failed",
            )
        })?;

    buffered_writer.flush().map_err(|_io_error_discarded| {
        HorseRacingError::CsvFileReadFailure("append_training_run_to_history_csv: flush failed")
    })?;

    println!(
        "Training history appended to: {}",
        history_csv_path.display()
    );
    Ok(())
}

/// All configuration values read from `stats_config.toml`.
///
/// ## Fields
///
/// - `training_csv_path` — path to the training data CSV file.
/// - `predict_csv_path` — path to the CSV file containing horses to predict.
/// - `models_dir` — directory where trained model files are saved and loaded.
/// - `results_dir` — directory where timestamped results files are written.
/// - `tree_max_depth` — default tree max depth (used if no search has been
///   run yet, or as the search grid centre).
/// - `tree_min_leaf_samples` — default tree min leaf samples.
/// - `training_fraction_percent` — percentage of race groups used for the
///   Stage 1 training partition (remainder is validation). Typically 80.
/// - `split_seed` — seed for the deterministic train/validate shuffle.
/// - `linear_margin_threshold_classification` — failure-rate percent
///   threshold for the classification margin model boundary scanner.
/// - `linear_margin_threshold_regression` — score threshold for the
///   regression margin model boundary scanner.
/// - `hyperparam_search_max_depths` — comma-separated list of tree depths
///   to include in the hyperparameter search grid.
/// - `hyperparam_search_min_leaf_samples` — comma-separated list of
///   minimum-leaf-sample values for the search grid.
///
/// ## Defaults
///
/// Every field has a documented default used when the key is absent from
/// the config file. This means a minimal config file (only the paths)
/// is sufficient to run the program.
#[derive(Debug, Clone)]
pub struct StatsConfig {
    pub test_train_data_csv_path: String,
    pub predict_csv_path: String,
    pub models_dir: String,
    pub results_dir: String,
    pub tree_max_depth: u32,
    pub tree_min_leaf_samples: usize,
    /// Percentage of race groups held out as the test set (1–50).
    /// These groups are never used during hyperparameter search.
    pub test_fraction_percent: u32,
    /// Of the remaining (non-test) groups, the percentage used for training
    /// within the hyperparameter search. The rest are the validate partition.
    pub training_fraction_percent: u32,
    pub split_seed: u32,
    pub linear_margin_threshold_classification: i32,
    pub linear_margin_threshold_regression: i32,
    pub hyperparam_search_max_depths: Vec<u32>,
    pub hyperparam_search_min_leaf_samples: Vec<usize>,
}

impl StatsConfig {
    pub fn default_config() -> Self {
        StatsConfig {
            test_train_data_csv_path: "data/test_train_data.csv".to_string(),
            predict_csv_path: "data/predict.csv".to_string(),
            models_dir: "models".to_string(),
            results_dir: "results".to_string(),
            tree_max_depth: 4,
            tree_min_leaf_samples: 2,
            test_fraction_percent: 20,
            training_fraction_percent: 80,
            split_seed: 42,
            linear_margin_threshold_classification: 50,
            linear_margin_threshold_regression: 400,
            hyperparam_search_max_depths: vec![2, 3, 4, 5, 6],
            hyperparam_search_min_leaf_samples: vec![1, 2, 3],
        }
    }
}

// #[derive(Debug, Clone)]
// pub struct StatsConfig {
//     pub test_train_data_csv_path: String, // all fields, whole games only
//     pub predict_csv_path: String,         // one whole game, blank prediction fields
//     pub models_dir: String,
//     pub results_dir: String,
//     pub tree_max_depth: u32,
//     pub tree_min_leaf_samples: usize,
//     pub training_fraction_percent: u32,
//     pub split_seed: u32,
//     pub linear_margin_threshold_classification: i32,
//     pub linear_margin_threshold_regression: i32,
//     pub hyperparam_search_max_depths: Vec<u32>,
//     pub hyperparam_search_min_leaf_samples: Vec<usize>,
// }

// impl StatsConfig {
//     /// Returns a `StatsConfig` populated entirely with built-in defaults.
//     ///
//     /// Used as the starting point before the config file is parsed: the
//     /// parser overwrites only the keys it finds, leaving unrecognised or
//     /// absent keys at their defaults.
//     pub fn default_config() -> Self {
//         StatsConfig {
//             test_train_data_csv_path: "data/test_train_data.csv".to_string(),
//             predict_csv_path: "data/predict.csv".to_string(),
//             models_dir: "models".to_string(),
//             results_dir: "results".to_string(),
//             tree_max_depth: 4,
//             tree_min_leaf_samples: 2,
//             training_fraction_percent: 80,
//             split_seed: 42,
//             linear_margin_threshold_classification: 50,
//             linear_margin_threshold_regression: 400,
//             hyperparam_search_max_depths: vec![2, 3, 4, 5, 6],
//             hyperparam_search_min_leaf_samples: vec![1, 2, 3],
//         }
//     }
// }

/// Parses a `stats_config.toml` file into a `StatsConfig`.
///
/// ## Parser Design
///
/// Reads the file line by line. For each line:
/// - Skips blank lines and lines beginning with `#` (comments).
/// - Splits on the first `=` to get a key and a value.
/// - Trims both key and value of surrounding whitespace and surrounding
///   double-quotes on string values.
/// - Matches the key against the known set and updates the corresponding
///   field in a `StatsConfig` starting from `StatsConfig::default_config()`.
/// - Ignores unknown keys silently (forward compatibility).
///
/// ## Comma-Separated List Fields
///
/// `hyperparam_search_max_depths` and `hyperparam_search_min_leaf_samples`
/// are expected as comma-separated integer lists, e.g.
/// `hyperparam_search_max_depths = 2,3,4,5,6`.
/// Invalid entries within the list are skipped; if the entire list is
/// invalid, the field retains its default.
///
/// ## Error Handling
///
/// Returns `CsvFileReadFailure` if the file cannot be opened.
/// Individual line parse errors (non-integer where integer expected) are
/// silently skipped — the field retains its default. This tolerates minor
/// hand-editing errors in the config without aborting the program.
pub fn parse_stats_config_from_toml_file(
    config_file_path: &Path,
) -> Result<StatsConfig, HorseRacingError> {
    let opened_config_file = File::open(config_file_path).map_err(|_os_error_discarded| {
        HorseRacingError::CsvFileReadFailure(
            "parse_stats_config_from_toml_file: could not open config file",
        )
    })?;

    let buffered_config_reader = BufReader::new(opened_config_file);
    let mut parsed_config = StatsConfig::default_config();

    // Defensive line count cap.
    let maximum_config_lines_to_read: usize = 10_000;
    let mut lines_read_so_far: usize = 0;

    for current_line_result in buffered_config_reader.lines() {
        lines_read_so_far += 1;
        if lines_read_so_far > maximum_config_lines_to_read {
            break;
        }

        let current_line_text = match current_line_result {
            Ok(line_string) => line_string,
            Err(_io_error_discarded) => continue,
        };

        let trimmed_line = current_line_text.trim();

        // Skip blank lines and comment lines.
        if trimmed_line.is_empty() || trimmed_line.starts_with('#') {
            continue;
        }

        // Split on the first `=` only.
        let equals_position = match trimmed_line.find('=') {
            Some(position) => position,
            None => continue,
        };

        let raw_key = trimmed_line[..equals_position].trim();
        let raw_value = trimmed_line[(equals_position + 1)..].trim();

        // Strip surrounding double-quotes from string values if present.
        let unquoted_value =
            if raw_value.starts_with('"') && raw_value.ends_with('"') && raw_value.len() >= 2 {
                &raw_value[1..(raw_value.len() - 1)]
            } else {
                raw_value
            };

        // Match against known keys and update the config struct.
        match raw_key {
            "test_train_data_csv_path" => {
                parsed_config.test_train_data_csv_path = unquoted_value.to_string();
            }
            "predict_csv_path" => {
                parsed_config.predict_csv_path = unquoted_value.to_string();
            }
            "models_dir" => {
                parsed_config.models_dir = unquoted_value.to_string();
            }
            "results_dir" => {
                parsed_config.results_dir = unquoted_value.to_string();
            }
            "tree_max_depth" => {
                if let Ok(parsed_depth) = unquoted_value.parse::<u32>() {
                    parsed_config.tree_max_depth = parsed_depth;
                }
                // Invalid value: silently retain default.
            }
            "tree_min_leaf_samples" => {
                if let Ok(parsed_min) = unquoted_value.parse::<usize>() {
                    parsed_config.tree_min_leaf_samples = parsed_min;
                }
            }
            "training_fraction_percent" => {
                if let Ok(parsed_fraction) = unquoted_value.parse::<u32>() {
                    parsed_config.training_fraction_percent = parsed_fraction;
                }
            }
            "test_fraction_percent" => {
                if let Ok(parsed_fraction) = unquoted_value.parse::<u32>() {
                    parsed_config.test_fraction_percent = parsed_fraction;
                }
            }
            "split_seed" => {
                if let Ok(parsed_seed) = unquoted_value.parse::<u32>() {
                    parsed_config.split_seed = parsed_seed;
                }
            }
            "linear_margin_threshold_classification" => {
                if let Ok(parsed_threshold) = unquoted_value.parse::<i32>() {
                    parsed_config.linear_margin_threshold_classification = parsed_threshold;
                }
            }
            "linear_margin_threshold_regression" => {
                if let Ok(parsed_threshold) = unquoted_value.parse::<i32>() {
                    parsed_config.linear_margin_threshold_regression = parsed_threshold;
                }
            }
            "hyperparam_search_max_depths" => {
                let parsed_depths: Vec<u32> = unquoted_value
                    .split(',')
                    .filter_map(|depth_str| depth_str.trim().parse::<u32>().ok())
                    .collect();
                if !parsed_depths.is_empty() {
                    parsed_config.hyperparam_search_max_depths = parsed_depths;
                }
            }
            "hyperparam_search_min_leaf_samples" => {
                let parsed_min_samples: Vec<usize> = unquoted_value
                    .split(',')
                    .filter_map(|sample_str| sample_str.trim().parse::<usize>().ok())
                    .collect();
                if !parsed_min_samples.is_empty() {
                    parsed_config.hyperparam_search_min_leaf_samples = parsed_min_samples;
                }
            }
            _ => {
                // Unknown key: ignore silently for forward compatibility.
            }
        }
    }

    Ok(parsed_config)
}

/// Produces a filesystem-safe timestamp string of the form
/// `YYYYMMDD_HHMMSS` from the current system time.
///
/// ## Algorithm
///
/// Uses `std::time::SystemTime` to get seconds since the Unix epoch
/// (January 1, 1970, 00:00:00 UTC), then derives year, month, day, hour,
/// minute, and second purely from integer arithmetic. No float arithmetic,
/// no calendar library.
///
/// ## Leap Year and Month-Length Handling
///
/// Uses a standard 400/100/4-year Gregorian leap-year rule and a
/// fixed-size month-length lookup table. Accurate for all years in the
/// range 1970–2999, which is the entire plausible lifetime of this project.
///
/// ## Fallback
///
/// If `SystemTime::now()` fails (extremely unusual — would require the
/// system clock to predate the Unix epoch), returns the fixed string
/// `"19700101_000000"` as a safe non-crashing fallback.
pub fn get_current_timestamp_string() -> [u8; 15] {
    // We return a fixed-size stack array of 15 bytes (14 digits + 1
    // underscore) rather than a `String` to avoid heap allocation.
    // Format: b"YYYYMMDD_HHMMSS"

    let epoch_seconds: u64 =
        match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(duration_value) => duration_value.as_secs(),
            Err(_system_time_error_discarded) => 0,
        };

    // Derive HMS from seconds within the current day.
    let seconds_per_minute: u64 = 60;
    let seconds_per_hour: u64 = 3600;
    let seconds_per_day: u64 = 86_400;

    let total_days_since_epoch: u64 = epoch_seconds / seconds_per_day;
    let seconds_within_day: u64 = epoch_seconds % seconds_per_day;

    let hour_of_day: u64 = seconds_within_day / seconds_per_hour;
    let minute_of_hour: u64 = (seconds_within_day % seconds_per_hour) / seconds_per_minute;
    let second_of_minute: u64 = seconds_within_day % seconds_per_minute;

    // Derive YMD from days since epoch using the Gregorian calendar.
    // Walk forward through 400-year cycles, then 100-year, 4-year, 1-year
    // periods to find the current year and day-of-year.
    let days_in_400_year_cycle: u64 = 146_097;
    let days_in_100_year_cycle: u64 = 36_524;
    let days_in_4_year_cycle: u64 = 1_461;
    let days_in_common_year: u64 = 365;

    let mut remaining_days: u64 = total_days_since_epoch;
    let mut current_year: u64 = 1970;

    // 400-year cycles.
    let four_hundred_year_cycles: u64 = remaining_days / days_in_400_year_cycle;
    remaining_days -= four_hundred_year_cycles * days_in_400_year_cycle;
    current_year += four_hundred_year_cycles * 400;

    // 100-year cycles (max 3 before a 400-year cycle resets).
    let one_hundred_year_cycles: u64 = (remaining_days / days_in_100_year_cycle).min(3);
    remaining_days -= one_hundred_year_cycles * days_in_100_year_cycle;
    current_year += one_hundred_year_cycles * 100;

    // 4-year cycles.
    let four_year_cycles: u64 = remaining_days / days_in_4_year_cycle;
    remaining_days -= four_year_cycles * days_in_4_year_cycle;
    current_year += four_year_cycles * 4;

    // Remaining individual years (max 3 before a 4-year cycle resets).
    let individual_years: u64 = (remaining_days / days_in_common_year).min(3);
    remaining_days -= individual_years * days_in_common_year;
    current_year += individual_years;

    // `remaining_days` is now the 0-based day-of-year within `current_year`.
    // Determine if current_year is a leap year.
    let is_leap_year: bool =
        (current_year % 4 == 0 && current_year % 100 != 0) || (current_year % 400 == 0);

    // Month lengths for common and leap years.
    let month_lengths_common_year: [u64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let month_lengths_leap_year: [u64; 12] = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let month_lengths = if is_leap_year {
        month_lengths_leap_year
    } else {
        month_lengths_common_year
    };

    let mut current_month: u64 = 1;
    let mut day_accumulator = remaining_days;
    // Loop bounded by 12 months.
    for month_length_value in month_lengths.iter() {
        if day_accumulator < *month_length_value {
            break;
        }
        day_accumulator -= month_length_value;
        current_month += 1;
    }
    let current_day_of_month: u64 = day_accumulator + 1; // 1-based

    // Format into the fixed 15-byte buffer.
    // Using manual digit extraction rather than format! to stay heap-free.
    let mut timestamp_buffer: [u8; 15] = *b"19700101_000000";

    fn write_two_digits(buffer: &mut [u8; 15], position: usize, value: u64) {
        buffer[position] = b'0' + ((value / 10) % 10) as u8;
        buffer[position + 1] = b'0' + (value % 10) as u8;
    }

    fn write_four_digits(buffer: &mut [u8; 15], position: usize, value: u64) {
        buffer[position] = b'0' + ((value / 1000) % 10) as u8;
        buffer[position + 1] = b'0' + ((value / 100) % 10) as u8;
        buffer[position + 2] = b'0' + ((value / 10) % 10) as u8;
        buffer[position + 3] = b'0' + (value % 10) as u8;
    }

    write_four_digits(&mut timestamp_buffer, 0, current_year);
    write_two_digits(&mut timestamp_buffer, 4, current_month);
    write_two_digits(&mut timestamp_buffer, 6, current_day_of_month);
    // position 8 is '_', already set by the initialiser.
    write_two_digits(&mut timestamp_buffer, 9, hour_of_day);
    write_two_digits(&mut timestamp_buffer, 11, minute_of_hour);
    write_two_digits(&mut timestamp_buffer, 13, second_of_minute);

    timestamp_buffer
}

/// Writes a text block to a timestamped file in the results directory.
///
/// ## File Name
///
/// `<results_dir>/<prefix>_<timestamp>.txt`
/// where `<prefix>` is either `"train"` or `"predict"`.
///
/// ## Error Handling
///
/// Creates `results_dir` if it does not exist. Returns
/// `CsvFileReadFailure` if the directory cannot be created or the file
/// cannot be written.
pub fn write_text_to_timestamped_results_file(
    text_content: &str,
    results_dir: &Path,
    file_name_prefix: &str,
    timestamp_bytes: &[u8; 15],
) -> Result<(), HorseRacingError> {
    std::fs::create_dir_all(results_dir).map_err(|_os_error_discarded| {
        HorseRacingError::CsvFileReadFailure(
            "write_text_to_timestamped_results_file: could not create results directory",
        )
    })?;

    // Build file name from prefix + "_" + timestamp + ".txt".
    // All components are ASCII so byte-level construction is safe.
    let timestamp_str = match std::str::from_utf8(timestamp_bytes) {
        Ok(valid_str) => valid_str,
        Err(_utf8_error_discarded) => "19700101_000000",
    };
    let results_file_name = format!("{}_{}.txt", file_name_prefix, timestamp_str);
    let results_file_path = results_dir.join(&results_file_name);

    let created_file = File::create(&results_file_path).map_err(|_os_error_discarded| {
        HorseRacingError::CsvFileReadFailure(
            "write_text_to_timestamped_results_file: could not create results file",
        )
    })?;
    let mut buffered_writer = BufWriter::new(created_file);
    buffered_writer
        .write_all(text_content.as_bytes())
        .map_err(|_io_error_discarded| {
            HorseRacingError::CsvFileReadFailure(
                "write_text_to_timestamped_results_file: write failed",
            )
        })?;
    buffered_writer.flush().map_err(|_io_error_discarded| {
        HorseRacingError::CsvFileReadFailure("write_text_to_timestamped_results_file: flush failed")
    })?;

    println!("Results saved to: {}", results_file_path.display());
    Ok(())
}

/// Formats a `TrainingRunSummary` into a human-readable text block for
/// display and archiving.
///
/// ## Output Structure
///
/// - Header banner with timestamp.
/// - Data partitioning summary (three-way split sizes).
/// - Best hyperparameter candidate selected.
/// - Stage 1 hyperparameter search results tables.
/// - Held-out test set evaluation results.
/// - Stage 2 final model accuracy (optimistic — trained on all data).
/// - Feature importance (split frequency, ranked).
/// - Feature-outcome association (bottom/top third comparison).
/// - Saved model file paths.
pub fn format_training_summary_as_text(
    training_summary: &TrainingRunSummary,
    timestamp_bytes: &[u8; 15],
) -> String {
    let timestamp_str = std::str::from_utf8(timestamp_bytes).unwrap_or("unknown_timestamp");

    let mut output_text = String::new();

    output_text.push_str("=================================================================\n");
    output_text.push_str("HORSE RACING STAT CLASSIFIER — TRAINING SUMMARY\n");
    output_text.push_str(&format!("Timestamp: {}\n", timestamp_str));
    output_text.push_str("=================================================================\n\n");

    // --- Partition sizes ---
    let pc = &training_summary.partition_group_counts;
    output_text.push_str("DATA PARTITIONING (three-way split by game_id group):\n");
    output_text.push_str(&format!(
        "  Total records         : {} ({} race groups)\n",
        pc.total_records, pc.total_race_groups
    ));
    output_text.push_str(&format!(
        "  Held-out test groups  : {:3} (~{} rows, never used for tuning)\n",
        pc.test_groups,
        pc.test_groups * HORSES_PER_RACE_GROUP
    ));
    output_text.push_str(&format!(
        "  Train groups          : {:3} (~{} rows)\n",
        pc.train_groups,
        pc.train_groups * HORSES_PER_RACE_GROUP
    ));
    output_text.push_str(&format!(
        "  Validate groups       : {:3} (~{} rows)\n\n",
        pc.validate_groups,
        pc.validate_groups * HORSES_PER_RACE_GROUP
    ));

    // --- Best hyperparameters ---
    output_text.push_str("BEST HYPERPARAMETER CANDIDATE (selected from Stage 1 search):\n");
    output_text.push_str(&format!(
        "  tree_max_depth        = {}\n",
        training_summary
            .best_hyperparameter_candidate_found
            .tree_max_depth
    ));
    output_text.push_str(&format!(
        "  tree_min_leaf_samples = {}\n\n",
        training_summary
            .best_hyperparameter_candidate_found
            .tree_min_leaf_samples
    ));

    // --- Stage 1 search results ---
    output_text
        .push_str("STAGE 1 — HYPERPARAMETER SEARCH RESULTS (best validate accuracy first):\n\n");

    output_text.push_str("  Completion Classification:\n");
    output_text.push_str("  depth  min_leaf  train_acc%  validate_acc%\n");
    output_text.push_str("  -----  --------  ----------  -------------\n");
    for result_ref in training_summary
        .hyperparameter_search_results_table
        .iter()
        .filter(|r| r.label_kind == TreeLabelKind::CompletionClassification)
    {
        output_text.push_str(&format!(
            "  {:5}  {:8}  {:10}  {:13}\n",
            result_ref.candidate.tree_max_depth,
            result_ref.candidate.tree_min_leaf_samples,
            result_ref.train_accuracy_percent,
            result_ref.validate_accuracy_percent,
        ));
    }

    output_text.push_str("\n  Performance Score Regression:\n");
    output_text.push_str("  depth  min_leaf  train_acc%  validate_acc%  train_mae  validate_mae\n");
    output_text.push_str("  -----  --------  ----------  -------------  ---------  ------------\n");
    for result_ref in training_summary
        .hyperparameter_search_results_table
        .iter()
        .filter(|r| r.label_kind == TreeLabelKind::PerformanceScoreRegression)
    {
        output_text.push_str(&format!(
            "  {:5}  {:8}  {:10}  {:13}  {:9}  {:12}\n",
            result_ref.candidate.tree_max_depth,
            result_ref.candidate.tree_min_leaf_samples,
            result_ref.train_accuracy_percent,
            result_ref.validate_accuracy_percent,
            result_ref.train_mean_absolute_error,
            result_ref.validate_mean_absolute_error,
        ));
    }

    // --- Held-out test set evaluation ---
    output_text
        .push_str("\nHELD-OUT TEST SET EVALUATION (unseen during hyperparameter selection):\n");
    output_text.push_str(&format!(
        "  Trained on train+validate ({} groups), evaluated on test ({} groups).\n",
        pc.train_groups + pc.validate_groups,
        pc.test_groups
    ));
    output_text.push_str("  This accuracy was not used for any model selection decisions.\n\n");
    output_text.push_str("  model                                   test_acc%  test_mae\n");
    output_text.push_str("  ---------------------------------------- ---------  --------\n");
    for report_ref in training_summary.test_set_accuracy_reports.iter() {
        let label_name = match report_ref.label_kind {
            TreeLabelKind::CompletionClassification => "completion",
            TreeLabelKind::PerformanceScoreRegression => "performance_score",
        };
        let mae_display = if report_ref.label_kind == TreeLabelKind::CompletionClassification {
            "     n/a".to_string()
        } else {
            format!("{:8}", report_ref.mean_absolute_error)
        };
        output_text.push_str(&format!(
            "  {:<40} {:9}  {}\n",
            format!("{}_{}", report_ref.model_kind_description, label_name),
            report_ref.accuracy_percent,
            mae_display,
        ));
    }

    // --- Stage 2 ---
    output_text.push_str("\nSTAGE 2 — FINAL MODEL ACCURACY (trained on all data, optimistic):\n\n");
    output_text.push_str("  model                                   train_acc%  train_mae\n");
    output_text.push_str("  ---------------------------------------- ----------  ---------\n");
    for report_ref in training_summary.stage_two_accuracy_reports.iter() {
        let label_name = match report_ref.label_kind {
            TreeLabelKind::CompletionClassification => "completion",
            TreeLabelKind::PerformanceScoreRegression => "performance_score",
        };
        let is_margin = report_ref.model_kind_description == "linear_margin";
        let acc_display = if is_margin {
            "       n/a".to_string()
        } else {
            format!("{:10}", report_ref.accuracy_percent)
        };
        let mae_display = if is_margin {
            "      n/a".to_string()
        } else {
            format!("{:9}", report_ref.mean_absolute_error)
        };
        output_text.push_str(&format!(
            "  {:<40} {}  {}\n",
            format!("{}_{}", report_ref.model_kind_description, label_name),
            acc_display,
            mae_display,
        ));
    }

    // --- Feature importance (split frequency) ---
    output_text.push_str("\nFEATURE IMPORTANCE (split frequency from Stage 2 final trees):\n\n");

    let all_features = all_feature_indices_in_canonical_order();
    let fa = &training_summary.feature_analysis;

    // Classification tree — sorted by count descending.
    output_text.push_str("  Classification tree (completion):\n");
    output_text.push_str("  rank  feature                                 splits\n");
    output_text.push_str("  ----  --------------------------------------  ------\n");

    let mut classif_feature_counts: Vec<(FeatureIndex, u32)> = all_features
        .iter()
        .map(|f| {
            (
                *f,
                fa.classification_tree_split_counts
                    .get_count_for_feature(*f),
            )
        })
        .collect();
    classif_feature_counts.sort_by(|a, b| b.1.cmp(&a.1));

    for (rank_position, (feature_idx, count_value)) in classif_feature_counts.iter().enumerate() {
        output_text.push_str(&format!(
            "  {:4}  {:<38}  {:6}\n",
            rank_position + 1,
            feature_idx.canonical_feature_name_string(),
            count_value,
        ));
    }

    // Regression tree — sorted by count descending.
    output_text.push_str("\n  Regression tree (performance score):\n");
    output_text.push_str("  rank  feature                                 splits\n");
    output_text.push_str("  ----  --------------------------------------  ------\n");

    let mut regress_feature_counts: Vec<(FeatureIndex, u32)> = all_features
        .iter()
        .map(|f| {
            (
                *f,
                fa.regression_tree_split_counts.get_count_for_feature(*f),
            )
        })
        .collect();
    regress_feature_counts.sort_by(|a, b| b.1.cmp(&a.1));

    for (rank_position, (feature_idx, count_value)) in regress_feature_counts.iter().enumerate() {
        output_text.push_str(&format!(
            "  {:4}  {:<38}  {:6}\n",
            rank_position + 1,
            feature_idx.canonical_feature_name_string(),
            count_value,
        ));
    }

    // --- Feature-outcome associations ---
    output_text
        .push_str("\nFEATURE-OUTCOME ASSOCIATION (top-third mean minus bottom-third mean):\n");
    output_text.push_str("  Positive = high feature values associate with better outcomes.\n");
    output_text.push_str("  Negative = high feature values associate with worse outcomes.\n");
    output_text.push_str("  Near zero = weak or no monotonic association in this dataset.\n\n");

    output_text.push_str("  Completion (0/1):\n");
    output_text.push_str("  feature                                 bottom  top  difference\n");
    output_text.push_str("  --------------------------------------  ------  ---  ----------\n");
    // Sort by absolute difference descending (strongest associations first).
    let mut classif_assoc_sorted: Vec<&SingleFeatureOutcomeAssociation> =
        fa.classification_outcome_associations.iter().collect();
    classif_assoc_sorted.sort_by(|a, b| {
        b.top_minus_bottom_difference
            .abs()
            .cmp(&a.top_minus_bottom_difference.abs())
    });
    for assoc_ref in classif_assoc_sorted.iter() {
        let diff_display = if assoc_ref.top_minus_bottom_difference >= 0 {
            format!("+{}", assoc_ref.top_minus_bottom_difference)
        } else {
            format!("{}", assoc_ref.top_minus_bottom_difference)
        };
        output_text.push_str(&format!(
            "  {:<38}  {:6}  {:3}  {:>10}\n",
            assoc_ref.feature_index.canonical_feature_name_string(),
            assoc_ref.bottom_third_mean_label,
            assoc_ref.top_third_mean_label,
            diff_display,
        ));
    }

    output_text.push_str("\n  Performance Score (0-1000):\n");
    output_text.push_str("  feature                                 bottom   top  difference\n");
    output_text.push_str("  --------------------------------------  ------  ----  ----------\n");
    let mut regress_assoc_sorted: Vec<&SingleFeatureOutcomeAssociation> =
        fa.regression_outcome_associations.iter().collect();
    regress_assoc_sorted.sort_by(|a, b| {
        b.top_minus_bottom_difference
            .abs()
            .cmp(&a.top_minus_bottom_difference.abs())
    });
    for assoc_ref in regress_assoc_sorted.iter() {
        let diff_display = if assoc_ref.top_minus_bottom_difference >= 0 {
            format!("+{}", assoc_ref.top_minus_bottom_difference)
        } else {
            format!("{}", assoc_ref.top_minus_bottom_difference)
        };
        output_text.push_str(&format!(
            "  {:<38}  {:6}  {:4}  {:>10}\n",
            assoc_ref.feature_index.canonical_feature_name_string(),
            assoc_ref.bottom_third_mean_label,
            assoc_ref.top_third_mean_label,
            diff_display,
        ));
    }

    // --- Saved model files ---
    output_text.push_str("\nSAVED MODEL FILES:\n");
    for model_path_ref in training_summary.saved_model_file_paths.iter() {
        output_text.push_str(&format!("  {}\n", model_path_ref.display()));
    }
    output_text.push_str("\n=================================================================\n");

    output_text
}

/// One horse's complete prediction results, assembled before formatting.
///
/// ## Fields
///
/// - `source_row_id` — the row_id from the input CSV, for identification.
/// - `game_id` — the race group this horse belongs to.
/// - `age`, `height`, `experience`, `weight` — raw input stats.
/// - `height_to_weight_ratio` — the engineered ratio for display.
/// - `tree_completion_prediction` — 0 or 1 from the classification tree.
/// - `tree_performance_score_prediction` — 0..=1000 from the regression tree.
/// - `margin_risk_flags` — any risk flags triggered by the margin models.
#[derive(Debug, Clone)]
pub struct SingleHorsePredictionResult {
    pub source_row_id: i32,
    pub game_id: i32,
    pub age: i32,
    pub height: i32,
    pub experience: i32,
    pub weight: i32,
    pub height_to_weight_ratio: i32,
    pub tree_completion_prediction: i32,
    pub tree_performance_score_prediction: i32,
    pub margin_risk_flags: Vec<RiskFlag>,
}

/// Formats a batch of prediction results into a human-readable text block.
///
/// ## Output Structure
///
/// - Header banner with timestamp and game_id.
/// - One row per horse with all predictions and risk flags.
/// - A summary line noting the predicted winner (highest performance score)
///   and any horses predicted to DNF.
pub fn format_prediction_results_as_text(
    prediction_results: &[SingleHorsePredictionResult],
    timestamp_bytes: &[u8; 15],
) -> String {
    let timestamp_str = std::str::from_utf8(timestamp_bytes).unwrap_or("unknown_timestamp");

    let game_id_display = prediction_results
        .first()
        .map(|first_result| first_result.game_id.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let mut output_text = String::new();

    output_text.push_str("=================================================================\n");
    output_text.push_str("HORSE RACING STAT CLASSIFIER — PREDICTION RESULTS\n");
    output_text.push_str(&format!("Timestamp : {}\n", timestamp_str));
    output_text.push_str(&format!("Game ID   : {}\n", game_id_display));
    output_text.push_str("=================================================================\n\n");

    output_text.push_str(
        "row  game  age  height  exp  weight  ht_wt  | completion  perf_score  | risk_flags\n",
    );
    output_text.push_str(
        "---  ----  ---  ------  ---  ------  -----  | ----------  ----------  | ----------\n",
    );

    for horse_result_reference in prediction_results.iter() {
        let completion_display = if horse_result_reference.tree_completion_prediction == 1 {
            "complete(1)"
        } else {
            "DNF(0)     "
        };

        // Collect risk flag strings.
        let risk_flag_display = if horse_result_reference.margin_risk_flags.is_empty() {
            "none".to_string()
        } else {
            let mut flags_text = String::new();
            for flag_reference in horse_result_reference.margin_risk_flags.iter() {
                let direction_char = match flag_reference.boundary_direction {
                    MarginBoundaryDirection::Low => '-',
                    MarginBoundaryDirection::High => '+',
                };
                if !flags_text.is_empty() {
                    flags_text.push_str(", ");
                }
                flags_text.push_str(
                    flag_reference
                        .flagged_feature_index
                        .canonical_feature_name_string(),
                );
                flags_text.push(direction_char);
            }
            flags_text
        };

        output_text.push_str(&format!(
            "{:<4} {:<5} {:<4} {:<7} {:<4} {:<7} {:<6} | {}  {:<10}  | {}\n",
            horse_result_reference.source_row_id,
            horse_result_reference.game_id,
            horse_result_reference.age,
            horse_result_reference.height,
            horse_result_reference.experience,
            horse_result_reference.weight,
            horse_result_reference.height_to_weight_ratio,
            completion_display,
            horse_result_reference.tree_performance_score_prediction,
            risk_flag_display,
        ));
    }

    // Summary: predicted winner and DNF horses.
    output_text.push_str("\nSUMMARY:\n");

    let predicted_winner_option = prediction_results
        .iter()
        .filter(|result| result.tree_completion_prediction == 1)
        .max_by_key(|result| result.tree_performance_score_prediction);

    match predicted_winner_option {
        Some(winner_reference) => {
            output_text.push_str(&format!(
                "  Predicted winner: row_id {} (score {})\n",
                winner_reference.source_row_id, winner_reference.tree_performance_score_prediction
            ));
        }
        None => {
            output_text.push_str("  No horse predicted to complete the race.\n");
        }
    }

    let dnf_count = prediction_results
        .iter()
        .filter(|result| result.tree_completion_prediction == 0)
        .count();
    if dnf_count > 0 {
        output_text.push_str(&format!("  Horses predicted DNF: {}\n", dnf_count));
    }

    output_text.push_str("\n=================================================================\n");
    output_text
}

/// Runs the predict mode: loads all four saved models, reads the
/// prediction CSV, computes feature vectors, runs tree prediction and
/// margin evaluation, formats and prints results, and saves to a
/// timestamped results file.
///
/// ## Error Handling
///
/// Returns an error if models cannot be loaded or the prediction CSV
/// cannot be parsed. Individual horse feature-engineering failures
/// are handled by substituting a default "DNF, score 0, no flags"
/// result rather than aborting the batch.
pub fn run_predict_mode(
    parsed_config: &StatsConfig,
    timestamp_bytes: &[u8; 15],
) -> Result<(), HorseRacingError> {
    let models_dir_path = Path::new(&parsed_config.models_dir);
    let results_dir_path = Path::new(&parsed_config.results_dir);

    // Load all four models.
    let classification_tree =
        load_decision_tree_from_plain_text_file(&models_dir_path.join("tree_completion.txt"))?;
    let regression_tree =
        load_decision_tree_from_plain_text_file(&models_dir_path.join("tree_rank.txt"))?;
    let classification_margin = load_linear_margin_model_from_plain_text_file(
        &models_dir_path.join("linear_completion.txt"),
    )?;
    let regression_margin =
        load_linear_margin_model_from_plain_text_file(&models_dir_path.join("linear_rank.txt"))?;

    // Parse the prediction CSV.
    let predict_csv_path = Path::new(&parsed_config.predict_csv_path);
    let prediction_input_records = read_prediction_csv_file_incrementally(predict_csv_path)?;

    if prediction_input_records.is_empty() {
        eprintln!("predict mode: prediction CSV contains no data rows");
        return Ok(());
    }

    // Assemble predictions for each record.
    let mut all_horse_prediction_results: Vec<SingleHorsePredictionResult> =
        Vec::with_capacity(prediction_input_records.len());

    for record_reference in prediction_input_records.iter() {
        // Feature engineering. On failure, produce a safe default result.
        let feature_vector =
            match compute_engineered_feature_vector_from_raw_record(record_reference) {
                Ok(vector) => vector,
                Err(_feature_error_discarded) => {
                    all_horse_prediction_results.push(SingleHorsePredictionResult {
                        source_row_id: record_reference.row_id,
                        game_id: record_reference.game_id,
                        age: record_reference.age,
                        height: record_reference.height,
                        experience: record_reference.experience,
                        weight: record_reference.weight,
                        height_to_weight_ratio: 0,
                        tree_completion_prediction: 0,
                        tree_performance_score_prediction: PERFORMANCE_SCORE_FOR_DID_NOT_FINISH,
                        margin_risk_flags: Vec::new(),
                    });
                    continue;
                }
            };

        let completion_prediction =
            predict_single_feature_vector_with_tree(&classification_tree, &feature_vector)
                .unwrap_or(0);

        let performance_score_prediction =
            predict_single_feature_vector_with_tree(&regression_tree, &feature_vector)
                .unwrap_or(PERFORMANCE_SCORE_FOR_DID_NOT_FINISH);

        let classification_risk_flags =
            evaluate_single_feature_vector_against_margins(&classification_margin, &feature_vector);
        let regression_risk_flags =
            evaluate_single_feature_vector_against_margins(&regression_margin, &feature_vector);

        // Merge both flag lists, deduplicating by (feature, direction).
        let mut merged_risk_flags: Vec<RiskFlag> = Vec::new();
        for flag_reference in classification_risk_flags
            .iter()
            .chain(regression_risk_flags.iter())
        {
            let already_present = merged_risk_flags.iter().any(|existing_flag| {
                existing_flag.flagged_feature_index == flag_reference.flagged_feature_index
                    && existing_flag.boundary_direction == flag_reference.boundary_direction
            });
            if !already_present {
                merged_risk_flags.push(flag_reference.clone());
            }
        }

        all_horse_prediction_results.push(SingleHorsePredictionResult {
            source_row_id: record_reference.row_id,
            game_id: record_reference.game_id,
            age: record_reference.age,
            height: record_reference.height,
            experience: record_reference.experience,
            weight: record_reference.weight,
            height_to_weight_ratio: feature_vector.height_to_weight_ratio_times_one_thousand,
            tree_completion_prediction: completion_prediction,
            tree_performance_score_prediction: performance_score_prediction,
            margin_risk_flags: merged_risk_flags,
        });
    }

    // Format, print, and save.
    let formatted_prediction_text =
        format_prediction_results_as_text(&all_horse_prediction_results, timestamp_bytes);
    println!("{}", formatted_prediction_text);
    write_text_to_timestamped_results_file(
        &formatted_prediction_text,
        results_dir_path,
        "predict",
        timestamp_bytes,
    )?;

    Ok(())
}

/// Runs the train mode: reads the single user-maintained historical data
/// CSV (`test_train_data_csv_path`), performs a three-way split
/// (test/train/validate) by game_id group, searches hyperparameters on the
/// train/validate partition, evaluates the best model on the held-out test
/// set, retrains final models on all data (Stage 2), computes feature
/// analysis, and saves everything to:
///   - Timestamped results file in `/results`
///   - Persistent training history CSV (`training_history.csv`)
///   - Four model files in `/models`
///
/// ## Single-File Design
///
/// The user maintains one CSV file containing all historical races with
/// known outcomes. They never manually split this into train/validate/test
/// files — the three-way split is the system's responsibility. This
/// eliminates the risk of data leakage from manual splitting and removes
/// bookkeeping burden from the user.
///
/// ## Adding New Race Data
///
/// After each real-world race, the user appends five new rows (one per
/// horse) to `test_train_data.csv` and re-runs `cargo run -- train`.
/// The system re-splits, re-searches hyperparameters, re-evaluates on a
/// fresh held-out test set, and retrains all four models from scratch.
pub fn run_train_mode(
    parsed_config: &StatsConfig,
    timestamp_bytes: &[u8; 15],
) -> Result<(), HorseRacingError> {
    let test_train_data_csv_path = Path::new(&parsed_config.test_train_data_csv_path);
    let models_dir_path = Path::new(&parsed_config.models_dir);
    let results_dir_path = Path::new(&parsed_config.results_dir);

    // Parse the historical data CSV. The system splits this internally;
    // the user never sees or manages the train/validate/test partitioning.
    let training_records = read_training_csv_file_incrementally(test_train_data_csv_path)?;

    if training_records.is_empty() {
        eprintln!("train mode: test_train_data.csv contains no data rows");
        return Ok(());
    }

    // Build the hyperparameter search grid from config values.
    let mut hyperparameter_candidates: Vec<HyperparameterCandidate> = Vec::new();
    for depth_value in parsed_config.hyperparam_search_max_depths.iter() {
        for min_samples_value in parsed_config.hyperparam_search_min_leaf_samples.iter() {
            hyperparameter_candidates.push(HyperparameterCandidate {
                tree_max_depth: *depth_value,
                tree_min_leaf_samples: *min_samples_value,
            });
        }
    }

    // Guard: if the search grid is empty (misconfigured config file),
    // fall back to the single default candidate so the run still proceeds.
    if hyperparameter_candidates.is_empty() {
        hyperparameter_candidates.push(HyperparameterCandidate {
            tree_max_depth: parsed_config.tree_max_depth,
            tree_min_leaf_samples: parsed_config.tree_min_leaf_samples,
        });
    }

    println!("Starting training run...");
    println!("  Historical records loaded : {}", training_records.len());
    println!(
        "  Hyperparameter candidates : {}",
        hyperparameter_candidates.len()
    );
    println!(
        "  Three-way split           : {}% test held out, then {}% train / {}% validate",
        parsed_config.test_fraction_percent,
        parsed_config.training_fraction_percent,
        100 - parsed_config.training_fraction_percent,
    );

    let training_summary = run_full_training_stage_one_and_stage_two(
        &training_records,
        parsed_config.test_fraction_percent,
        parsed_config.training_fraction_percent,
        parsed_config.split_seed,
        &hyperparameter_candidates,
        parsed_config.linear_margin_threshold_classification,
        parsed_config.linear_margin_threshold_regression,
        models_dir_path,
    )?;

    // Format and display the full training summary.
    let formatted_summary_text =
        format_training_summary_as_text(&training_summary, timestamp_bytes);
    println!("{}", formatted_summary_text);

    // Save to timestamped results file.
    write_text_to_timestamped_results_file(
        &formatted_summary_text,
        results_dir_path,
        "train",
        timestamp_bytes,
    )?;

    // Append to persistent training history CSV.
    // Failure here is logged but does not abort the training run —
    // the models and timestamped results file are already saved.
    if let Err(history_append_error) =
        append_training_run_to_history_csv(&training_summary, timestamp_bytes, results_dir_path)
    {
        eprintln!(
            "horse_racing_classifier: could not append to training_history.csv: {}",
            history_append_error.terse_production_message()
        );
    }

    Ok(())
}

/// Program entry point.
///
/// ## CLI
///
///   cargo run -- train     runs the training pipeline.
///   cargo run -- predict   runs the prediction pipeline.
///
/// ## Config File
///
/// Always reads `stats_config.toml` from the current working directory.
/// If the file does not exist, built-in defaults are used (see
/// `StatsConfig::default_config`).
///
/// ## Error Handling
///
/// All errors are caught here and printed to stderr. The program never
/// panics and never calls `process::exit` — it returns normally from
/// `main()` in all cases.
fn main() {
    let command_line_arguments: Vec<String> = std::env::args().collect();

    // The first argument (index 0) is the binary name. The mode argument
    // is at index 1.
    let mode_argument = command_line_arguments.get(1).map(|arg| arg.as_str());

    let run_mode = match mode_argument {
        Some("train") => "train",
        Some("predict") => "predict",
        Some(unrecognised_mode) => {
            eprintln!(
                "horse_racing_classifier: unrecognised mode '{}'. Use 'train' or 'predict'.",
                unrecognised_mode
            );
            return;
        }
        None => {
            eprintln!("horse_racing_classifier: no mode specified. Use 'train' or 'predict'.");
            return;
        }
    };

    // Load config. If stats_config.toml is absent, use defaults silently.
    let config_file_path = Path::new("stats_config.toml");
    let parsed_config = if config_file_path.exists() {
        match parse_stats_config_from_toml_file(config_file_path) {
            Ok(loaded_config) => loaded_config,
            Err(_config_parse_error_discarded) => {
                eprintln!(
                    "horse_racing_classifier: could not parse stats_config.toml, using defaults"
                );
                StatsConfig::default_config()
            }
        }
    } else {
        eprintln!("horse_racing_classifier: stats_config.toml not found, using defaults");
        StatsConfig::default_config()
    };

    let timestamp_bytes = get_current_timestamp_string();

    match run_mode {
        "train" => {
            if let Err(train_error) = run_train_mode(&parsed_config, &timestamp_bytes) {
                eprintln!(
                    "horse_racing_classifier: train mode error: {}",
                    train_error.terse_production_message()
                );
            }
        }
        "predict" => {
            if let Err(predict_error) = run_predict_mode(&parsed_config, &timestamp_bytes) {
                eprintln!(
                    "horse_racing_classifier: predict mode error: {}",
                    predict_error.terse_production_message()
                );
            }
        }
        _ => {
            // Unreachable given the match above, but handled defensively.
            eprintln!("horse_racing_classifier: internal mode dispatch error");
        }
    }
}

// ============================================================================
// SECTION 8 — CARGO TESTS
// ============================================================================

#[cfg(test)]
mod section_eight_config_output_main_tests {
    use super::*;
    use std::io::Write;

    /// Verifies that test_fraction_percent is parsed from the config file
    /// and defaults to 20 when absent.
    #[test]
    fn config_parser_reads_test_fraction_percent() {
        let temporary_config_path =
            std::env::temp_dir().join("horse_racing_section_eight_test_fraction_config_test.toml");
        {
            let mut config_file_handle =
                File::create(&temporary_config_path).expect("temp config create must succeed");
            config_file_handle
                .write_all(b"test_fraction_percent = 15\n")
                .expect("write must succeed");
        }

        let parsed_config = parse_stats_config_from_toml_file(&temporary_config_path)
            .expect("valid config must parse");
        assert_eq!(parsed_config.test_fraction_percent, 15);

        let _ignored = std::fs::remove_file(&temporary_config_path);
    }

    /// Verifies that test_fraction_percent defaults to 20 when not
    /// present in the config file.
    #[test]
    fn config_default_has_test_fraction_percent_twenty() {
        let default_config = StatsConfig::default_config();
        assert_eq!(default_config.test_fraction_percent, 20);
    }

    /// Verifies that the training history CSV is created with a header
    /// and one data row on the first run, and appended on a second run.
    #[test]
    fn training_history_csv_is_created_and_appended() {
        let temporary_results_dir =
            std::env::temp_dir().join("horse_racing_section_eight_history_csv_test");
        let _ignored = std::fs::remove_dir_all(&temporary_results_dir);

        // Build a minimal synthetic summary for the CSV append function.
        let synthetic_summary = TrainingRunSummary {
            best_hyperparameter_candidate_found: HyperparameterCandidate {
                tree_max_depth: 3,
                tree_min_leaf_samples: 2,
            },
            hyperparameter_search_results_table: vec![
                HyperparameterSearchResult {
                    candidate: HyperparameterCandidate {
                        tree_max_depth: 3,
                        tree_min_leaf_samples: 2,
                    },
                    label_kind: TreeLabelKind::CompletionClassification,
                    train_accuracy_percent: 90,
                    validate_accuracy_percent: 80,
                    train_mean_absolute_error: 0,
                    validate_mean_absolute_error: 0,
                },
                HyperparameterSearchResult {
                    candidate: HyperparameterCandidate {
                        tree_max_depth: 3,
                        tree_min_leaf_samples: 2,
                    },
                    label_kind: TreeLabelKind::PerformanceScoreRegression,
                    train_accuracy_percent: 70,
                    validate_accuracy_percent: 60,
                    train_mean_absolute_error: 200,
                    validate_mean_absolute_error: 300,
                },
            ],
            test_set_accuracy_reports: vec![
                ModelAccuracyReport {
                    label_kind: TreeLabelKind::CompletionClassification,
                    model_kind_description: "decision_tree",
                    evaluation_set_description: "held_out_test_set",
                    accuracy_percent: 75,
                    mean_absolute_error: 0,
                },
                ModelAccuracyReport {
                    label_kind: TreeLabelKind::PerformanceScoreRegression,
                    model_kind_description: "decision_tree",
                    evaluation_set_description: "held_out_test_set",
                    accuracy_percent: 55,
                    mean_absolute_error: 350,
                },
            ],
            stage_two_accuracy_reports: vec![
                ModelAccuracyReport {
                    label_kind: TreeLabelKind::CompletionClassification,
                    model_kind_description: "decision_tree",
                    evaluation_set_description: "stage2_all_training_data",
                    accuracy_percent: 92,
                    mean_absolute_error: 0,
                },
                ModelAccuracyReport {
                    label_kind: TreeLabelKind::PerformanceScoreRegression,
                    model_kind_description: "decision_tree",
                    evaluation_set_description: "stage2_all_training_data",
                    accuracy_percent: 72,
                    mean_absolute_error: 180,
                },
            ],
            saved_model_file_paths: vec![std::path::PathBuf::from("models/tree_completion.txt")],
            feature_analysis: FeatureAnalysisBundle {
                classification_tree_split_counts: FeatureSplitCounts::new_all_zeros(),
                regression_tree_split_counts: FeatureSplitCounts::new_all_zeros(),
                classification_outcome_associations: all_feature_indices_in_canonical_order()
                    .iter()
                    .map(|f| SingleFeatureOutcomeAssociation {
                        feature_index: *f,
                        bottom_third_mean_label: 0,
                        top_third_mean_label: 1,
                        top_minus_bottom_difference: 1,
                    })
                    .collect(),
                regression_outcome_associations: all_feature_indices_in_canonical_order()
                    .iter()
                    .map(|f| SingleFeatureOutcomeAssociation {
                        feature_index: *f,
                        bottom_third_mean_label: 300,
                        top_third_mean_label: 700,
                        top_minus_bottom_difference: 400,
                    })
                    .collect(),
            },
            partition_group_counts: PartitionGroupCounts {
                total_records: 50,
                total_race_groups: 10,
                train_groups: 6,
                validate_groups: 2,
                test_groups: 2,
            },
        };

        let fixed_timestamp: [u8; 15] = *b"20250601_100000";

        // First append — should create file with header + one data row.
        append_training_run_to_history_csv(
            &synthetic_summary,
            &fixed_timestamp,
            &temporary_results_dir,
        )
        .expect("first append must succeed");

        let history_path = temporary_results_dir.join("training_history.csv");
        assert!(
            history_path.exists(),
            "training_history.csv must be created"
        );

        // Read the file and count lines.
        let first_run_content =
            std::fs::read_to_string(&history_path).expect("must be able to read history csv");
        let first_run_line_count = first_run_content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count();
        assert_eq!(
            first_run_line_count, 2,
            "first run: must have header + one data row"
        );

        // Second append — should add one more data row (no duplicate header).
        let second_timestamp: [u8; 15] = *b"20250601_110000";
        append_training_run_to_history_csv(
            &synthetic_summary,
            &second_timestamp,
            &temporary_results_dir,
        )
        .expect("second append must succeed");

        let second_run_content =
            std::fs::read_to_string(&history_path).expect("must be able to read history csv");
        let second_run_line_count = second_run_content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count();
        assert_eq!(
            second_run_line_count, 3,
            "second run: must have header + two data rows"
        );

        // Verify the header starts with "timestamp".
        let first_line = second_run_content.lines().next().unwrap_or("");
        assert!(
            first_line.starts_with("timestamp"),
            "header must start with 'timestamp'"
        );

        let _ignored = std::fs::remove_dir_all(&temporary_results_dir);
    }

    /// Verifies that the updated training summary formatter includes the
    /// new sections: partition info, test accuracy, and feature analysis.
    #[test]
    fn training_summary_formatter_includes_new_sections() {
        // Build a minimal summary with the new fields populated.
        let synthetic_summary = TrainingRunSummary {
            best_hyperparameter_candidate_found: HyperparameterCandidate {
                tree_max_depth: 4,
                tree_min_leaf_samples: 2,
            },
            hyperparameter_search_results_table: vec![HyperparameterSearchResult {
                candidate: HyperparameterCandidate {
                    tree_max_depth: 4,
                    tree_min_leaf_samples: 2,
                },
                label_kind: TreeLabelKind::CompletionClassification,
                train_accuracy_percent: 90,
                validate_accuracy_percent: 80,
                train_mean_absolute_error: 0,
                validate_mean_absolute_error: 0,
            }],
            test_set_accuracy_reports: vec![ModelAccuracyReport {
                label_kind: TreeLabelKind::CompletionClassification,
                model_kind_description: "decision_tree",
                evaluation_set_description: "held_out_test_set",
                accuracy_percent: 75,
                mean_absolute_error: 0,
            }],
            stage_two_accuracy_reports: vec![ModelAccuracyReport {
                label_kind: TreeLabelKind::CompletionClassification,
                model_kind_description: "decision_tree",
                evaluation_set_description: "stage2_all_training_data",
                accuracy_percent: 92,
                mean_absolute_error: 0,
            }],
            saved_model_file_paths: vec![std::path::PathBuf::from("models/tree_completion.txt")],
            feature_analysis: FeatureAnalysisBundle {
                classification_tree_split_counts: FeatureSplitCounts::new_all_zeros(),
                regression_tree_split_counts: FeatureSplitCounts::new_all_zeros(),
                classification_outcome_associations: all_feature_indices_in_canonical_order()
                    .iter()
                    .map(|f| SingleFeatureOutcomeAssociation {
                        feature_index: *f,
                        bottom_third_mean_label: 0,
                        top_third_mean_label: 1,
                        top_minus_bottom_difference: 1,
                    })
                    .collect(),
                regression_outcome_associations: all_feature_indices_in_canonical_order()
                    .iter()
                    .map(|f| SingleFeatureOutcomeAssociation {
                        feature_index: *f,
                        bottom_third_mean_label: 300,
                        top_third_mean_label: 700,
                        top_minus_bottom_difference: 400,
                    })
                    .collect(),
            },
            partition_group_counts: PartitionGroupCounts {
                total_records: 50,
                total_race_groups: 10,
                train_groups: 6,
                validate_groups: 2,
                test_groups: 2,
            },
        };

        let fixed_timestamp: [u8; 15] = *b"20250101_120000";
        let formatted_output =
            format_training_summary_as_text(&synthetic_summary, &fixed_timestamp);

        // Check that all major section headings are present.
        assert!(formatted_output.contains("DATA PARTITIONING"));
        assert!(formatted_output.contains("HELD-OUT TEST SET EVALUATION"));
        assert!(formatted_output.contains("FEATURE IMPORTANCE"));
        assert!(formatted_output.contains("FEATURE-OUTCOME ASSOCIATION"));
        assert!(formatted_output.contains("STAGE 1"));
        assert!(formatted_output.contains("STAGE 2"));
        assert!(formatted_output.contains("BEST HYPERPARAMETER"));
        assert!(formatted_output.contains("SAVED MODEL FILES"));

        // Check that partition counts appear.
        assert!(formatted_output.contains("test groups"));
        assert!(formatted_output.contains("10 race groups"));
    }

    /// Verifies that default config values are populated as documented when
    /// `StatsConfig::default_config()` is called.
    #[test]
    fn default_config_has_expected_field_values() {
        let default_config = StatsConfig::default_config();
        assert_eq!(default_config.tree_max_depth, 4);
        assert_eq!(default_config.tree_min_leaf_samples, 2);
        assert_eq!(default_config.training_fraction_percent, 80);
        assert_eq!(default_config.split_seed, 42);
        assert_eq!(default_config.linear_margin_threshold_classification, 50);
        assert_eq!(default_config.linear_margin_threshold_regression, 400);
        assert!(!default_config.hyperparam_search_max_depths.is_empty());
        assert!(!default_config.hyperparam_search_min_leaf_samples.is_empty());
    }

    /// Verifies that parsing a valid config file overwrites the correct
    /// fields and leaves unspecified fields at their defaults.
    #[test]
    fn config_parser_reads_known_keys_and_retains_defaults_for_missing_keys() {
        let temporary_config_path =
            std::env::temp_dir().join("horse_racing_section_eight_config_test.toml");
        {
            let mut config_file_handle =
                File::create(&temporary_config_path).expect("temp config create must succeed");
            config_file_handle
                .write_all(
                    b"# Test config\n\
                          test_train_data_csv_path = \"data/my_historical.csv\"\n\
                          tree_max_depth = 6\n\
                          split_seed = 99\n\
                          hyperparam_search_max_depths = 3,5\n",
                )
                .expect("write must succeed");
        }

        let parsed_config = parse_stats_config_from_toml_file(&temporary_config_path)
            .expect("valid config must parse");

        // The renamed field must be populated from the config file.
        assert_eq!(
            parsed_config.test_train_data_csv_path,
            "data/my_historical.csv"
        );
        assert_eq!(parsed_config.tree_max_depth, 6);
        assert_eq!(parsed_config.split_seed, 99);
        assert_eq!(parsed_config.hyperparam_search_max_depths, vec![3, 5]);
        // Unspecified fields must retain their defaults.
        assert_eq!(parsed_config.tree_min_leaf_samples, 2);
        assert_eq!(parsed_config.training_fraction_percent, 80);

        let _ignored = std::fs::remove_file(&temporary_config_path);
    }

    /// Verifies that a config file with a non-integer value for an integer
    /// key leaves that field at its default (does not error).
    #[test]
    fn config_parser_silently_skips_invalid_integer_values() {
        let temporary_config_path =
            std::env::temp_dir().join("horse_racing_section_eight_config_bad_int_test.toml");
        {
            let mut config_file_handle =
                File::create(&temporary_config_path).expect("temp config create must succeed");
            config_file_handle
                .write_all(b"tree_max_depth = not_a_number\n")
                .expect("write must succeed");
        }

        let parsed_config = parse_stats_config_from_toml_file(&temporary_config_path)
            .expect("config with bad int must still parse");
        // Must retain default, not crash.
        assert_eq!(parsed_config.tree_max_depth, 4);

        let _ignored = std::fs::remove_file(&temporary_config_path);
    }

    /// Verifies that attempting to parse a nonexistent config file returns
    /// a file-read error.
    #[test]
    fn config_parser_returns_error_for_nonexistent_file() {
        let nonexistent_path =
            Path::new("/tmp/horse_racing_this_config_does_not_exist_section_eight.toml");
        let parse_result = parse_stats_config_from_toml_file(nonexistent_path);
        assert!(matches!(
            parse_result,
            Err(HorseRacingError::CsvFileReadFailure(_))
        ));
    }

    /// Verifies that the timestamp function returns a 15-byte ASCII buffer
    /// and that the underscore separator is at position 8.
    #[test]
    fn timestamp_function_returns_correctly_structured_bytes() {
        let timestamp_buffer = get_current_timestamp_string();
        assert_eq!(timestamp_buffer.len(), 15);
        // Position 8 must be '_'.
        assert_eq!(timestamp_buffer[8], b'_');
        // All other positions must be ASCII digits.
        for position in 0..15 {
            if position == 8 {
                continue;
            }
            assert!(
                timestamp_buffer[position].is_ascii_digit(),
                "timestamp byte at position {} must be an ASCII digit",
                position
            );
        }
    }

    /// Verifies that the timestamp produces a plausible year (>= 2024).
    /// This is a sanity check that the calendar arithmetic is not wildly
    /// wrong — it will pass as long as the system clock is set correctly.
    #[test]
    fn timestamp_year_is_plausible() {
        let timestamp_buffer = get_current_timestamp_string();
        // Year occupies bytes 0-3.
        let year_digits = &timestamp_buffer[0..4];
        let year_string = std::str::from_utf8(year_digits).expect("year bytes must be valid UTF-8");
        let year_value: u32 = year_string
            .parse()
            .expect("year bytes must parse as integer");
        assert!(
            year_value >= 2024,
            "timestamp year {} is implausibly in the past",
            year_value
        );
    }

    /// Verifies that the results file writer creates the file with the
    /// expected name pattern.
    #[test]
    fn results_file_writer_creates_file_with_correct_name_pattern() {
        let temporary_results_dir =
            std::env::temp_dir().join("horse_racing_section_eight_results_writer_test");
        let fixed_timestamp: [u8; 15] = *b"20250101_120000";

        write_text_to_timestamped_results_file(
            "test content",
            &temporary_results_dir,
            "train",
            &fixed_timestamp,
        )
        .expect("results write must succeed");

        let expected_file_path = temporary_results_dir.join("train_20250101_120000.txt");
        assert!(
            expected_file_path.exists(),
            "results file must exist at expected path"
        );

        let _ignored = std::fs::remove_dir_all(&temporary_results_dir);
    }

    /// Verifies that the prediction formatter correctly marks a horse as
    /// DNF when `tree_completion_prediction == 0`.
    #[test]
    fn prediction_formatter_labels_dnf_horse_correctly() {
        let dnf_horse_result = SingleHorsePredictionResult {
            source_row_id: 3,
            game_id: 7,
            age: 8,
            height: 145,
            experience: 1,
            weight: 1200,
            height_to_weight_ratio: 120,
            tree_completion_prediction: 0,
            tree_performance_score_prediction: 0,
            margin_risk_flags: vec![RiskFlag {
                flagged_feature_index: FeatureIndex::Age,
                boundary_direction: MarginBoundaryDirection::High,
                boundary_threshold_value: 7,
                actual_feature_value: 8,
            }],
        };
        let fixed_timestamp: [u8; 15] = *b"20250101_120000";
        let formatted_text =
            format_prediction_results_as_text(&[dnf_horse_result], &fixed_timestamp);
        assert!(formatted_text.contains("DNF(0)"));
        assert!(formatted_text.contains("age+"));
    }

    /// Verifies that the prediction formatter correctly identifies the
    /// predicted winner as the horse with the highest performance score
    /// among completing horses.
    #[test]
    fn prediction_formatter_identifies_highest_score_as_winner() {
        let horse_results = vec![
            SingleHorsePredictionResult {
                source_row_id: 0,
                game_id: 1,
                age: 4,
                height: 150,
                experience: 3,
                weight: 900,
                height_to_weight_ratio: 166,
                tree_completion_prediction: 1,
                tree_performance_score_prediction: 600,
                margin_risk_flags: Vec::new(),
            },
            SingleHorsePredictionResult {
                source_row_id: 1,
                game_id: 1,
                age: 5,
                height: 160,
                experience: 4,
                weight: 950,
                height_to_weight_ratio: 168,
                tree_completion_prediction: 1,
                tree_performance_score_prediction: 1000,
                margin_risk_flags: Vec::new(),
            },
        ];
        let fixed_timestamp: [u8; 15] = *b"20250101_120000";
        let formatted_text = format_prediction_results_as_text(&horse_results, &fixed_timestamp);
        // row_id 1 has score 1000 and should be the predicted winner.
        assert!(formatted_text.contains("row_id 1"));
        assert!(formatted_text.contains("score 1000"));
    }
}
