"""Generate reference outputs from R for parity tests."""

import pandas as pd
import subprocess
import os

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "parity", "fixtures")
EXPECTED_DIR = os.path.join(PROJECT_ROOT, "tests", "parity", "expected")


def generate_reference_outputs():
    """Generate reference outputs using R."""
    r_script = f"""
    library(sppt.aggregated.data)
    library(dplyr)

    setwd("{FIXTURES_DIR}")

    # Basic test
    data <- read.csv("basic_test.csv")
    result <- sppt(
      data = data,
      group_col = "group",
      count_col = c("Base", "Test"),
      B = 200,
      check_overlap = TRUE,
      use_percentages = TRUE,
      seed = 42
    )
    write.csv(as.data.frame(result), "expected_basic_test.csv", row.names = FALSE)
    s_index <- attr(result, "s_index")
    robust_s_index <- attr(result, "robust_s_index")
    cat("Basic test - S-Index:", s_index, "\\n")
    cat("Basic test - Robust S-Index:", robust_s_index, "\\n")

    # With zeros test
    data <- read.csv("with_zeros.csv")
    result <- sppt(
      data = data,
      group_col = "group",
      count_col = c("Base", "Test"),
      B = 200,
      check_overlap = TRUE,
      use_percentages = TRUE,
      seed = 42
    )
    write.csv(as.data.frame(result), "expected_with_zeros.csv", row.names = FALSE)
    s_index <- attr(result, "s_index")
    robust_s_index <- attr(result, "robust_s_index")
    cat("With zeros test - S-Index:", s_index, "\\n")
    cat("With zeros test - Robust S-Index:", robust_s_index, "\\n")
    """

    # Write R script
    r_script_path = os.path.join(SCRIPT_DIR, "generate_ref.R")
    with open(r_script_path, "w") as f:
        f.write(r_script)

    # Run R script
    result = subprocess.run(
        ["Rscript", r_script_path],
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("R error:", result.stderr)

    # Clean up R script
    os.remove(r_script_path)

    # Move output files to expected directory
    os.makedirs(EXPECTED_DIR, exist_ok=True)

    expected_basic = os.path.join(SCRIPT_DIR, "expected_basic_test.csv")
    expected_zeros = os.path.join(SCRIPT_DIR, "expected_with_zeros.csv")

    if os.path.exists(expected_basic):
        os.rename(expected_basic, os.path.join(EXPECTED_DIR, "basic_test.csv"))
        print(f"Created: {os.path.join(EXPECTED_DIR, 'basic_test.csv')}")
    if os.path.exists(expected_zeros):
        os.rename(expected_zeros, os.path.join(EXPECTED_DIR, "with_zeros.csv"))
        print(f"Created: {os.path.join(EXPECTED_DIR, 'with_zeros.csv')}")


if __name__ == "__main__":
    generate_reference_outputs()
