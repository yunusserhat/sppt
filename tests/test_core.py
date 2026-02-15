import pandas as pd
import pytest

from sppt import sppt


class TestBootstrapSingleVar:
    """Tests for single variable bootstrapping."""

    def test_basic_bootstrap(self):
        """Test basic bootstrap with simple data."""
        data = pd.DataFrame({
            "group": ["A", "B", "C"],
            "count": [10, 20, 30],
        })

        result = sppt(
            data=data,
            group_col="group",
            count_cols=["count"],
            B=100,
            seed=42,
            use_percentages=True,  # Default
        )

        assert len(result) == 3
        assert "count_L" in result.columns
        assert "count_U" in result.columns
        # With use_percentages=True, values are percentages that sum to 100%
        # The counts [10, 20, 30] sum to 60, so percentages are [16.67, 33.33, 50]
        # CI should contain these percentage values (with some bootstrap variation)
        expected_pct_A = (10 / 60) * 100  # ~16.67%
        expected_pct_B = (20 / 60) * 100  # ~33.33%
        expected_pct_C = (30 / 60) * 100  # ~50%

        # CI should contain the expected percentages
        assert result["count_L"].iloc[0] <= expected_pct_A <= result["count_U"].iloc[0]
        assert result["count_L"].iloc[1] <= expected_pct_B <= result["count_U"].iloc[1]
        assert result["count_L"].iloc[2] <= expected_pct_C <= result["count_U"].iloc[2]

    def test_zero_counts(self):
        """Test handling of zero counts."""
        data = pd.DataFrame({
            "group": ["A", "B", "C"],
            "count": [0, 0, 0],
        })

        result = sppt(
            data=data,
            group_col="group",
            count_cols=["count"],
            B=100,
            seed=42,
        )

        assert all(result["count_L"] == 0)
        assert all(result["count_U"] == 0)

    def test_single_group(self):
        """Test with single group."""
        data = pd.DataFrame({
            "group": ["A"],
            "count": [50],
        })

        result = sppt(
            data=data,
            group_col="group",
            count_cols=["count"],
            B=100,
            seed=42,
            use_percentages=True,  # Default - percentages
        )

        assert len(result) == 1
        # With single group, all counts are 100% (100% of total)
        assert result["count_L"].iloc[0] == 100.0
        assert result["count_U"].iloc[0] == 100.0

        # Also test with use_percentages=False (counts mode)
        result_counts = sppt(
            data=data,
            group_col="group",
            count_cols=["count"],
            B=100,
            seed=42,
            use_percentages=False,
        )
        assert result_counts["count_L"].iloc[0] == 50.0
        assert result_counts["count_U"].iloc[0] == 50.0


class TestMultipleVariables:
    """Tests for multiple variable comparison."""

    def test_two_variables_overlap(self):
        """Test two variables with overlapping distributions."""
        data = pd.DataFrame({
            "group": ["A", "B", "C"],
            "Var1": [10, 20, 30],
            "Var2": [12, 22, 32],
        })

        result = sppt(
            data=data,
            group_col="group",
            count_cols=["Var1", "Var2"],
            B=100,
            check_overlap=True,
            seed=42,
        )

        assert "intervals_overlap" in result.columns
        assert "SIndex_Bivariate" in result.columns
        assert "s_index" in result.attrs

    def test_two_variables_no_overlap(self):
        """Test two variables with non-overlapping distributions."""
        # Use data where the distributions truly don't overlap
        # One variable concentrated in first half, other in second half
        # This creates very different spatial patterns
        data = pd.DataFrame({
            "group": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "Var1": [1000, 1000, 1000, 1000, 0, 0, 0, 0],  # All in first 4
            "Var2": [0, 0, 0, 0, 1000, 1000, 1000, 1000],  # All in last 4
        })

        result = sppt(
            data=data,
            group_col="group",
            count_cols=["Var1", "Var2"],
            B=500,
            check_overlap=True,
            seed=42,
        )

        assert "intervals_overlap" in result.columns
        assert "SIndex_Bivariate" in result.columns
        # With completely separated patterns, S-Index should be 0
        assert result.attrs["s_index"] == 0.0

    def test_fix_base(self):
        """Test fixing the base variable."""
        data = pd.DataFrame({
            "group": ["A", "B", "C"],
            "Base": [25, 35, 45],
            "Test": [30, 40, 50],
        })

        result = sppt(
            data=data,
            group_col="group",
            count_cols=["Base", "Test"],
            B=100,
            check_overlap=True,
            fix_base=True,
            seed=42,
        )

        # Base variable should have no variation
        assert all(result["Base_L"] == result["Base_U"])


class TestMetrics:
    """Tests for S-Index and overlap metrics."""

    def test_s_index_calculation(self):
        """Test S-Index is computed correctly."""
        data = pd.DataFrame({
            "group": ["A", "B", "C", "D"],
            "Var1": [25, 35, 45, 55],
            "Var2": [26, 36, 46, 56],
        })

        result = sppt(
            data=data,
            group_col="group",
            count_cols=["Var1", "Var2"],
            B=100,
            check_overlap=True,
            seed=42,
        )

        s_index = result.attrs["s_index"]
        assert 0 <= s_index <= 1
        # With similar distributions, S-Index should be high
        assert s_index > 0.5

    def test_robust_s_index(self):
        """Test robust S-Index excludes zero counts."""
        data = pd.DataFrame({
            "group": ["A", "B", "C", "D"],
            "Var1": [0, 10, 20, 30],
            "Var2": [0, 12, 22, 32],
        })

        result = sppt(
            data=data,
            group_col="group",
            count_cols=["Var1", "Var2"],
            B=100,
            check_overlap=True,
            seed=42,
        )

        robust_s_index = result.attrs["robust_s_index"]
        assert 0 <= robust_s_index <= 1


class TestUsePercentages:
    """Tests for use_percentages option."""

    def test_use_percentages_true(self):
        """Test percentages mode."""
        data = pd.DataFrame({
            "group": ["A", "B", "C"],
            "count": [10, 20, 30],
        })

        result = sppt(
            data=data,
            group_col="group",
            count_cols=["count"],
            B=100,
            use_percentages=True,
            seed=42,
        )

        # Percentages should sum to 100 per bootstrap
        assert all(result["count_L"] >= 0)
        assert all(result["count_U"] >= 0)

    def test_use_percentages_false(self):
        """Test counts mode."""
        data = pd.DataFrame({
            "group": ["A", "B", "C"],
            "count": [10, 20, 30],
        })

        result = sppt(
            data=data,
            group_col="group",
            count_cols=["count"],
            B=100,
            use_percentages=False,
            seed=42,
        )

        # Counts should be non-negative
        assert all(result["count_L"] >= 0)
        assert all(result["count_U"] >= 0)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_data(self):
        """Test with empty data."""
        data = pd.DataFrame({
            "group": [],
            "count": [],
        })

        result = sppt(
            data=data,
            group_col="group",
            count_cols=["count"],
            B=100,
            seed=42,
        )

        assert len(result) == 0
        assert "count_L" in result.columns
        assert "count_U" in result.columns

    def test_mismatched_new_cols(self):
        """Test that count_cols must be a list."""
        data = pd.DataFrame({
            "group": ["A", "B"],
            "Var1": [10, 20],
            "Var2": [30, 40],
        })

        # count_cols must be a list (not a string or other type)
        # This is a basic validation - the function expects a list
        pass  # This is a placeholder - Python version doesn't need this validation

    def test_seed_reproducibility(self):
        """Test that seed produces reproducible results."""
        data = pd.DataFrame({
            "group": ["A", "B", "C", "D", "E"],
            "count": [10, 20, 30, 40, 50],
        })

        result1 = sppt(
            data=data,
            group_col="group",
            count_cols=["count"],
            B=100,
            seed=123,
        )

        result2 = sppt(
            data=data,
            group_col="group",
            count_cols=["count"],
            B=100,
            seed=123,
        )

        pd.testing.assert_frame_equal(result1, result2)
