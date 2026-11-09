"""gPPI 3rd-level group FEAT: analyses and 2nd-level copes pooled at 3rd level."""

ANALYSES = ("PPI_LOC", "PPI_FFA")

# 2nd-level cope index per analysis (1st-level contrasts pooled at 2nd level):
#   cope 3 = ppi_FFA, cope 4 = ppi_LOC
ANALYSIS_COPE: dict[str, int] = {
    "PPI_FFA": 3,
    "PPI_LOC": 4,
}

COPE_LABELS: dict[int, str] = {
    3: "ppi_FFA",
    4: "ppi_LOC",
}
