#!/usr/bin/env python3
"""Canonical direct full-versus-partial reference-target analysis.

This entry point always uses the independent GLM/MNAR full-reference replica,
then combines those corrected rows with the immutable non-GLM/MNAR comparison
results. It writes a new output directory and never replaces historical v1 or
v2 analysis artifacts.

The prior generic implementation is preserved in Git history. The former
``analyze_glm_mnar_independent_full_vs_partial.py`` command is a compatibility
alias for this same implementation.
"""
from __future__ import annotations

from analyze_glm_mnar_independent_full_vs_partial import main


if __name__ == "__main__":
    main()
