"""
Quality scoring module for data quality assessment.

Note: QualityScorer is implemented in processor.py for now.
This file is kept for future expansion and organizational purposes.
"""

# Import from processor for backward compatibility
from .processor import QualityScorer

__all__ = ["QualityScorer"]