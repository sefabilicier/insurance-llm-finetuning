"""
Data processing pipeline for insurance dataset.

Handles:
- PII masking (emails, phones, SSNs, credit cards)
- Deduplication (MinHash)
- Quality scoring
- Text normalization
- Format validation
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PIIMasker:
    """Mask personally identifiable information (PII) in text."""

    # Regex patterns for common PII
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.]?)?(?:\(\d{3}\)|\d{3})[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    }

    REPLACEMENTS = {
        "email": "[EMAIL]",
        "phone": "[PHONE]",
        "ssn": "[SSN]",
        "credit_card": "[CREDIT_CARD]",
        "ip_address": "[IP_ADDRESS]",
    }

    def mask(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Mask PII in text.

        Args:
            text: Input text

        Returns:
            Tuple of (masked_text, counts_dict)
        """
        masked_text = text
        counts = defaultdict(int)

        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                replacement = self.REPLACEMENTS[pii_type]
                masked_text = re.sub(pattern, replacement, masked_text)
                counts[pii_type] = len(matches)

        return masked_text, dict(counts)


class TextDeduplicator:
    """Deduplicate texts using MinHash-like approach."""

    def __init__(self, threshold: float = 0.95, ngram_size: int = 2):
        """
        Initialize deduplicator.

        Args:
            threshold: Similarity threshold (0-1, higher = more strict)
            ngram_size: N-gram size for fingerprinting
        """
        self.threshold = threshold
        self.ngram_size = ngram_size

    def _get_ngrams(self, text: str) -> set:
        """Extract n-grams from text."""
        text = text.lower().strip()
        words = text.split()

        if len(words) < self.ngram_size:
            return {text}

        ngrams = set()
        for i in range(len(words) - self.ngram_size + 1):
            ngram = " ".join(words[i:i + self.ngram_size])
            ngrams.add(ngram)

        return ngrams

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def deduplicate(self, texts: List[str]) -> List[int]:
        """
        Identify duplicate texts.

        Args:
            texts: List of text strings

        Returns:
            List of indices to keep (remove duplicates)
        """
        keep_indices = []
        seen_ngrams = []

        logger.info(f"Deduplicating {len(texts)} texts (threshold={self.threshold})...")

        for i, text in enumerate(texts):
            ngrams = self._get_ngrams(text)

            is_duplicate = False
            for seen_ng in seen_ngrams:
                similarity = self._jaccard_similarity(ngrams, seen_ng)
                if similarity >= self.threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep_indices.append(i)
                seen_ngrams.append(ngrams)

        removed = len(texts) - len(keep_indices)
        logger.info(f"Removed {removed} duplicates ({len(keep_indices)} remaining)")

        return keep_indices


class QualityScorer:
    """Score text quality for insurance domain."""

    def __init__(self):
        self.min_length = 20
        self.max_length = 1000

    def score(self, example: Dict[str, str]) -> float:
        """
        Score example quality (0-1).

        Args:
            example: Dict with 'user' and 'assistant' keys

        Returns:
            Quality score (0-1)
        """
        user_text = example.get("user", "")
        assistant_text = example.get("assistant", "")

        score = 1.0

        # Length penalties
        if len(user_text) < self.min_length or len(user_text) > self.max_length:
            score -= 0.3

        if len(assistant_text) < self.min_length or len(assistant_text) > self.max_length:
            score -= 0.3

        # Repetition check
        if self._has_excessive_repetition(user_text) or self._has_excessive_repetition(assistant_text):
            score -= 0.2

        # Language quality
        if not self._looks_like_english(user_text) or not self._looks_like_english(assistant_text):
            score -= 0.1

        # No placeholder check
        if "[PLACEHOLDER]" in user_text or "[PLACEHOLDER]" in assistant_text:
            score -= 0.5

        return max(0.0, score)

    def _has_excessive_repetition(self, text: str, threshold: float = 0.3) -> bool:
        """Check if text has excessive word repetition."""
        if not text:
            return False

        words = text.lower().split()
        if len(words) < 5:
            return False

        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1

        max_count = max(word_counts.values())
        return max_count / len(words) > threshold

    def _looks_like_english(self, text: str) -> bool:
        """Check if text looks like English."""
        if not text:
            return False

        # Check for English letters
        english_letters = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        total_letters = sum(1 for c in text if c.isalpha())

        return total_letters == 0 or english_letters / total_letters > 0.7


class DataProcessor:
    """Main data processing pipeline."""

    def __init__(
        self,
        min_quality_score: float = 0.6,
        dedup_threshold: float = 0.95,
        mask_pii: bool = True,
        verbose: bool = True
    ):
        """
        Initialize processor.

        Args:
            min_quality_score: Minimum quality score to keep (0-1)
            dedup_threshold: Deduplication threshold (0-1)
            mask_pii: Whether to mask PII
            verbose: Enable logging
        """
        self.min_quality_score = min_quality_score
        self.pii_masker = PIIMasker() if mask_pii else None
        self.deduplicator = TextDeduplicator(threshold=dedup_threshold)
        self.quality_scorer = QualityScorer()
        self.verbose = verbose

    def process(
        self,
        examples: List[Dict[str, str]],
        remove_duplicates: bool = True,
        filter_by_quality: bool = True
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Process examples through full pipeline.

        Args:
            examples: List of examples
            remove_duplicates: Remove duplicates
            filter_by_quality: Filter by quality score

        Returns:
            Tuple of (processed_examples, stats)
        """
        stats = {
            "original_count": len(examples),
            "after_pii_masking": 0,
            "after_deduplication": 0,
            "after_quality_filtering": 0,
            "pii_found": defaultdict(int),
        }

        if self.verbose:
            logger.info("=" * 60)
            logger.info("DATA PROCESSING PIPELINE")
            logger.info("=" * 60)

        # Step 1: PII Masking
        if self.pii_masker:
            if self.verbose:
                logger.info("\n[Step 1] PII Masking...")
            examples = self._mask_pii(examples, stats)
        else:
            stats["after_pii_masking"] = len(examples)

        # Step 2: Deduplication
        if remove_duplicates:
            if self.verbose:
                logger.info("\n[Step 2] Deduplication...")
            examples = self._deduplicate(examples, stats)
        else:
            stats["after_deduplication"] = len(examples)

        # Step 3: Quality Filtering
        if filter_by_quality:
            if self.verbose:
                logger.info("\n[Step 3] Quality Filtering...")
            examples = self._filter_quality(examples, stats)
        else:
            stats["after_quality_filtering"] = len(examples)

        if self.verbose:
            logger.info("\n" + "=" * 60)
            logger.info("PROCESSING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Original examples: {stats['original_count']}")
            logger.info(f"After masking: {stats['after_pii_masking']}")
            logger.info(f"After dedup: {stats['after_deduplication']}")
            logger.info(f"After quality filter: {stats['after_quality_filtering']}")
            logger.info(f"Final count: {len(examples)}")
            logger.info(f"Retention rate: {len(examples) / stats['original_count'] * 100:.1f}%")

        return examples, stats

    def _mask_pii(
        self,
        examples: List[Dict[str, str]],
        stats: Dict
    ) -> List[Dict[str, str]]:
        """Mask PII in all examples."""
        masked = []

        for example in examples:
            masked_user, counts_user = self.pii_masker.mask(example["user"])
            masked_assistant, counts_asst = self.pii_masker.mask(example["assistant"])

            for pii_type, count in counts_user.items():
                stats["pii_found"][f"user_{pii_type}"] += count

            for pii_type, count in counts_asst.items():
                stats["pii_found"][f"assistant_{pii_type}"] += count

            masked.append({
                **example,
                "user": masked_user,
                "assistant": masked_assistant,
            })

        stats["after_pii_masking"] = len(masked)

        if self.verbose:
            logger.info(f"  PII found: {dict(stats['pii_found'])}")

        return masked

    def _deduplicate(
        self,
        examples: List[Dict[str, str]],
        stats: Dict
    ) -> List[Dict[str, str]]:
        """Remove duplicate examples."""
        # Combine user + assistant for dedup
        texts = [f"{ex['user']} {ex['assistant']}" for ex in examples]
        keep_indices = self.deduplicator.deduplicate(texts)

        deduplicated = [examples[i] for i in keep_indices]
        stats["after_deduplication"] = len(deduplicated)

        return deduplicated

    def _filter_quality(
        self,
        examples: List[Dict[str, str]],
        stats: Dict
    ) -> List[Dict[str, str]]:
        """Filter examples by quality score."""
        filtered = []
        scores = []

        for example in examples:
            score = self.quality_scorer.score(example)
            scores.append(score)

            if score >= self.min_quality_score:
                filtered.append(example)

        stats["after_quality_filtering"] = len(filtered)

        if self.verbose:
            avg_score = np.mean(scores)
            logger.info(f"  Average quality score: {avg_score:.3f}")
            logger.info(f"  Removed: {len(examples) - len(filtered)} low-quality examples")

        return filtered

    def save_processed(
        self,
        examples: List[Dict[str, str]],
        output_path: Path
    ) -> Path:
        """Save processed examples to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

        if self.verbose:
            logger.info(f"\n✓ Saved {len(examples)} examples to: {output_path}")

        return output_path