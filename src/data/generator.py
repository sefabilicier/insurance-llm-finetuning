"""
Synthetic data generator for insurance support conversations.

Uses Ollama + Llama 3.1 to generate realistic insurance customer support
conversations (user queries + agent responses).

Domain categories:
- Policy inquiry (poliçe bilgisi)
- Claim processing (talep işleme)
- Coverage questions (kapsam soruları)
- Premium/billing (ödeme bilgileri)
- Policy modifications (değişiklikler ve iptal)
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaGenerator:
    """Generate synthetic insurance data using Ollama + Llama 3.1."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama2",
        timeout: int = 120,
        max_retries: int = 3,
        verbose: bool = True
    ):
        """
        Initialize Ollama generator.

        Args:
            ollama_url: Ollama API endpoint
            model_name: Model name (e.g., "llama2", "llama3.1")
            timeout: Request timeout in seconds
            max_retries: Max retries on failure
            verbose: Enable logging
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose

        # Verify Ollama is running
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify connection to Ollama server."""
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            
            if self.verbose:
                logger.info(f"✓ Connected to Ollama at {self.ollama_url}")
                logger.info(f"✓ Using model: {self.model_name}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_url}. "
                f"Make sure Ollama is running: ollama serve\n{e}"
            )

    def generate_example(
        self,
        category: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Optional[Dict[str, str]]:
        """
        Generate a single user-agent conversation pair.

        Args:
            category: Domain category (policy_inquiry, claim_processing, etc.)
            temperature: Generation temperature (0-1, higher = more random)
            max_tokens: Max tokens in response

        Returns:
            Dictionary with 'user' and 'assistant' keys, or None on failure
        """
        prompt = self._build_prompt(category)

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "stream": False,
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()
                generated_text = result.get("response", "").strip()

                if generated_text:
                    parsed = self._parse_response(generated_text, category)
                    if parsed:
                        return parsed

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.warning(f"Error in generation (attempt {attempt + 1}): {e}")
                time.sleep(1)

        logger.error(f"Failed to generate example for {category} after {self.max_retries} attempts")
        return None

    def _build_prompt(self, category: str) -> str:
        """Build prompt for specific insurance category."""
        category_prompts = {
            "policy_inquiry": """You are a helpful insurance agent. Generate a realistic customer support conversation between a customer and an insurance agent in Turkish insurance company context.

Category: Policy Inquiry (asking about policy details)

Format the response EXACTLY as:
USER: [Customer question about their policy details, coverage, deductible, or terms]
AGENT: [Professional response from insurance agent explaining the policy details clearly and helpfully]

Important:
- Customer should ask ONE specific question about their insurance policy
- Agent response should be accurate, professional, and within 2-3 sentences
- Use realistic policy details (deductibles, coverage limits, etc.)
- Write in English but mention Turkish insurance concepts

Now generate one conversation:""",

            "claim_processing": """You are a helpful insurance agent. Generate a realistic customer support conversation about claim processing in Turkish insurance context.

Category: Claim Processing (how to file and process claims)

Format the response EXACTLY as:
USER: [Customer asks about how to file a claim or inquire about their claim status]
AGENT: [Agent explains claim process step-by-step or provides claim information]

Important:
- Customer should ask about filing a claim or checking claim status
- Agent should provide clear, step-by-step guidance
- Include realistic claim process information
- Keep response concise (2-3 sentences max)

Now generate one conversation:""",

            "coverage_questions": """You are a helpful insurance agent. Generate a realistic customer support conversation about insurance coverage.

Category: Coverage Questions (what does insurance cover)

Format the response EXACTLY as:
USER: [Customer asks what is covered under their policy or if specific situation is covered]
AGENT: [Agent clearly explains what is and isn't covered, with specific examples]

Important:
- Customer should ask about specific coverage scenarios
- Agent should give clear yes/no with explanation
- Use realistic coverage examples
- Keep to 2-3 sentences

Now generate one conversation:""",

            "premium_billing": """You are a helpful insurance agent. Generate a realistic customer support conversation about insurance premiums and billing.

Category: Premium/Billing (payment, renewal, billing questions)

Format the response EXACTLY as:
USER: [Customer asks about premium amount, payment methods, renewal, or billing]
AGENT: [Agent provides clear information about billing and payment options]

Important:
- Customer should ask about costs, payment options, or renewal
- Agent should provide helpful billing information
- Include realistic premium/payment details
- Keep response concise (2-3 sentences)

Now generate one conversation:""",

            "policy_modifications": """You are a helpful insurance agent. Generate a realistic customer support conversation about modifying or canceling policies.

Category: Policy Modifications (changes, cancellations, updates)

Format the response EXACTLY as:
USER: [Customer asks about modifying coverage, updating information, or canceling policy]
AGENT: [Agent explains what changes are possible and how to make them]

Important:
- Customer should ask about making changes to their policy
- Agent should explain modification process clearly
- Be realistic about policy change options
- Keep to 2-3 sentences

Now generate one conversation:"""
        }

        return category_prompts.get(category, category_prompts["policy_inquiry"])

    def _parse_response(self, text: str, category: str) -> Optional[Dict[str, str]]:
        """
        Parse generated text into user/assistant pair.

        Args:
            text: Generated text
            category: Category for validation

        Returns:
            Dict with 'user' and 'assistant' keys, or None if parsing fails
        """
        try:
            # Split by USER and AGENT markers
            if "USER:" in text and "AGENT:" in text:
                user_part = text.split("USER:")[1].split("AGENT:")[0].strip()
                agent_part = text.split("AGENT:")[1].strip()

                # Validation
                if len(user_part) < 10 or len(agent_part) < 10:
                    return None
                
                if len(user_part) > 500 or len(agent_part) > 500:
                    return None

                return {
                    "category": category,
                    "user": user_part,
                    "assistant": agent_part,
                }
        except Exception as e:
            logger.debug(f"Parse error: {e}")

        return None

    def generate_batch(
        self,
        category: str,
        count: int,
        temperature: float = 0.7,
        delay: float = 0.5
    ) -> List[Dict[str, str]]:
        """
        Generate multiple examples for a category.

        Args:
            category: Domain category
            count: Number of examples to generate
            temperature: Generation temperature
            delay: Delay between requests (seconds)

        Returns:
            List of generated examples
        """
        examples = []
        failed = 0

        logger.info(f"Generating {count} examples for '{category}'...")

        for i in range(count):
            example = self.generate_example(category, temperature=temperature)

            if example:
                examples.append(example)
                logger.info(f"  [{i+1}/{count}] ✓ Generated")
            else:
                failed += 1
                logger.info(f"  [{i+1}/{count}] ✗ Failed")

            if i < count - 1:
                time.sleep(delay)

        logger.info(f"Completed: {len(examples)} succeeded, {failed} failed")
        return examples

    def generate_all_categories(
        self,
        total_examples: int = 1000,
        balanced: bool = True,
        temperature: float = 0.7,
        delay: float = 0.5
    ) -> List[Dict[str, str]]:
        """
        Generate examples across all insurance categories.

        Args:
            total_examples: Total number of examples
            balanced: If True, distribute evenly across categories
            temperature: Generation temperature
            delay: Delay between requests

        Returns:
            List of all generated examples
        """
        categories = [
            "policy_inquiry",
            "claim_processing",
            "coverage_questions",
            "premium_billing",
            "policy_modifications"
        ]

        all_examples = []

        if balanced:
            examples_per_category = total_examples // len(categories)
            logger.info(
                f"Generating {examples_per_category} examples per category "
                f"({len(categories)} categories = {examples_per_category * len(categories)} total)"
            )

            for category in categories:
                examples = self.generate_batch(
                    category,
                    examples_per_category,
                    temperature=temperature,
                    delay=delay
                )
                all_examples.extend(examples)
                logger.info(f"Category '{category}' completed: {len(examples)} examples")
        else:
            # Random distribution
            remaining = total_examples
            for category in random.choices(categories, k=total_examples):
                example = self.generate_example(category, temperature=temperature)
                if example:
                    all_examples.append(example)
                    remaining -= 1

        logger.info(f"\n✓ Generation complete: {len(all_examples)} total examples")
        return all_examples


def generate_synthetic_dataset(
    output_path: Path,
    num_examples: int = 1000,
    model_name: str = "llama2",
    ollama_url: str = "http://localhost:11434"
) -> Path:
    """
    Main function to generate synthetic insurance dataset.

    Args:
        output_path: Where to save the generated dataset
        num_examples: Total number of examples to generate
        model_name: Ollama model name
        ollama_url: Ollama server URL

    Returns:
        Path to saved dataset
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = OllamaGenerator(
        ollama_url=ollama_url,
        model_name=model_name,
        verbose=True
    )

    # Generate balanced dataset
    examples = generator.generate_all_categories(
        total_examples=num_examples,
        balanced=True,
        temperature=0.7,
        delay=0.5
    )

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Dataset saved to: {output_path}")
    logger.info(f"  - Total examples: {len(examples)}")
    logger.info(f"  - File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    output_path = Path("./data/raw/synthetic_insurance_data.json")
    num_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    try:
        generate_synthetic_dataset(output_path, num_examples=num_examples)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)