"""
Hybrid synthetic data generator for insurance support conversations.

Strategy:
1. Template-Based Generation (~80%) — Instant, deterministic, high quality
2. Ollama LLM Enrichment (~20%) — Natural variation and edge cases

Domain categories:
- Policy inquiry (poliçe bilgisi)
- Claim processing (talep işleme)
- Coverage questions (kapsam soruları)
- Premium/billing (ödeme bilgileri)
- Policy modifications (değişiklikler ve iptal)

References:
- Google Cloud AI: Best practices for training data quality
- AWS ML: Synthetic data generation patterns
- IBM Think: Domain-specific fine-tuning strategies
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


# ============================================================
# TEMPLATE POOL — Rich, domain-realistic variations
# ============================================================

# Randomization pools for template filling
CUSTOMER_NAMES = [
    "Ahmet Yilmaz", "Fatma Demir", "Mehmet Kaya", "Ayse Celik",
    "Mustafa Sahin", "Zeynep Ozturk", "Ali Arslan", "Emine Tas",
    "Hasan Dogan", "Hatice Koc", "Ibrahim Polat", "Merve Yildiz",
    "Omer Gunes", "Selin Aktas", "Burak Erdem", "Elif Tuncer",
    "Kemal Korkmaz", "Deniz Aydin", "Can Ozkan", "Derya Aksoy",
]

POLICY_NUMBERS = [
    "POL-2024-{:06d}".format(i) for i in range(100000, 100200)
]

CLAIM_NUMBERS = [
    "CLM-2024-{:06d}".format(i) for i in range(500000, 500200)
]

POLICY_TYPES = [
    "auto insurance", "health insurance", "home insurance",
    "life insurance", "travel insurance", "business insurance",
    "pet insurance", "motorcycle insurance", "renters insurance",
    "umbrella insurance",
]

COVERAGE_AMOUNTS = [
    "$25,000", "$50,000", "$100,000", "$150,000", "$200,000",
    "$250,000", "$500,000", "$750,000", "$1,000,000",
]

DEDUCTIBLE_AMOUNTS = [
    "$250", "$500", "$750", "$1,000", "$1,500", "$2,000", "$2,500",
]

PREMIUM_AMOUNTS = [
    "$75/month", "$120/month", "$150/month", "$200/month",
    "$250/month", "$300/month", "$89/month", "$175/month",
    "$450/quarter", "$900/semi-annual", "$1,800/year", "$2,400/year",
]

VEHICLE_TYPES = [
    "2022 Toyota Corolla", "2023 Honda Civic", "2021 BMW 3 Series",
    "2024 Mercedes C-Class", "2020 Ford Focus", "2023 Volkswagen Golf",
    "2022 Hyundai Tucson", "2021 Audi A4", "2023 Renault Clio",
    "2024 Fiat Egea", "2022 Opel Astra", "2023 Peugeot 308",
]

ACCIDENT_TYPES = [
    "rear-end collision", "side-impact accident", "parking lot damage",
    "hit-and-run incident", "weather-related damage", "theft",
    "vandalism", "windshield crack", "fender bender", "rollover",
    "multi-vehicle accident", "single-car accident",
]

HEALTH_CONDITIONS = [
    "routine check-up", "dental procedure", "eye exam",
    "physical therapy session", "specialist consultation",
    "emergency room visit", "surgical procedure", "lab work",
    "prescription medication", "mental health counseling",
]

PROPERTY_TYPES = [
    "single-family home", "apartment", "condominium",
    "townhouse", "duplex", "rental property",
]

PROPERTY_DAMAGE_TYPES = [
    "water damage from burst pipe", "fire damage in kitchen",
    "storm damage to roof", "flooding in basement",
    "electrical damage from lightning", "tree fell on garage",
    "broken window from hail", "smoke damage",
    "earthquake cracks in foundation", "mold damage",
]

PAYMENT_METHODS = [
    "credit card", "bank transfer", "direct debit",
    "online payment portal", "mobile app", "check",
]

MODIFICATION_TYPES = [
    "add a new driver", "remove a vehicle", "add comprehensive coverage",
    "increase liability limits", "add roadside assistance",
    "change my deductible", "update my address", "add a beneficiary",
    "switch to annual payment", "add umbrella coverage",
    "remove collision coverage", "update my vehicle information",
]

AGENT_GREETINGS = [
    "Thank you for contacting us.",
    "I'd be happy to help you with that.",
    "Of course, I can assist you with this.",
    "Great question, let me help you with that.",
    "I understand your concern.",
    "Thank you for reaching out to us.",
    "I appreciate you contacting our support team.",
    "Certainly, I can look into that for you.",
]

AGENT_CLOSINGS = [
    "Is there anything else I can help you with today?",
    "Please don't hesitate to contact us if you have any other questions.",
    "Feel free to reach out if you need further assistance.",
    "We're here to help if you need anything else.",
    "Let me know if you have any other concerns.",
    "Would you like me to help you with anything else?",
]

URGENCY_LEVELS = ["routine", "standard", "urgent", "emergency"]

TIMEFRAMES = [
    "within 24 hours", "within 3-5 business days", "within 7-10 business days",
    "within 2 weeks", "immediately", "within 48 hours",
]


# ============================================================
# TEMPLATE DEFINITIONS
# ============================================================

TEMPLATES = {
    "policy_inquiry": [
        # --- Template 1: General policy details ---
        {
            "user": "I'd like to know the details of my {policy_type} policy. My policy number is {policy_number}.",
            "assistant": "{greeting} Your {policy_type} policy ({policy_number}) provides coverage up to {coverage_amount} with a {deductible} deductible. Your current premium is {premium}. {closing}",
        },
        # --- Template 2: Coverage limits ---
        {
            "user": "What are the coverage limits on my {policy_type} policy ({policy_number})?",
            "assistant": "{greeting} Your policy {policy_number} has a maximum coverage limit of {coverage_amount}. This includes liability coverage and comprehensive protection as outlined in your policy documents. Your deductible is set at {deductible}. {closing}",
        },
        # --- Template 3: Policy status ---
        {
            "user": "Can you tell me if my {policy_type} policy is still active? Policy number {policy_number}.",
            "assistant": "{greeting} I can confirm that your {policy_type} policy ({policy_number}) is currently active and in good standing. Your next renewal date is coming up, and your current premium is {premium}. {closing}",
        },
        # --- Template 4: Deductible inquiry ---
        {
            "user": "What is my deductible for my {policy_type} policy?",
            "assistant": "{greeting} Your {policy_type} deductible is currently set at {deductible}. This is the amount you would need to pay out-of-pocket before your insurance coverage kicks in for any covered claim. {closing}",
        },
        # --- Template 5: Policy renewal ---
        {
            "user": "When does my {policy_type} policy ({policy_number}) come up for renewal?",
            "assistant": "{greeting} Your {policy_type} policy ({policy_number}) is scheduled for renewal in the next billing cycle. Your current premium of {premium} will remain the same unless there are changes to your coverage or risk profile. We'll send you a renewal notice {timeframe}. {closing}",
        },
        # --- Template 6: What's included ---
        {
            "user": "Can you explain what's included in my {policy_type} package?",
            "assistant": "{greeting} Your {policy_type} package includes coverage up to {coverage_amount}, with a {deductible} deductible per incident. The policy also covers third-party liability and provides additional benefits such as emergency assistance. Your monthly premium is {premium}. {closing}",
        },
        # --- Template 7: Co-pay inquiry ---
        {
            "user": "I'm not sure what my co-pay is for my {policy_type}. Can you check?",
            "assistant": "{greeting} Based on your {policy_type} plan, your standard co-pay is {deductible} per visit or service. For specialist visits, the co-pay may differ. I recommend checking your policy summary for the complete schedule of co-pays. {closing}",
        },
        # --- Template 8: Named insured ---
        {
            "user": "Who is listed as the primary insured on policy {policy_number}?",
            "assistant": "{greeting} The primary insured on policy {policy_number} is {customer_name}. If you need to add or change the named insured, I can guide you through the process. Please have valid identification ready for any changes. {closing}",
        },
        # --- Template 9: Waiting period ---
        {
            "user": "Is there a waiting period on my new {policy_type} policy?",
            "assistant": "{greeting} Yes, most {policy_type} policies have a standard waiting period of 30 days for pre-existing conditions. During this period, certain claims may not be covered. After the waiting period, your full coverage of {coverage_amount} becomes effective. {closing}",
        },
        # --- Template 10: Discounts ---
        {
            "user": "Are there any discounts available on my {policy_type} policy ({policy_number})?",
            "assistant": "{greeting} Based on your profile, you may be eligible for multi-policy, loyalty, and claims-free discounts. Currently your premium is {premium}, but bundling policies could save you up to 15%. I can run a discount analysis for you if you'd like. {closing}",
        },
    ],

    "claim_processing": [
        # --- Template 1: How to file ---
        {
            "user": "I need to file a claim for a {accident_type} involving my {vehicle}. How do I start?",
            "assistant": "{greeting} I'm sorry to hear about the incident. To file a claim for the {accident_type}, please follow these steps: 1) Gather all relevant documentation including photos, police reports, and witness information. 2) Submit your claim through our online portal or call our claims hotline. 3) A claims adjuster will be assigned to your case {timeframe}. Your claim number will be provided upon submission. {closing}",
        },
        # --- Template 2: Claim status ---
        {
            "user": "Can you check the status of my claim? The claim number is {claim_number}.",
            "assistant": "{greeting} Your claim {claim_number} is currently being reviewed by our claims department. The adjuster assigned to your case is processing the documentation. You can expect an update {timeframe}. If you have any additional documentation to submit, you can upload it through our online portal. {closing}",
        },
        # --- Template 3: Required documents ---
        {
            "user": "What documents do I need to file a {policy_type} claim for {accident_type}?",
            "assistant": "{greeting} For a {policy_type} claim related to {accident_type}, you'll need: 1) A completed claim form. 2) Photos or videos of the damage. 3) A police report (if applicable). 4) Repair estimates from certified providers. 5) Your policy number ({policy_number}). Please submit all documents through our portal or email them to claims@insurance.com. {closing}",
        },
        # --- Template 4: Claim timeline ---
        {
            "user": "How long will it take to process my claim ({claim_number}) for {accident_type}?",
            "assistant": "{greeting} Most {policy_type} claims are processed within 10-15 business days after all documentation is received. Your claim {claim_number} for {accident_type} is currently in the review stage. The timeline depends on the complexity of the case and any required inspections. We'll keep you updated at each step. {closing}",
        },
        # --- Template 5: Claim denied ---
        {
            "user": "My claim {claim_number} was denied. Can you explain why?",
            "assistant": "{greeting} I understand your concern about claim {claim_number}. Claims can be denied for several reasons, including: the incident not being covered under your policy, missing documentation, or the claim exceeding your coverage limits. I recommend reviewing the denial letter for specific reasons. You have the right to appeal the decision within 30 days. {closing}",
        },
        # --- Template 6: Emergency claim ---
        {
            "user": "I've just been in a {accident_type} and need to file an emergency claim right away.",
            "assistant": "{greeting} I'm sorry to hear about the {accident_type}. For emergency claims, please first ensure everyone's safety and contact emergency services if needed. To expedite your claim: 1) Take photos of the damage and scene. 2) Note the other party's information if applicable. 3) Call our 24/7 claims hotline immediately. We'll assign a priority adjuster to your case within 2 hours. {closing}",
        },
        # --- Template 7: Health claim ---
        {
            "user": "I need to submit a claim for a {health_condition}. What's the process?",
            "assistant": "{greeting} For your {health_condition} claim under your {policy_type} policy, please submit the following: 1) The itemized bill from your healthcare provider. 2) Your Explanation of Benefits (EOB) if available. 3) Any referral or pre-authorization documentation. You can submit these through our online portal. Most health claims are processed {timeframe}. {closing}",
        },
        # --- Template 8: Claim payout ---
        {
            "user": "When will I receive the payout for my approved claim {claim_number}?",
            "assistant": "{greeting} Congratulations on your claim {claim_number} being approved! The payout will be processed {timeframe}. The payment will be sent via {payment_method} to the account on file. The approved amount reflects your coverage minus your {deductible} deductible. {closing}",
        },
        # --- Template 9: Property claim ---
        {
            "user": "There's been {property_damage} at my {property_type}. How do I file a claim?",
            "assistant": "{greeting} I'm sorry about the {property_damage} at your {property_type}. To file your {policy_type} claim: 1) Document the damage with photos and videos immediately. 2) Take reasonable steps to prevent further damage (e.g., tarp a damaged roof). 3) Do not dispose of damaged items until the adjuster has inspected them. 4) Submit your claim through our portal with your policy number {policy_number}. An adjuster will contact you {timeframe}. {closing}",
        },
        # --- Template 10: Supplemental claim ---
        {
            "user": "I need to add additional damage to my existing claim {claim_number}.",
            "assistant": "{greeting} You can submit supplemental information for claim {claim_number} by uploading additional photos, repair estimates, or documentation through our online portal. Reference your existing claim number when submitting. The adjuster will review the additional damage and update your claim accordingly. This may extend the processing timeline by a few business days. {closing}",
        },
    ],

    "coverage_questions": [
        # --- Template 1: Specific scenario ---
        {
            "user": "Does my {policy_type} cover {accident_type}?",
            "assistant": "{greeting} Yes, your {policy_type} policy does cover {accident_type} under your comprehensive coverage. Your policy provides protection up to {coverage_amount} for such incidents, subject to your {deductible} deductible. Please file a claim as soon as possible after the incident occurs. {closing}",
        },
        # --- Template 2: Not covered ---
        {
            "user": "I'm wondering if my {policy_type} covers cosmetic damage to my {vehicle}.",
            "assistant": "{greeting} Unfortunately, standard {policy_type} policies typically do not cover purely cosmetic damage unless it resulted from a covered incident such as an accident or vandalism. If the cosmetic damage was caused by a covered event, you may file a claim. Your comprehensive coverage limit is {coverage_amount}. {closing}",
        },
        # --- Template 3: Travel coverage ---
        {
            "user": "Am I covered if I travel internationally with my {policy_type}?",
            "assistant": "{greeting} Your {policy_type} policy provides limited international coverage depending on your plan level. Standard plans typically cover emergency medical expenses up to {coverage_amount} while traveling. I recommend reviewing your policy documents or purchasing a supplemental travel insurance plan for comprehensive international coverage. {closing}",
        },
        # --- Template 4: Natural disaster ---
        {
            "user": "Does my {policy_type} cover damage from natural disasters like earthquakes or floods?",
            "assistant": "{greeting} Standard {policy_type} policies typically cover storm and wind damage, but earthquake and flood damage usually require separate endorsements or policies. Your current coverage is {coverage_amount} for standard perils. I can provide you with a quote for earthquake or flood coverage if you're interested. {closing}",
        },
        # --- Template 5: Rental car ---
        {
            "user": "If my {vehicle} is in the shop after an accident, does my policy cover a rental car?",
            "assistant": "{greeting} Your {policy_type} policy includes rental car reimbursement coverage if you have the optional rental endorsement. This typically covers up to $50/day for a maximum of 30 days while your {vehicle} is being repaired due to a covered claim. Please check your policy details or I can verify your specific coverage. {closing}",
        },
        # --- Template 6: Liability ---
        {
            "user": "What does the liability portion of my {policy_type} actually cover?",
            "assistant": "{greeting} The liability portion of your {policy_type} covers damages and injuries you may cause to others. This includes bodily injury liability (up to {coverage_amount} per person) and property damage liability. It does not cover damage to your own vehicle or injuries to yourself — that's covered under collision and medical payments coverage. {closing}",
        },
        # --- Template 7: Home office ---
        {
            "user": "I work from home. Does my {policy_type} cover my home office equipment?",
            "assistant": "{greeting} Standard {policy_type} policies provide limited coverage for home office equipment, usually up to $2,500 for business property. If you have high-value equipment or extensive home office setup, I recommend adding a home business endorsement to your policy. Your current coverage limit is {coverage_amount}. {closing}",
        },
        # --- Template 8: Pre-existing condition ---
        {
            "user": "Does my {policy_type} cover pre-existing conditions like diabetes?",
            "assistant": "{greeting} Coverage for pre-existing conditions depends on your specific {policy_type} plan and when you enrolled. Under current regulations, most health plans cannot deny coverage for pre-existing conditions, but there may be waiting periods. Your policy provides coverage up to {coverage_amount}. I recommend reviewing the specific terms in your policy document. {closing}",
        },
        # --- Template 9: Uninsured motorist ---
        {
            "user": "What happens if I'm hit by an uninsured driver? Does my {policy_type} cover that?",
            "assistant": "{greeting} Yes, if you have uninsured/underinsured motorist (UM/UIM) coverage on your {policy_type} policy, you're protected. This covers your medical expenses and vehicle damage up to {coverage_amount} when the at-fault driver has no insurance or insufficient coverage. Your {deductible} deductible applies. {closing}",
        },
        # --- Template 10: Pet coverage ---
        {
            "user": "I'm considering getting a pet. Does my {policy_type} cover pet-related liability?",
            "assistant": "{greeting} Your {policy_type} policy may include limited pet liability coverage, which protects you if your pet causes injury to others or damages their property. However, certain dog breeds may be excluded. Coverage is typically up to {coverage_amount}. For comprehensive pet health coverage, I'd recommend a separate pet insurance policy. {closing}",
        },
    ],

    "premium_billing": [
        # --- Template 1: Premium amount ---
        {
            "user": "How much is my current premium for my {policy_type} policy ({policy_number})?",
            "assistant": "{greeting} Your current {policy_type} premium is {premium}. This rate is based on your current coverage level of {coverage_amount} with a {deductible} deductible. Your next payment is due on the 1st of next month. {closing}",
        },
        # --- Template 2: Payment methods ---
        {
            "user": "What payment methods do you accept for my insurance premium?",
            "assistant": "{greeting} We accept several payment methods including credit card, bank transfer, direct debit, online payment through our portal, and our mobile app. You can set up automatic payments to ensure you never miss a due date. Your current premium is {premium}. {closing}",
        },
        # --- Template 3: Late payment ---
        {
            "user": "I missed my premium payment for policy {policy_number}. What happens now?",
            "assistant": "{greeting} I understand that payments can sometimes be missed. Your policy {policy_number} has a 30-day grace period for late payments. During this period, your coverage remains active. To avoid any lapse in coverage, please make your payment of {premium} as soon as possible through our online portal or by calling our billing department. {closing}",
        },
        # --- Template 4: Premium increase ---
        {
            "user": "Why did my {policy_type} premium increase this year?",
            "assistant": "{greeting} Premium adjustments can occur for several reasons including: changes in risk assessment, claims history, inflation adjustments, regulatory changes, and changes in coverage area risk profiles. Your premium changed from the previous rate to {premium}. I can review your policy to identify any available discounts or coverage adjustments to help reduce your premium. {closing}",
        },
        # --- Template 5: Switch payment frequency ---
        {
            "user": "Can I switch from monthly to annual payment for my {policy_type}?",
            "assistant": "{greeting} Yes, you can switch your payment frequency at any time. Switching to annual payments often comes with a discount of 5-10%. Your current monthly premium is {premium}. I can calculate the annual rate for you and process the switch if you'd like to proceed. The change will take effect on your next billing cycle. {closing}",
        },
        # --- Template 6: Auto-pay setup ---
        {
            "user": "How do I set up automatic payments for my {policy_type} premium?",
            "assistant": "{greeting} You can set up automatic payments through our online portal or mobile app. Simply log in, navigate to 'Billing & Payments,' and select 'Set Up Auto-Pay.' You'll need your {payment_method} information. Auto-pay ensures your premium of {premium} is paid on time each month, and you'll receive a confirmation email for each payment. {closing}",
        },
        # --- Template 7: Refund request ---
        {
            "user": "I overpaid on my {policy_type} policy. Can I get a refund?",
            "assistant": "{greeting} If you've made an overpayment on your policy, we can absolutely process a refund. I can see your policy {policy_number} and will check for any overpayment. Refunds are typically processed {timeframe} and returned to your original payment method. Alternatively, the overpayment can be applied as a credit toward your next premium. {closing}",
        },
        # --- Template 8: Bundle discount ---
        {
            "user": "Can I save money by bundling my {policy_type} with other policies?",
            "assistant": "{greeting} Absolutely! We offer multi-policy discounts of up to 20% when you bundle two or more policies. For example, combining your {policy_type} with a home or life insurance policy can significantly reduce your overall premium from {premium}. I'd be happy to provide a bundled quote for you. {closing}",
        },
        # --- Template 9: Payment confirmation ---
        {
            "user": "I made a payment of {premium} yesterday but haven't received confirmation.",
            "assistant": "{greeting} Payment confirmations are typically sent within 24 hours of processing. If you paid via {payment_method}, it may take 1-2 business days to reflect on your account. I can verify the payment status for your policy {policy_number} right now. Please check your email (including spam folder) for the confirmation receipt. {closing}",
        },
        # --- Template 10: Hardship ---
        {
            "user": "I'm having financial difficulty and can't afford my {policy_type} premium right now.",
            "assistant": "{greeting} I understand financial difficulties can be stressful. We have several options to help: 1) Payment plans to spread your premium over smaller installments. 2) Coverage adjustments to reduce your premium. 3) A temporary payment extension of up to 60 days. 4) Review of available discounts. Your current premium is {premium}, and I'd like to find a solution that keeps you covered. {closing}",
        },
    ],

    "policy_modifications": [
        # --- Template 1: Add coverage ---
        {
            "user": "I'd like to {modification} to my {policy_type} policy ({policy_number}).",
            "assistant": "{greeting} I can help you {modification} on your {policy_type} policy ({policy_number}). This change may affect your premium, which is currently {premium}. I'll provide you with an updated quote reflecting the modification. The change can take effect immediately or on your next billing cycle. Please confirm if you'd like to proceed. {closing}",
        },
        # --- Template 2: Cancel policy ---
        {
            "user": "I want to cancel my {policy_type} policy ({policy_number}). What's the process?",
            "assistant": "{greeting} I'm sorry to hear you'd like to cancel your {policy_type} policy. Before we proceed, I want to make sure you understand: 1) Any prepaid premium will be refunded on a pro-rata basis. 2) There may be a cancellation fee depending on your policy terms. 3) You'll need to have replacement coverage in place to avoid gaps. Would you like to discuss any concerns before we process the cancellation? {closing}",
        },
        # --- Template 3: Change address ---
        {
            "user": "I recently moved and need to update my address on my {policy_type} policy.",
            "assistant": "{greeting} I can update your address on policy {policy_number} right away. Please provide your new address, and I'll process the change. Note that your premium of {premium} may be adjusted based on your new location's risk profile. The update will take effect {timeframe}. {closing}",
        },
        # --- Template 4: Add driver ---
        {
            "user": "I need to add my spouse to my {policy_type} policy as an additional driver.",
            "assistant": "{greeting} Adding a driver to your {policy_type} policy ({policy_number}) is straightforward. I'll need the following information for the new driver: 1) Full legal name and date of birth. 2) Driver's license number. 3) Driving history (accidents or violations in the last 5 years). Adding a driver may adjust your premium from {premium}. I'll provide an updated quote once I have their details. {closing}",
        },
        # --- Template 5: Update vehicle ---
        {
            "user": "I just bought a {vehicle} and need to add it to my auto insurance policy.",
            "assistant": "{greeting} Congratulations on your new {vehicle}! To add it to your policy ({policy_number}), I'll need: 1) The Vehicle Identification Number (VIN). 2) Current mileage. 3) How the vehicle will be used (commute, pleasure, business). 4) Where it will be parked overnight. Your premium will be adjusted based on the vehicle's value and safety ratings. I'll provide an updated quote shortly. {closing}",
        },
        # --- Template 6: Increase coverage ---
        {
            "user": "I'd like to increase my coverage limit on my {policy_type} policy from {coverage_amount} to a higher amount.",
            "assistant": "{greeting} I can help you increase your coverage limit on policy {policy_number}. Currently you're covered for {coverage_amount} with a {deductible} deductible. Increasing your coverage will provide better protection but will adjust your premium from {premium}. I'll prepare a quote with several coverage options for you to review. {closing}",
        },
        # --- Template 7: Remove coverage ---
        {
            "user": "Can I remove the comprehensive coverage from my {policy_type} policy to lower my premium?",
            "assistant": "{greeting} Yes, you can remove comprehensive coverage from your {policy_type} policy ({policy_number}). However, please be aware that this means you won't be covered for non-collision events like {accident_type}, theft, or weather damage. Your premium would decrease from {premium}. I recommend keeping comprehensive coverage if your {vehicle} has significant value. {closing}",
        },
        # --- Template 8: Change deductible ---
        {
            "user": "I want to change my deductible from {deductible} to a higher amount to lower my premium.",
            "assistant": "{greeting} Increasing your deductible is a common way to reduce your premium. By raising your {policy_type} deductible, you'll pay more out-of-pocket when you file a claim, but your monthly premium will decrease. I can show you how different deductible levels affect your premium of {premium}. Would you like me to run those calculations? {closing}",
        },
        # --- Template 9: Beneficiary update ---
        {
            "user": "I need to update the beneficiary on my {policy_type} policy.",
            "assistant": "{greeting} I can help you update the beneficiary on your {policy_type} policy ({policy_number}). You'll need to provide: 1) The new beneficiary's full legal name. 2) Their relationship to you. 3) Their date of birth. 4) Their contact information. The change will take effect once the form is processed, typically {timeframe}. {closing}",
        },
        # --- Template 10: Policy reinstatement ---
        {
            "user": "My {policy_type} policy lapsed due to non-payment. Can I get it reinstated?",
            "assistant": "{greeting} Policy reinstatement is possible depending on how long the policy has been lapsed. For your {policy_type} policy ({policy_number}): 1) If lapsed less than 30 days, we can reinstate immediately upon payment of {premium} plus any late fees. 2) If lapsed 30-90 days, reinstatement requires a new application review. 3) Beyond 90 days, a new policy may be required. I'll check your specific situation right now. {closing}",
        },
    ],
}


class TemplateGenerator:
    """Fast template-based data generation with controlled randomization."""

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.rng = random.Random(seed)
        self.generated_texts = set()  # Track uniqueness

    def _fill_template(self, template: Dict[str, str]) -> Dict[str, str]:
        """Fill template placeholders with random values."""
        # Build replacement pool
        replacements = {
            "customer_name": self.rng.choice(CUSTOMER_NAMES),
            "policy_number": self.rng.choice(POLICY_NUMBERS),
            "claim_number": self.rng.choice(CLAIM_NUMBERS),
            "policy_type": self.rng.choice(POLICY_TYPES),
            "coverage_amount": self.rng.choice(COVERAGE_AMOUNTS),
            "deductible": self.rng.choice(DEDUCTIBLE_AMOUNTS),
            "premium": self.rng.choice(PREMIUM_AMOUNTS),
            "vehicle": self.rng.choice(VEHICLE_TYPES),
            "accident_type": self.rng.choice(ACCIDENT_TYPES),
            "health_condition": self.rng.choice(HEALTH_CONDITIONS),
            "property_type": self.rng.choice(PROPERTY_TYPES),
            "property_damage": self.rng.choice(PROPERTY_DAMAGE_TYPES),
            "payment_method": self.rng.choice(PAYMENT_METHODS),
            "modification": self.rng.choice(MODIFICATION_TYPES),
            "greeting": self.rng.choice(AGENT_GREETINGS),
            "closing": self.rng.choice(AGENT_CLOSINGS),
            "timeframe": self.rng.choice(TIMEFRAMES),
        }

        user_text = template["user"]
        assistant_text = template["assistant"]

        for key, value in replacements.items():
            user_text = user_text.replace("{" + key + "}", value)
            assistant_text = assistant_text.replace("{" + key + "}", value)

        return {
            "user": user_text,
            "assistant": assistant_text,
        }

    def generate(
        self,
        category: str,
        count: int
    ) -> List[Dict[str, str]]:
        """
        Generate examples for a category using templates.

        Args:
            category: Domain category
            count: Number of examples

        Returns:
            List of generated examples
        """
        templates = TEMPLATES.get(category, [])
        if not templates:
            logger.warning(f"No templates found for category: {category}")
            return []

        examples = []
        attempts = 0
        max_attempts = count * 5  # Prevent infinite loops

        while len(examples) < count and attempts < max_attempts:
            template = self.rng.choice(templates)
            filled = self._fill_template(template)

            # Uniqueness check via user text fingerprint
            fingerprint = filled["user"][:80]
            if fingerprint not in self.generated_texts:
                self.generated_texts.add(fingerprint)
                filled["category"] = category
                examples.append(filled)

            attempts += 1

        return examples

    def generate_all(
        self,
        total: int = 1000,
        balanced: bool = True
    ) -> List[Dict[str, str]]:
        """
        Generate examples across all categories.

        Args:
            total: Total number of examples
            balanced: If True, distribute evenly

        Returns:
            List of all generated examples
        """
        categories = list(TEMPLATES.keys())
        all_examples = []

        if balanced:
            per_category = total // len(categories)
            for category in categories:
                examples = self.generate(category, per_category)
                all_examples.extend(examples)
                logger.info(
                    f"  [{category}] Generated {len(examples)}/{per_category} examples"
                )
        else:
            # Weighted random
            for _ in range(total):
                category = self.rng.choice(categories)
                examples = self.generate(category, 1)
                all_examples.extend(examples)

        # Shuffle final dataset
        self.rng.shuffle(all_examples)

        logger.info(f"\n✓ Template generation complete: {len(all_examples)} examples")
        return all_examples


class OllamaGenerator:
    """Generate enriched examples using Ollama + Llama 3.1."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama3.1",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify Ollama server is running."""
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            resp.raise_for_status()
            logger.info(f"✓ Ollama connected ({self.model_name})")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_url}. "
                f"Ensure Ollama is running: ollama serve\n{e}"
            )

    def generate_example(
        self,
        category: str,
        temperature: float = 0.8,
    ) -> Optional[Dict[str, str]]:
        """Generate one example via Ollama."""
        prompt = self._build_prompt(category)

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "temperature": temperature,
                        "num_predict": 400,
                        "stream": False,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                text = resp.json().get("response", "").strip()

                if text:
                    parsed = self._parse_response(text, category)
                    if parsed:
                        return parsed

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout (attempt {attempt + 1})")
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning(f"Error (attempt {attempt + 1}): {e}")
                time.sleep(1)

        return None

    def _build_prompt(self, category: str) -> str:
        """Build category-specific prompt."""
        descriptions = {
            "policy_inquiry": "asking about their insurance policy details, coverage, deductible, or terms",
            "claim_processing": "filing an insurance claim, checking claim status, or asking about claim documents",
            "coverage_questions": "asking what is and isn't covered under their insurance policy",
            "premium_billing": "asking about premium payments, billing, discounts, or payment methods",
            "policy_modifications": "requesting changes to their policy like adding drivers, updating address, or canceling",
        }

        desc = descriptions.get(category, descriptions["policy_inquiry"])

        return f"""Generate a realistic customer support conversation between a customer and an insurance agent.

The customer is {desc}.

Format EXACTLY as:
USER: [one specific customer question]
AGENT: [professional 2-3 sentence response]

Generate one unique, realistic conversation now:"""

    def _parse_response(
        self, text: str, category: str
    ) -> Optional[Dict[str, str]]:
        """Parse LLM output into user/assistant pair."""
        try:
            if "USER:" in text and "AGENT:" in text:
                user_part = text.split("USER:")[1].split("AGENT:")[0].strip()
                agent_part = text.split("AGENT:")[-1].strip()

                # Remove any trailing conversations
                for marker in ["USER:", "\n\n\n"]:
                    if marker in agent_part:
                        agent_part = agent_part.split(marker)[0].strip()

                if 10 < len(user_part) < 500 and 10 < len(agent_part) < 800:
                    return {
                        "category": category,
                        "user": user_part,
                        "assistant": agent_part,
                        "source": "ollama",
                    }
        except Exception:
            pass
        return None

    def generate_batch(
        self,
        category: str,
        count: int,
        delay: float = 0.5,
    ) -> List[Dict[str, str]]:
        """Generate multiple examples."""
        examples = []
        for i in range(count):
            example = self.generate_example(category)
            if example:
                examples.append(example)
                logger.info(f"  [Ollama {category}] {i+1}/{count} ✓")
            else:
                logger.info(f"  [Ollama {category}] {i+1}/{count} ✗")
            if i < count - 1:
                time.sleep(delay)
        return examples


class HybridGenerator:
    """
    Hybrid data generator combining templates and Ollama.

    Strategy:
    - ~80% from templates (instant, deterministic)
    - ~20% from Ollama (natural variation, edge cases)
    """

    def __init__(
        self,
        template_ratio: float = 0.8,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "llama3.1",
        seed: int = 42,
    ):
        self.template_ratio = template_ratio
        self.seed = seed

        # Template generator (always available)
        self.template_gen = TemplateGenerator(seed=seed)

        # Ollama generator (optional, graceful fallback)
        self.ollama_gen = None
        try:
            self.ollama_gen = OllamaGenerator(
                ollama_url=ollama_url,
                model_name=model_name,
            )
        except ConnectionError:
            logger.warning(
                "⚠ Ollama not available. Using template-only generation. "
                "Start Ollama for hybrid mode: ollama serve"
            )

    def generate(
        self,
        total: int = 1000,
        balanced: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Generate full dataset using hybrid approach.

        Args:
            total: Total number of examples
            balanced: Distribute evenly across categories

        Returns:
            Combined list of examples
        """
        categories = list(TEMPLATES.keys())

        # Calculate split
        n_template = int(total * self.template_ratio)
        n_ollama = total - n_template

        logger.info("=" * 60)
        logger.info("HYBRID DATA GENERATION")
        logger.info("=" * 60)
        logger.info(f"Total target: {total}")
        logger.info(f"Template: {n_template} ({self.template_ratio*100:.0f}%)")
        logger.info(f"Ollama: {n_ollama} ({(1-self.template_ratio)*100:.0f}%)")
        logger.info(f"Categories: {len(categories)}")
        logger.info("=" * 60)

        # --- Phase 1: Template generation (instant) ---
        logger.info("\n[Phase 1] Template-based generation...")
        template_examples = self.template_gen.generate_all(
            total=n_template, balanced=balanced
        )

        # Tag source
        for ex in template_examples:
            ex["source"] = "template"

        # --- Phase 2: Ollama enrichment (if available) ---
        ollama_examples = []
        if self.ollama_gen and n_ollama > 0:
            logger.info(f"\n[Phase 2] Ollama enrichment ({n_ollama} examples)...")
            per_cat = n_ollama // len(categories)

            for category in categories:
                batch = self.ollama_gen.generate_batch(category, per_cat)
                ollama_examples.extend(batch)
                logger.info(
                    f"  [{category}] Ollama: {len(batch)}/{per_cat}"
                )
        else:
            if n_ollama > 0:
                logger.warning(
                    "Ollama not available, generating extra templates instead"
                )
                extra = self.template_gen.generate_all(
                    total=n_ollama, balanced=balanced
                )
                for ex in extra:
                    ex["source"] = "template_fallback"
                ollama_examples = extra

        # --- Combine ---
        all_examples = template_examples + ollama_examples
        rng = random.Random(self.seed)
        rng.shuffle(all_examples)

        # --- Stats ---
        logger.info("\n" + "=" * 60)
        logger.info("GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total generated: {len(all_examples)}")

        from collections import Counter
        cat_counts = Counter(ex["category"] for ex in all_examples)
        src_counts = Counter(ex.get("source", "unknown") for ex in all_examples)

        logger.info("\nBy category:")
        for cat, count in sorted(cat_counts.items()):
            logger.info(f"  {cat}: {count}")

        logger.info("\nBy source:")
        for src, count in sorted(src_counts.items()):
            logger.info(f"  {src}: {count}")

        logger.info("=" * 60)

        return all_examples


def generate_synthetic_dataset(
    output_path: Path,
    num_examples: int = 1000,
    model_name: str = "llama3.1",
    ollama_url: str = "http://localhost:11434",
    template_ratio: float = 0.8,
    seed: int = 42,
) -> Path:
    """
    Main entry point for synthetic data generation.

    Args:
        output_path: Where to save the dataset
        num_examples: Total examples to generate
        model_name: Ollama model name
        ollama_url: Ollama server URL
        template_ratio: Fraction from templates (0-1)
        seed: Random seed

    Returns:
        Path to saved dataset
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = HybridGenerator(
        template_ratio=template_ratio,
        ollama_url=ollama_url,
        model_name=model_name,
        seed=seed,
    )

    examples = generator.generate(total=num_examples, balanced=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Dataset saved: {output_path}")
    logger.info(f"  Examples: {len(examples)}")
    logger.info(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output = Path("./data/raw/synthetic_insurance_data.json")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    generate_synthetic_dataset(output, num_examples=n)