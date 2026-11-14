"""Structured JSON Output Engine with Pydantic Validation and Schema Repair.

Provides predefined analysis schemas (Invoice, Resume, Contract, Research, General)
and resilient JSON extraction, schema validation, and retry mechanisms.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Type, Union, Tuple
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# =====================================================================
# PREDEFINED PYDANTIC SCHEMAS
# =====================================================================

class InvoiceItem(BaseModel):
    description: str = Field(description="Line item description")
    quantity: Optional[float] = Field(default=1.0, description="Quantity")
    unit_price: Optional[float] = Field(default=None, description="Price per unit")
    amount: Optional[float] = Field(default=None, description="Total amount for line item")


class InvoiceExtraction(BaseModel):
    """Schema for extracting structured financial invoice information."""
    invoice_number: Optional[str] = Field(default=None, description="Unique invoice ID / number")
    invoice_date: Optional[str] = Field(default=None, description="Date invoice was issued (YYYY-MM-DD if possible)")
    vendor: Optional[str] = Field(default=None, description="Name of company or person issuing invoice")
    customer: Optional[str] = Field(default=None, description="Name of recipient / bill-to customer")
    total_amount: Optional[float] = Field(default=None, description="Total final invoice amount payable")
    tax_amount: Optional[float] = Field(default=None, description="Tax or VAT amount charged")
    currency: Optional[str] = Field(default="USD", description="Currency symbol or code, e.g. USD, EUR, INR")
    line_items: List[InvoiceItem] = Field(default_factory=list, description="List of itemized invoice lines")
    citations: List[str] = Field(default_factory=list, description="Source document and page citations")


class EducationEntry(BaseModel):
    institution: str = Field(description="University, college, or school name")
    degree: Optional[str] = Field(default=None, description="Degree or program completed")
    year_graduated: Optional[str] = Field(default=None, description="Graduation year or date range")


class ExperienceEntry(BaseModel):
    company: str = Field(description="Company or organization name")
    role: str = Field(description="Job title or role held")
    duration: Optional[str] = Field(default=None, description="Dates or duration of employment")
    responsibilities: List[str] = Field(default_factory=list, description="Key duties and achievements")


class ContactInfo(BaseModel):
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    location: Optional[str] = Field(default=None, description="City, State, Country")
    linkedin_or_portfolio: Optional[str] = Field(default=None, description="LinkedIn profile or website")


class ResumeExtraction(BaseModel):
    """Schema for candidate resume and CV analysis."""
    candidate_name: Optional[str] = Field(default=None, description="Full name of candidate")
    contact_information: Optional[ContactInfo] = Field(default_factory=ContactInfo, description="Contact details")
    skills: List[str] = Field(default_factory=list, description="Technical and soft skills")
    education: List[EducationEntry] = Field(default_factory=list, description="Educational background")
    work_experience: List[ExperienceEntry] = Field(default_factory=list, description="Work history")
    certifications: List[str] = Field(default_factory=list, description="Certifications and licenses")
    summary: Optional[str] = Field(default=None, description="Executive summary or candidate objective")
    citations: List[str] = Field(default_factory=list, description="Source document and page citations")


class ContractAnalysis(BaseModel):
    """Schema for legal contract and agreement review."""
    parties: List[str] = Field(default_factory=list, description="Names of contracting entities/parties")
    effective_date: Optional[str] = Field(default=None, description="Start date of agreement")
    expiration_date: Optional[str] = Field(default=None, description="End or renewal date")
    payment_terms: Optional[str] = Field(default=None, description="Summary of payment schedules and terms")
    obligations: List[str] = Field(default_factory=list, description="Primary duties and deliverables")
    termination_clauses: List[str] = Field(default_factory=list, description="Conditions allowing contract termination")
    potential_risks: List[str] = Field(default_factory=list, description="Unfavorable indemnities, liabilities, or ambiguities")
    governing_law: Optional[str] = Field(default=None, description="Jurisdiction / governing legal jurisdiction")
    citations: List[str] = Field(default_factory=list, description="Clause and page citations")


class ResearchPaperAnalysis(BaseModel):
    """Schema for academic research paper extraction."""
    title: Optional[str] = Field(default=None, description="Title of the research paper")
    authors: List[str] = Field(default_factory=list, description="Author names and affiliations")
    research_problem: Optional[str] = Field(default=None, description="Core problem or research hypothesis")
    methodology: Optional[str] = Field(default=None, description="Experimental setup, algorithms, or dataset")
    findings: List[str] = Field(default_factory=list, description="Key discoveries, quantitative metrics, results")
    limitations: List[str] = Field(default_factory=list, description="Authors' acknowledged constraints or weaknesses")
    conclusion: Optional[str] = Field(default=None, description="Overall conclusion and future directions")
    citations: List[str] = Field(default_factory=list, description="Source paper citations")


class GeneralDocumentExtraction(BaseModel):
    """Schema for generic document extraction and custom user fields."""
    document_title: Optional[str] = Field(default=None, description="Title or inferred document topic")
    document_type: Optional[str] = Field(default=None, description="Category (report, policy, memo, manual, etc.)")
    key_entities: List[str] = Field(default_factory=list, description="Organizations, people, products mentioned")
    key_takeaways: List[str] = Field(default_factory=list, description="Important bullet point conclusions")
    extracted_fields: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary custom extracted key-values")
    citations: List[str] = Field(default_factory=list, description="Source document citations")


# Schema Registry
SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "invoice": InvoiceExtraction,
    "resume": ResumeExtraction,
    "contract": ContractAnalysis,
    "research": ResearchPaperAnalysis,
    "general": GeneralDocumentExtraction
}


# =====================================================================
# STRUCTURED OUTPUT PARSER & VALIDATOR
# =====================================================================

class StructuredOutputParser:
    """Handles extracting, cleaning, repairing, and validating JSON from LLM outputs."""

    @staticmethod
    def get_schema_class(schema_name: str) -> Type[BaseModel]:
        return SCHEMA_REGISTRY.get(schema_name.lower(), GeneralDocumentExtraction)

    @staticmethod
    def get_schema_json_instructions(schema_class: Type[BaseModel]) -> str:
        """Generate a strict JSON schema prompt instruction for the LLM."""
        schema_dict = schema_class.model_json_schema()
        # Pretty print schema for prompt clarity
        schema_str = json.dumps(schema_dict.get("properties", {}), indent=2)
        required = schema_dict.get("required", [])

        instructions = (
            "OUTPUT FORMAT INSTRUCTIONS:\n"
            "You MUST respond ONLY with a single valid JSON object adhering to this schema:\n"
            f"```json\n{schema_str}\n```\n"
        )
        if required:
            instructions += f"Required fields: {', '.join(required)}.\n"
        instructions += (
            "RULES:\n"
            "1. Output raw JSON ONLY. Do not write introductory words, conversational remarks, or closing commentary.\n"
            "2. If a field value is not found in the documents, set it to null or [] (do NOT invent data).\n"
            "3. Ground all extracted data in the provided document passages."
        )
        return instructions

    @classmethod
    def clean_and_extract_json(cls, raw_text: str) -> Dict[str, Any]:
        """Extract and parse a JSON object from raw LLM text, stripping markdown code fences."""
        text = raw_text.strip()

        # 1. Direct parse attempt
        try:
            return json.loads(text)
        except Exception:
            pass

        # 2. Extract from markdown ```json ... ``` or ``` ... ```
        fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.IGNORECASE)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except Exception:
                pass

        # 3. Find outermost curly braces { ... }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            sub = text[start_idx:end_idx + 1]
            try:
                return json.loads(sub)
            except Exception:
                # Common repair: replace single quotes with double quotes
                try:
                    repaired = sub.replace("'", '"')
                    # Remove trailing commas before closing braces
                    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
                    return json.loads(repaired)
                except Exception:
                    pass

        raise ValueError(f"Could not parse valid JSON from LLM response. Content begins: {text[:200]}")

    @classmethod
    def validate_and_parse(
        cls,
        raw_text: str,
        schema_class: Type[BaseModel]
    ) -> Tuple[Optional[BaseModel], Optional[str], Optional[Dict[str, Any]]]:
        """Parse raw text, validate against schema_class, and return (instance, error_message, raw_dict)."""
        try:
            parsed_dict = cls.clean_and_extract_json(raw_text)
        except Exception as e:
            return None, f"JSON Extraction Error: {str(e)}", None

        try:
            instance = schema_class.model_validate(parsed_dict)
            return instance, None, parsed_dict
        except ValidationError as ve:
            error_details = ve.errors()
            err_summary = "; ".join([f"{'.'.join(str(l) for l in err['loc'])}: {err['msg']}" for err in error_details])
            logger.warning(f"Schema validation error: {err_summary}")
            return None, f"Schema Validation Error: {err_summary}", parsed_dict
