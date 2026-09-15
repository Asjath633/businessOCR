"""
Pydantic schemas for Business Card data validation.
Defines the structured JSON output format for extracted business card info.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class BusinessCardData(BaseModel):
    """Structured data extracted from a business card."""

    is_business_card: bool = Field(
        ..., 
        description="Whether the OCR text represents a business card"
    )

    name: Optional[List[str]] = Field(
        default_factory=list,
        description="Full name(s) of the person(s) on the card"
    )

    organization: Optional[str] = Field(
        None,
        description="Company or organization name"
    )

    designation: Optional[str] = Field(
        None,
        description="Job title or designation"
    )

    email: Optional[List[str]] = Field(
        default_factory=list,
        description="Email address(es)"
    )

    phone: Optional[List[str]] = Field(
        default_factory=list,
        description="Phone numbers"
    )

    address: Optional[str] = Field(
        None,
        description="Physical address"
    )

    @field_validator("name", "email", "phone", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v or []

class OCRResult(BaseModel):
    """Raw OCR extraction result."""

    raw_text: str = Field(
        ...,
        description="Raw text extracted by OCR engine"
    )

    confidence: Optional[float] = Field(
        None,
        description="Overall OCR confidence score"
    )