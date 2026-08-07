"""Request schemas for the semantic highlighter.

Unknown fields are ignored rather than rejected. This sits in the search hot
path, and a connector template that grows a field should degrade to "highlight
what I understand" instead of failing every ``_search`` that reaches it.
"""

from typing import Optional

from pydantic import BaseModel


class Item(BaseModel):
    question: str
    context: str


class BatchRequest(BaseModel):
    # Batch mode (OpenSearch 3.3+): an array of pairs.
    inputs: Optional[list[Item]] = None
