"""Genotype class."""

from __future__ import annotations

from base import Base
from revolve2.core.database import HasId
from revolve2.core.optimization.ea import Parameters as GenericParameters


class Genotype(Base, HasId, GenericParameters):
    """A genotype that is an array of parameters."""

    __tablename__ = "genotype"