from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str = Field(description="The canonical name of the entity")
    entity_type: str = Field(
        description="Type: Person, Organisation, Location, Money, Date, Event, Law, Infrastructure, Facility, Political Group"
    )
    description: str = Field(
        default="", description="Brief description of this entity in context"
    )


class Relation(BaseModel):
    source: str = Field(description="Name of the source entity")
    target: str = Field(description="Name of the target entity")
    relation_type: str = Field(
        description="Type of relationship, e.g. FUNDS, ANNOUNCES, LOCATED_IN"
    )
    description: str = Field(default="", description="Brief description of the relationship")


class KnowledgeGraph(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
