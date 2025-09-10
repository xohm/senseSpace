from dataclasses import dataclass
from typing import List, Dict, Union
from .enums import Body18Joint, Body34Joint


@dataclass
class Joint:
    i: int                        # raw index from server (0..17 or 0..33)
    pos: Dict[str, float]         # {"x":..,"y":..,"z":..}
    ori: Dict[str, float]         # quaternion {"x":..,"y":..,"z":..,"w":..}
    conf: float                   # confidence 0..1

    def as_enum(self, body_model: str) -> Union[Body18Joint, Body34Joint]:
        """Return this joint as the appropriate enum"""
        if body_model == "BODY_18":
            return Body18Joint(self.i)
        elif body_model == "BODY_34":
            return Body34Joint(self.i)
        else:
            return self.i  # fallback: raw index

    def to_dict(self):
        return {
            "i": self.i,
            "pos": self.pos,
            "ori": self.ori,
            "conf": self.conf
        }

    @staticmethod
    def from_dict(d):
        return Joint(i=d["i"], pos=d["pos"], ori=d["ori"], conf=d["conf"])


@dataclass
class Person:
    id: int
    tracking_state: str
    confidence: float
    skeleton: List[Joint]

    def to_dict(self):
        return {
            "id": self.id,
            "tracking_state": self.tracking_state,
            "confidence": self.confidence,
            "skeleton": [j.to_dict() for j in self.skeleton]
        }

    @staticmethod
    def from_dict(d):
        return Person(
            id=d["id"],
            tracking_state=d["tracking_state"],
            confidence=d["confidence"],
            skeleton=[Joint.from_dict(j) for j in d["skeleton"]]
        )


@dataclass
class Frame:
    timestamp: float
    people: List[Person]
    body_model: str = ""   # set from config
    floor_height: float = None  # ZED SDK detected floor height in mm

    def to_dict(self):
        data = {
            "timestamp": self.timestamp,
            "people": [p.to_dict() for p in self.people],
            "body_model": self.body_model,
        }
        # Only include floor_height when present
        if self.floor_height is not None:
            data["floor_height"] = self.floor_height
        return data

    @staticmethod
    def from_dict(d):
        return Frame(
            timestamp=d["timestamp"],
            people=[Person.from_dict(p) for p in d["people"]],
            body_model=d.get("body_model", ""),
            floor_height=d.get("floor_height", None)
        )
