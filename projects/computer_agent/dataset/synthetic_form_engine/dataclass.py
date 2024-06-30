from dataclasses import dataclass

@dataclass
class Label:
    id: str
    text: str
    top_left_x: int
    top_left_y: int

@dataclass
class TextField:
    id: str
    height: int
    width: int
    top_left_x: int
    top_left_y: int
    label: Label
    answer: str

@dataclass
class Button:
    id: str
    text: str
    height: int
    width: int
    top_left_x: int
    top_left_y: int