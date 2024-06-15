from dataclasses import dataclass

@dataclass
class InputField:
    id: str
    required: bool
    height: int
    width: int
    middle_x: int
    middle_y: int

@dataclass
class Label:
    id: str
    text: str
    top_left_x: int
    top_left_y: int

@dataclass
class Button:
    id: str
    text: str
    height: int
    width: int
    middle_x: int
    middle_y: int
