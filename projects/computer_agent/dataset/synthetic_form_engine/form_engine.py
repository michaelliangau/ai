import random

from . import dataclass
from . import constants
import datasets

class FormEngine:
    def __init__(self, min_text_fields: int = constants.MIN_NUM_TEXT_FIELDS, max_text_fields: int =constants.MAX_NUM_TEXT_FIELDS):
        self.num_text_fields = random.randint(min_text_fields, max_text_fields)
        self.text_field_dataset = datasets.load_from_disk(
            "/Users/michael/Desktop/wip/ai/projects/computer_agent/dataset/data/squad_v2_qa"
        )

    def generate_sequential_id(self, prefix, index):
        return f"{prefix}_{index}"

    def generate_text_field_dataclass(self, index: int, min_text_field_top_left_y:int=0):
        """
        Generate an input field dataclass

        Args:
            index (int): The index of the input field in the form.
            min_text_field_top_left_y (int): The minimum top left y position of the input field.
        
        Returns:
            dataclass.TextField: The generated input field.
        """
        # Generate id
        field_id = self.generate_sequential_id("textField", index)
        
        # Define position
        height = constants.TEXT_FIELD_HEIGHT
        width = random.randint(constants.MIN_TEXT_FIELD_WIDTH, constants.MAX_TEXT_FIELD_WIDTH)
        top_left_x = random.randint(0, constants.SCREEN_WIDTH // 2 - width)
        top_left_y = min_text_field_top_left_y + constants.GAP_BETWEEN_ELEMENTS
        label_id = self.generate_sequential_id("label", index)

        # Get a random question and answer
        rand_int = random.randint(0, len(self.text_field_dataset) - 1)
        text_field_text = self.text_field_dataset[rand_int]["question"]
        text_field_answer = self.text_field_dataset[rand_int]["answers"]["text"][0]

        # Create label (above input field)
        label = dataclass.Label(
            id=label_id,
            text=text_field_text,
            top_left_x=top_left_x,
            top_left_y=top_left_y - constants.GAP_BETWEEN_TEXT_FIELD_AND_LABEL
        )
        
        # Create input field
        text_field = dataclass.TextField(
            id=field_id,
            height=height,
            width=width,
            top_left_x=top_left_x,
            top_left_y=top_left_y,
            label=label,
            answer=text_field_answer
        )
        return text_field, top_left_y + height

    def generate_button_dataclass(self, index, min_button_top_left_y:int=0):
        """
        Generate a button dataclass

        Args:
            index (int): The index of the button in the form.
            min_button_top_left_y (int): The minimum top left y position of the button.
        
        Returns:
            dataclass.Button: The generated button.
        """
        button_id = self.generate_sequential_id("button", index)
        button_text = random.choice(["Submit", "Finish"])
        button = dataclass.Button(
            id=button_id,
            text=button_text,
            height=constants.BUTTON_HEIGHT,
            width=constants.BUTTON_WIDTH,
            top_left_x=random.randint(0, constants.SCREEN_WIDTH - constants.BUTTON_WIDTH),
            top_left_y=min_button_top_left_y + constants.GAP_BETWEEN_ELEMENTS
        )
        return button

    def generate_text_field_html(self, text_field):
        return f'<label for="{text_field.id}" style="position:absolute; left:{text_field.label.top_left_x}px; top:{text_field.label.top_left_y}px;">{text_field.label.text}:</label>' \
               f'<input type="text" id="{text_field.id}" name="{text_field.id}" ' \
               f'style="position:absolute; left:{text_field.top_left_x}px; top:{text_field.top_left_y}px; ' \
               f'width:{text_field.width}px; height:{text_field.height}px;"><br>'

    def generate_button_html(self, button):
        return f'<button id="{button.id}" type="button" ' \
               f'style="position:absolute; left:{button.top_left_x}px; top:{button.top_left_y}px; ' \
               f'width:{button.width}px; height:{button.height}px;">{button.text}</button><br>'
    
    def generate_form(self):
        elements = []
        html = [
            '<!DOCTYPE html>',
            '<html lang="en">',
            '<head>',
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            '<title>Form</title>',
            '</head>',
            '<body>',
            '<form id="form">'
        ]
        max_element_y = 0
        for idx in range(self.num_text_fields):
            text_field, max_element_y = self.generate_text_field_dataclass(index=idx, min_text_field_top_left_y=max_element_y)
            html.append(self.generate_text_field_html(text_field))
            elements.append(text_field)
        button = self.generate_button_dataclass(index=0, min_button_top_left_y=max_element_y)
        html.append(self.generate_button_html(button))
        elements.append(button)
        html.extend([
            '</form>',
            '</body>',
            '</html>'
        ])
        out_html = "\n".join(html)
        return out_html, elements

