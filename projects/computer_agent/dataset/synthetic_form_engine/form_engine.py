import random

from . import dataclass
from . import constants
import datasets
class FormEngine:
    def __init__(self, min_input_fields=1, max_input_fields=5):
        self.num_fields = random.randint(min_input_fields, max_input_fields)
        self.form_html_parts = []
        self.input_field_engine = datasets.load_from_disk(
            "/Users/michael/Desktop/wip/ai/projects/computer_agent/dataset/data/squad_v2_qa"
        )

    def generate_sequential_id(self, prefix, index):
        return f"{prefix}_{index}"

    def generate_input_field(self, index: int, min_input_field_top_left_y:int=0):
        """
        Generate an input field with a random height, width, and position.

        Args:
            index (int): The index of the input field in the form.
            min_input_field_top_left_y (int): The minimum top left y position of the input field.
        
        Returns:
            dataclass.InputField: The generated input field.
        """
        # Generate id
        field_id = self.generate_sequential_id("inputField", index)
        
        # Define required or not
        required = random.choice([True, False])
        
        # Define position
        height = random.randint(constants.MIN_INPUT_FIELD_HEIGHT, constants.MAX_INPUT_FIELD_HEIGHT)
        width = random.randint(constants.MIN_INPUT_FIELD_WIDTH, constants.MAX_INPUT_FIELD_WIDTH)
        top_left_x = random.randint(0, constants.SCREEN_WIDTH - width)
        top_left_y = random.randint(min_input_field_top_left_y, min_input_field_top_left_y + constants.MAX_GAP_BETWEEN_INPUT_FIELDS)
        label_id = self.generate_sequential_id("label", index)

        # TODO: Match the position of the labels with the boxes

        # Get a random question and answer
        rand_int = random.randint(0, len(self.input_field_engine) - 1)
        input_field_text = self.input_field_engine[rand_int]["question"]
        input_field_answer = self.input_field_engine[rand_int]["answers"]["text"][0]

        # Create label (above input field)
        label = dataclass.Label(
            id=label_id,
            text=input_field_text,
            top_left_x=top_left_x,
            top_left_y=top_left_y - random.randint(10,20)
        )
        
        # Create input field
        input_field = dataclass.InputField(
            id=field_id,
            required=required,
            height=height,
            width=width,
            top_left_x=top_left_x,
            top_left_y=top_left_y,
            label=label,
            answer=input_field_answer
        )
        return input_field, top_left_y + height

    def generate_button(self, index):
        button_id = self.generate_sequential_id("button", index)
        button_text = random.choice(["Submit"])

        # TODO: Button requires 4 positional arguments...

        button = dataclass.Button(id=button_id, text=button_text)
        return button

    def decode_input_field(self, input_field):
        required = "required" if input_field.required else ""
        return f'<label for="{input_field.id}">Field {input_field.id}:</label>' \
               f'<input type="text" id="{input_field.id}" name="{input_field.id}" ' \
               f'{required} ' \
               f'style="position:absolute; left:{input_field.top_left_x}px; top:{input_field.top_left_y}px; ' \
               f'width:{input_field.width}px; height:{input_field.height}px;"><br>'

    def decode_button(self, button):
        return f'<button id="{button.id}" type="button">{button.text}</button><br>'

    def generate_form(self):
        input_fields = []
        self.form_html_parts = [
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
        min_input_field_top_left_y = 0
        for idx in range(self.num_fields):
            input_field, min_input_field_top_left_y = self.generate_input_field(idx, min_input_field_top_left_y=min_input_field_top_left_y)
            self.form_html_parts.append(self.decode_input_field(input_field))
            input_fields.append(input_field)
        # button = self.generate_button(0)
        # self.form_html_parts.append(self.decode_button(button))
        self.form_html_parts.extend([
            '</form>',
            '</body>',
            '</html>'
        ])
        self.form_html = "\n".join(self.form_html_parts)
        return self.form_html, input_fields

