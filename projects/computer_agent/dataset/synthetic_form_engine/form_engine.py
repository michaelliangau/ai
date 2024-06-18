import random

from . import dataclass
from . import constants
import datasets
class FormEngine:
    def __init__(self, min_input_fields=1, max_input_fields=5):
        self.num_fields = random.randint(min_input_fields, max_input_fields)
        self.form_html_parts = []
        self.input_field_engine = datasets.load_from_disk(
            "/home/michael/ai/projects/computer_agent/data/squad_v2_qa"
        )

    def generate_sequential_id(self, prefix, index):
        return f"{prefix}_{index}"

    def generate_input_field(self, index):
        field_id = self.generate_sequential_id("inputField", index)
        required = random.choice([True, False])
        height = random.randint(constants.MIN_INPUT_FIELD_HEIGHT, constants.MAX_INPUT_FIELD_HEIGHT)
        width = random.randint(constants.MIN_INPUT_FIELD_WIDTH, constants.MAX_INPUT_FIELD_WIDTH)
        
        # Ensure enough room below the field for subsequent input elements
        max_middle_y = constants.SCREEN_HEIGHT - (self.num_fields - index) * constants.MIN_INPUT_FIELD_HEIGHT
        middle_y = random.randint(constants.MIN_INPUT_FIELD_HEIGHT // 2, max_middle_y)
        
        input_field = dataclass.InputField(
            id=field_id,
            required=required,
            height=height,
            width=width,
            middle_x=random.randint(constants.MIN_INPUT_FIELD_WIDTH // 2, constants.SCREEN_WIDTH - constants.MIN_INPUT_FIELD_WIDTH // 2),
            middle_y=middle_y,
        )
        label_id = self.generate_sequential_id("label", index)

        while True:
            rand_int = random.randint(0, len(self.input_field_engine) - 1)
            input_field_text = self.input_field_engine[rand_int]["question"]
            input_field_answer = self.input_field_engine[rand_int]["answers"]["text"]
            if input_field_answer:
                input_field_answer = input_field_answer[0]
                break
        label = dataclass.Label(
            id=label_id,
            text=input_field_text,
            top_left_x=input_field.middle_x - (input_field.width // 2),
            top_left_y=input_field.middle_y - (input_field.height // 2) - random.randint(10,20)
        )
        input_field.label = label
        input_field_answer_dict = {
            "id": label_id,
            "answer": input_field_answer
        }
        return input_field, input_field_answer_dict

    def generate_button(self, index):
        button_id = self.generate_sequential_id("button", index)
        button_text = random.choice(["Submit"])

        # TODO: Button requires 4 positional arguments...




        button = dataclass.Button(id=button_id, text=button_text)
        return button

    def decode_input_field(self, input_field):
        return f'<label for="{input_field.id}">Field {input_field.id}:</label><input type="text" id="{input_field.id}" name="{input_field.id}" {"required" if input_field.required else ""}><br>'

    def decode_button(self, button):
        return f'<button id="{button.id}" type="button">{button.text}</button><br>'

    def generate_form(self):
        input_field_answer_dicts = []
        self.form_html_parts.append('<form id="form">')
        for i in range(self.num_fields):
            input_field, input_field_answer_dict = self.generate_input_field(i)
            self.form_html_parts.append(self.decode_input_field(input_field))
            input_field_answer_dicts.append(input_field_answer_dict)
        button = self.generate_button(0)
        self.form_html_parts.append(self.decode_button(button))
        self.form_html_parts.append('</form>')
        self.form_html = "\n".join(self.form_html_parts)
        return self.form_html, input_field_answer_dicts

