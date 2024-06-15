import random

from . import dataclasses
from . import constants
import transformers
import torch

class FormEngine:
    def __init__(self, min_input_fields=1, max_input_fields=5):
        self.num_fields = random.randint(min_input_fields, max_input_fields)
        self.num_buttons = 1
        self.form_html_parts = []
        model_id = "meta-llama/Meta-Llama-3-8B"
        self.llm = transformers.pipeline(
            "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
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
        
        input_field = dataclasses.InputField(
            id=field_id,
            required=required,
            height=height,
            width=width,
            middle_x=random.randint(constants.MIN_INPUT_FIELD_WIDTH // 2, constants.SCREEN_WIDTH - constants.MIN_INPUT_FIELD_WIDTH // 2),
            middle_y=middle_y,
        )
        label_id = self.generate_sequential_id("label", index)
        hi = self.llm("Generate a label for an input field in a form.")
        import IPython; IPython.embed()
        label_text = f"Field {index + 1}"
        # TODO: Enable labels other than directly above the input field
        label = dataclasses.Label(
            id=label_id,
            text=label_text,
            top_left_x=input_field.middle_x - (input_field.width // 2),
            top_left_y=input_field.middle_y - (input_field.height // 2) - random.randint(10,20)
        )
        input_field.label = label
        return input_field

    def generate_button(self, index):
        button_id = self.generate_sequential_id("button", index)
        button_text = random.choice(["Submit", "Click Me", "Go"])
        button = Button(id=button_id, text=button_text)
        return button

    def decode_input_field(self, input_field):
        return f'<label for="{input_field.id}">Field {input_field.id}:</label><input type="text" id="{input_field.id}" name="{input_field.id}" {"required" if input_field.required else ""}><br>'

    def decode_button(self, button):
        return f'<button id="{button.id}" type="button">{button.text}</button><br>'

    def generate_form(self):
        self.form_html_parts.append('<form id="form">')
        for i in range(self.num_fields):
            input_field = self.generate_input_field(i)
            self.form_html_parts.append(self.decode_input_field(input_field))
        for i in range(self.num_buttons):
            button = self.generate_button(i)
            self.form_html_parts.append(self.decode_button(button))
        self.form_html_parts.append('</form>')
        self.form_html = "\n".join(self.form_html_parts)
        return self.form_html

# Example usage:
# engine = DynamicFormEngine(num_fields=3, num_buttons=2)
# form_html = engine.generate_form()
# print(form_html)
