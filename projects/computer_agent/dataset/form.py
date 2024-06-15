from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import json
import synthetic_form_engine.form_engine as form_engine

# Import from parent folder
import sys
sys.path.append("..")
import utils

output_path = '/home/michael/ai/projects/computer_agent/data/form_train_raw.json'

# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run headlessly
chrome_options.add_argument('--window-size=1440x900')  # Set window size
chrome_options.binary_location = '/home/michael/chromedriver/extracted/opt/google/chrome/google-chrome'  # Specify Chrome binary location

# Path to the ChromeDriver executable
service = Service(executable_path="/home/michael/chromedriver/chromedriver-linux64/chromedriver")
driver = webdriver.Chrome(service=service, options=chrome_options)
actions = ActionChains(driver)

# Define the HTML content for the form
form_generator = form_engine.FormEngine()
form_html = form_generator.generate_form()

# Write the HTML to a file
html_file_path = 'form.html'
with open(html_file_path, 'w') as file:
    file.write(form_html)

# Load the HTML file
driver.get(f'file:///home/michael/ai/projects/computer_agent/dataset/{html_file_path}')

target_actions = {
    "prompt": None,
    "actions" : []
}
try:
    # Wait until the current URL is as expected
    WebDriverWait(driver, 10).until(EC.url_contains("form.html"))
    print("WebDriver loaded the page successfully.")

    # Get bounding boxes of elements
    elements = utils.get_elements_dict(driver, element_ids=["inputField", "submitBtn"])
        
    target_actions["prompt"] = "Fill out this form. Your first name is Michael."

    # Take a screenshot
    screenshot_path = '/home/michael/ai/projects/computer_agent/data/form/0_0.png'
    driver.save_screenshot(screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")

    # Generate ground truth trajectory
    # Step 1: Click the middle of the box
    action = {
        "image_path": screenshot_path,
        "target": f"click:{elements['inputField']['middle_x']},{elements['inputField']['middle_y']}"
    }
    target_actions["actions"].append(action)

    # Interact and take a new screenshot
    actions.move_by_offset(elements["inputField"]["middle_x"], elements["inputField"]["middle_y"]).click().perform()
    screenshot_path = '/home/michael/ai/projects/computer_agent/data/form/0_1.png'
    driver.save_screenshot(screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")    

    # Step 2: Type input
    action = {
        "image_path": screenshot_path,
        "target": f"type:Michael"
    }
    target_actions["actions"].append(action)

    # Interact and take a new screenshot
    actions.send_keys("Michael")
    actions.perform()
    screenshot_path = '/home/michael/ai/projects/computer_agent/data/form/0_2.png'
    driver.save_screenshot(screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")

    # Step 3: Click button
    action = {
        "image_path": screenshot_path,
        "target": f"click:{elements['submitBtn']['middle_x']},{elements['submitBtn']['middle_y']}"
    }
    target_actions["actions"].append(action)

    # No need to execute action on the last one

    # Save to json
    with open(output_path, 'w') as f:
        json.dump(target_actions, f, indent=4)
    print(f"Target actions saved to {output_path}")

    # TODO: Make this form generation process entirely dynamic

    



finally:
    driver.quit()