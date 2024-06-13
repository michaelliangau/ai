from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run headlessly
chrome_options.add_argument('--window-size=1440x900')  # Set window size

# Path to the ChromeDriver executable
# chrome_driver_path = 'path/to/chromedriver'  # Update this path

# Initialize the WebDriver
service = Service(executable_path="/Users/michael/Downloads/chrome-headless-shell-mac-arm64")
driver = webdriver.Chrome(service=service, options=chrome_options)

# Define the HTML content for the form
form_html = """
<html>
<head><title>Simple Form</title></head>
<body>
    <form id="simpleForm">
        <label for="inputField">Enter Text:</label>
        <input type="text" id="inputField" name="inputField" required>
        <button type="submit">Submit</button>
    </form>
</body>
</html>
"""

# Write the HTML to a file
html_file_path = 'form.html'
with open(html_file_path, 'w') as file:
    file.write(form_html)

# Load the HTML file
driver.get(f'file://{html_file_path}')

try:
    # Wait until the form is present
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, 'simpleForm'))
    )

    # Take a screenshot
    screenshot_path = 'form_screenshot.png'
    driver.save_screenshot(screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")

finally:
    driver.quit()
