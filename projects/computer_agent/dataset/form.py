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
chrome_options.binary_location = '/home/michael/chromedriver/extracted/opt/google/chrome/google-chrome'  # Specify Chrome binary location

# Path to the ChromeDriver executable
service = Service(executable_path="/home/michael/chromedriver/chromedriver-linux64/chromedriver")
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
driver.get(f'file:///home/michael/ai/projects/computer_agent/dataset/{html_file_path}')

try:
    # Wait until the current URL is as expected
    WebDriverWait(driver, 10).until(EC.url_contains("form.html"))
    WebDriverWait(driver, 10).until(EC.title_is("Simple Form"))
    print("WebDriver loaded the page successfully.")

    # Take a screenshot
    screenshot_path = 'form_screenshot.png'
    driver.save_screenshot(screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")

finally:
    driver.quit()