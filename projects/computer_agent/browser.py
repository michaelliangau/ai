from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class Browser():
    def __init__(self):
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Optional: if you're running in a headless environment
        chrome_options.add_argument("--no-sandbox")  # Bypass OS security model, REQUIRED on Linux if not root
        chrome_options.add_argument("--window-size=1920,1080")

        # Connect to the Selenium Server running in Docker
        self.driver = webdriver.Remote(
            command_executor='http://localhost:4444/wd/hub',
            options=chrome_options  # Use the options argument instead of desired_capabilities
        )

    def open_url(self, url):
        self.driver.get(url)

    def take_screenshot(self, screenshot_path):
        self.driver.save_screenshot(screenshot_path)