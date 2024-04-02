from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# URL you want to open
url = "https://docs.google.com/forms/d/e/1FAIpQLSeSxglhKz5qludFOp4w3diD58RXFJbB-cXVeuE3PaXTkmnEGg/viewform"

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Optional: if you're running in a headless environment
chrome_options.add_argument("--no-sandbox")  # Bypass OS security model, REQUIRED on Linux if not root
chrome_options.add_argument("--window-size=1920,1080")

# Connect to the Selenium Server running in Docker
driver = webdriver.Remote(
    command_executor='http://localhost:4444/wd/hub',
    options=chrome_options  # Use the options argument instead of desired_capabilities
)

# Open the URL
driver.get(url)

# Specify the path and filename where you want to save the screenshot
screenshot_path = "data/screenshot.png"

# Take a screenshot of the current window and save it to the specified file
driver.save_screenshot(screenshot_path)

# Multimodal model
pretrained_path = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(pretrained_path)
model = FuyuForCausalLM.from_pretrained(pretrained_path, device_map='cpu')

# Map natural language actions to UI actions