import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

APP_URL = os.environ.get("APP_URL", "https://imageclassification-9dbztmit3lrgj9yj3npfxb.streamlit.app/")

options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

driver = webdriver.Chrome(options=options)

try:
    print(f"Visiting {APP_URL}...")
    driver.get(APP_URL)
    time.sleep(8)

    # Check for the Streamlit wake-up button
    wake_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Yes, get this app back up!')]")
    if not wake_buttons:
        wake_buttons = driver.find_elements(By.XPATH, "//button[contains(., 'get this app back up')]")

    if wake_buttons:
        print("App is sleeping! Clicking the wake-up button...")
        wake_buttons[0].click()
        time.sleep(10)
        print("Successfully sent wake-up signal to Streamlit! ✅")
    else:
        print("App is already active and awake! ✅")
        print(f"Page Title: {driver.title}")

except Exception as e:
    print(f"Error checking app state: {e}")
finally:
    driver.quit()
