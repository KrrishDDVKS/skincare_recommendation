from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

url = "https://www.lorealparisusa.com/skin-care/anti-aging"

options = ChromeOptions()
options.add_argument("--headless=new")   # remove for debugging visually
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=options)

try:
    driver.get(url)
    # wait for product cards to appear (adjust selector if site uses a different class)
    WebDriverWait(driver, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".oap-card__front"))
    )
    # optionally scroll to load lazy-loaded items
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    products = soup.find_all("div", class_="oap-card__front")
    for product in products:
        print(product.get_text(strip=True))
    
finally:
    driver.quit()
