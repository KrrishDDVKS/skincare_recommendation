import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

url = "https://www.lorealparisusa.com/skin-care/fragrance-free"

options = ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(options=options)

try:
    driver.get(url)
    WebDriverWait(driver, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".oap-card__front"))
    )
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)  # let lazy-loaded images load in
    scroll_height = driver.execute_script("return document.body.scrollHeight")
    for i in range(0, scroll_height, 400):
        driver.execute_script(f"window.scrollTo(0, {i});")
        time.sleep(0.5)

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    products = soup.find_all("div", class_="oap-card__front")
    data = []

    for idx, prod in enumerate(products, 1):
        img_tag = prod.select_one(".oap-card__thumbnail img")
        image_url = None
        # Download actual image
        image_file = None
        if img_tag:
            if img_tag.get('src') and 'png' in img_tag['src']:
                image_url = img_tag['src']
            elif img_tag.get('data-src'):
                image_url = img_tag['data-src']
            elif img_tag.get('srcset'):
                srcset_imgs = [s.strip().split(' ')[0] for s in img_tag['srcset'].split(',')]
                image_url = srcset_imgs[-1] if srcset_imgs else None
        if image_url and image_url.startswith('/'):
            image_url = "https://www.lorealparisusa.com" + image_url
        response = requests.get(image_url)
        image_file = f'tmp_product_{idx}.png'
        with open(image_file, 'wb') as f:
            f.write(response.content)

    # Scroll through page in increments to trigger all lazy-load images


        subtitle = prod.select_one(".oap-card__subtitle")
        title = prod.select_one(".oap-card__title")
        link_parent = prod.select_one(".oap-card__link")
        link = "https://www.lorealparisusa.com" + link_parent['href'] if link_parent and 'href' in link_parent.attrs else None
        price = prod.select_one(".oap-card__price p")
        rating = prod.select_one(".oap-rating__average")

        # Neatly print product details
        print(f"Product #{idx}")
        print(f"Title    : {title.get_text(strip=True) if title else 'N/A'}")
        print(f"Subtitle : {subtitle.get_text(strip=True) if subtitle else 'N/A'}")
        print(f"Price    : {price.get_text(strip=True) if price else 'N/A'}")
        print(f"Rating   : {rating.get_text(strip=True) if rating else 'N/A'}")
        print(f"Link     : {link if link else 'N/A'}")
        if image_file:
            print(f"Image file saved as: {image_file} [size: {os.path.getsize(image_file)} bytes]")
        else:
            print("Image    : N/A")
        print("=" * 60)

finally:
    driver.quit()
