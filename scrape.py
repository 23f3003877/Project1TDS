import requests
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from io import BytesIO
import json
import markdown
import os
from dotenv import load_dotenv
import os

load_dotenv()
cookies = {
    '_t': os.getenv("DISCOURSE_TOKEN")
}

def scrape_website(url):
    response = requests.get(url , cookies=cookies)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.get_text(separator=" " , strip=True)
    return content.strip()

def markdown_to_text(md_content):
    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(separator=" " , strip=True)
    return text

course_content = scrape_website("https://tds.s-anand.net/README.md")
course_content = markdown_to_text(course_content).replace("\n", " ")
with open("scraped_data/course_content.json", "w", encoding="utf-8") as file:
    json.dump({"course_content": course_content}, file, ensure_ascii=False, indent=4)



i = 0
end_reached = False
while True:
    url = f"https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34.json?page={i}"
    url_post = f"https://discourse.onlinedegree.iitm.ac.in/t/"
    page_resp = requests.get(url, cookies=cookies)
    if page_resp.status_code != 200:
        print(page_resp.text)
        break
    data = page_resp.json()
    if not data['topic_list']['topics']:
        break

    for topic in data['topic_list']['topics']:
        with open("scraped_data/topics.json", "a", encoding="utf-8") as file:
            j = 0
            the_end = False
            while True:
                post_url = f"{url_post}{topic['id']}.json?page={j}"
                post_resp = requests.get(post_url, cookies=cookies)
                if post_resp.status_code != 200:
                    break
                post_data = post_resp.json()
                if not post_data["post_stream"]["posts"]:
                    break

                for item in post_data["post_stream"]["posts"]:
                    if item["created_at"] > "2025-04-14T00:00:00Z":
                        the_end = True
                        break
                    if "2025-01-01T00:00:00Z" <= item["created_at"] <= "2025-04-14T00:00:00Z":
                        if not item["cooked"]:
                            continue

                        # 1️⃣ Clean the markdown/html to plain text
                        topic_heading = item["cooked"]
                        topic_heading_clean = markdown_to_text(topic_heading).replace("\n", " ")

                        # 2️⃣ Look for an image
                        soup = BeautifulSoup(topic_heading, "html.parser")

                        # find all images, but ignore avatars
                        img_tags = [
                            img for img in soup.find_all("img")
                            if "/uploads/" in img.get("src", "")  # only uploads CDN
                            and "/user_avatar/" not in img.get("src", "")
                        ]

                        if img_tags:
                            img_url = img_tags[0].get("src")
                            ocr_text = ""
                            try:
                                img_resp = requests.get(img_url, timeout=5)
                                image = Image.open(BytesIO(img_resp.content))
                                ocr_text = pytesseract.image_to_string(image).strip()
                            except Exception:
                                ocr_text = ""

                            if ocr_text:
                                topic_heading_clean += f" [Extracted Image Text: {ocr_text}]"
                            else:
                                topic_heading_clean += f" [Image URL: {img_url}]"

                        # 4️⃣ Dump whatever we've got
                        json.dump({
                            "topic_id": topic["id"],
                            "post_url": "https://discourse.onlinedegree.iitm.ac.in" + item["post_url"],
                            "postnumber": item["post_number"],
                            "reply_to_post_number": item["reply_to_post_number"],
                            "username": item["username"],
                            "date": item["created_at"],
                            "topic_content": topic_heading_clean
                        }, file, ensure_ascii=False, indent=4)
                        file.write("\n")

                j += 1
                if the_end:
                    break

        if topic["last_posted_at"] < "2025-01-01T00:00:00Z":
            end_reached = True
            break

    i += 1
    if end_reached:
        break