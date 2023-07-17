import requests
from bs4 import BeautifulSoup
	
im_class = input("Input class: ")

if im_class == "ideal":
    url = "https://www.google.com/search?q=ideal+smile+dental+image&tbm=isch&ved=2ahUKEwi1rbahpYSAAxUumScCHXGRAtQQ2-cCegQIABAA&oq=ideal+smile+dental+image&gs_lcp=CgNpbWcQA1AAWPcLYIQNaABwAHgCgAHuAogBvA6SAQcwLjUuMi4ymAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=tQ2sZPX-Ga6ynsEP8aKKoA0"

if im_class == "flat":
    url = ""

if im_class == "reversed":
    url = ""

htmldata = requests.get(url).text

def get_url(htmldata):
    urls = []
    soup = BeautifulSoup(htmldata, 'html.parser')
    count = 0
    for item in soup.find_all('img'):
        if count == 0:
            count += 1
            continue
        count += 1
        # print(item['src'])
        urls.append(item["src"])
    return urls
    
url_list = get_url(htmldata)

def download_image(url, save_path):

    response = requests.get(url)

    if response.status_code == 200:
        with open(save_path, "wb") as handle:
            handle.write(response.content)
        print(f"Image downloaded successfully to {save_path}")
    else:
        print("Failed to download the image.")

num = 0
for image_url in url_list:
    save_path = "dataset/{im_class}/{im_class}".format(im_class=im_class) + str(num) + ".jpg"
    download_image(image_url, save_path)
    num += 1