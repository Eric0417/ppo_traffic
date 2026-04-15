import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_m3u8_links(url):
    headers = {'User-Agent': 'Mozilla/5.0'} # Pretend to be a browser
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Check for errors
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    m3u8_links = set() # Use a set to avoid duplicates

    # Look for <a> tags and <video>/<source> tags
    for tag in soup.find_all(['a', 'video', 'source', 'iframe']):
        link = tag.get('href') or tag.get('src')
        
        if link and '.m3u8' in link:
            # Convert relative paths to absolute URLs
            full_url = urljoin(url, link)
            m3u8_links.add(full_url)

    return list(m3u8_links)

# Usage
target_site = "https://www.dsat.gov.mo/dsat/realtime.aspx" 
links = get_m3u8_links(target_site)

print(f"Found {len(links)} .m3u8 links:")
for l in links:
    print(l)