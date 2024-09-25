from multiprocessing import Pool
import os
import pandas as pd
import requests
from tqdm import tqdm
import time  # Make sure to import time for the delay

def download_single_image(image_url, download_folder, retries=3, delay=3):
    filename = os.path.join(download_folder, image_url.split("/")[-1])
    
    if os.path.exists(filename):
        return  # Skip downloading if the file already exists

    for attempt in range(retries):
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(filename, 'wb') as out_file:
                    out_file.write(response.content)
                print(f"Successfully downloaded {filename}")
                return
            else:
                print(f"Failed to download {image_url} with status code {response.status_code}")
        except Exception as e:
            print(f"Error downloading {image_url}: {e}")
        
        time.sleep(delay)  # Wait before retrying

    print(f"Giving up on {image_url} after {retries} attempts")

def download_image_wrapper(args):
    image_url, download_folder = args
    download_single_image(image_url, download_folder)

def download_images(image_urls, download_folder, num_workers=60):
    # Create download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Prepare arguments for the pool
    args = [(url, download_folder) for url in image_urls]

    # Use multiprocessing Pool to download images
    with Pool(processes=num_workers) as pool:
        # Use tqdm to show progress
        list(tqdm(pool.imap_unordered(download_image_wrapper, args),
                  total=len(image_urls),
                  desc="Downloading images"))

if __name__ == "__main__":
    # Path to the test dataset file
    test_path = 'C:/Users/divya/OneDrive/Documents/student_resource 3/dataset/test.csv'

    # Load the test dataset
    test_df = pd.read_csv(test_path)

    # Create folder for downloaded images
    image_folder = 'C:/Users/divya/OneDrive/Documents/student_resource 3/images'

    # Download images from the test dataset
    test_images = test_df['image_link'].tolist()

    # Download test images
    print("Downloading test images...")
    download_images(test_images, download_folder=image_folder, num_workers=60)
