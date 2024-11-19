import os
import requests
import argparse
from xml.etree import ElementTree

# The name of the S3 bucket
bucket_name = 'organellomics'
    
def download_s3_files(local_folder:str)->int:
    """Download files from S3 bucket to a given local folder

    Args:
        local_folder (str): The local folder to download to files to (the destination)

    Returns:
        int: How many files have been successfully downloaded
    """
    # How many files have been successfully downloaded
    downloaded_counter = 0
    
    # Create the local folder if it doesn't exist
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    # URL to list objects in the bucket
    list_url = f"https://{bucket_name}.s3.amazonaws.com/?list-type=2"

    # Send GET request to list all objects in the bucket
    response = requests.get(list_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the XML response
        root = ElementTree.fromstring(response.content)
        
        # Define the namespace to handle the xmlns in the XML
        namespaces = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        
        # Iterate over all Contents elements (files in the folder)
        for content in root.findall('.//s3:Contents', namespaces):
            object_key = content.find('s3:Key', namespaces).text
            file_url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
            
            # Skip folders (they do not have content in the file)
            if object_key.endswith('/'):
                continue
            
            # Construct the local file path based on the object key
            local_file_path = os.path.join(local_folder, object_key)
            
            # Create any necessary subdirectories for the file
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Download the file
            file_response = requests.get(file_url)
            
            if file_response.status_code == 200:
                # Save the file locally
                with open(local_file_path, 'wb') as file:
                    file.write(file_response.content)
                print(f"Downloaded {object_key} to {local_file_path}")
                downloaded_counter += 1
            else:
                print(f"Failed to download {object_key}. Status code: {file_response.status_code}")
    else:
        print(f"Failed to list objects in the bucket. Status code: {response.status_code}")
        
    return downloaded_counter

def main():
    parser = argparse.ArgumentParser(description=f'Download all images from the {bucket_name} S3 bucket to a local folder')
    parser.add_argument('local_folder', type=str, help='The local folder to save downloaded files (destination)')
    args = parser.parse_args()

    print(f"Local folder (Destination): {args.local_folder}")

    downloaded_counter = download_s3_files(args.local_folder)
    print(f"{downloaded_counter} have been downloaded successfully from {bucket_name} S3 bucket to {args.local_folder}")


# Entry point of the script
if __name__ == '__main__':
    main()

# Usage:
# python download_images_from_S3.py path_to_local_folder
