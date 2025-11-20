import os
from sklearn.datasets import fetch_20newsgroups
import time

def download_and_save_data(output_dir="data/docs", num_docs=200):
    print(f"Downloading 20 Newsgroups dataset (this may take a moment)...")
    try:
        # Try to download a smaller subset first if possible, or just the train set
        dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Dataset downloaded. Saving {num_docs} documents to {output_dir}...")
    
    count = 0
    for i, text in enumerate(dataset.data):
        if count >= num_docs:
            break
        
        if len(text.strip()) < 50: # Skip very short texts
            continue
            
        filename = os.path.join(output_dir, f"doc_{count:03d}.txt")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
            count += 1
            if count % 50 == 0:
                print(f"Saved {count} documents...")
        except Exception as e:
            print(f"Error saving file {filename}: {e}")
        
    print(f"Successfully saved {count} documents.")

if __name__ == "__main__":
    download_and_save_data()
