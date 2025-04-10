import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import re

def clean_wiki_text(text):
    """Basic cleaning of wiki markup"""
    if text is None:
        return ""

    # Remove references
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)

    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Remove templates {{...}}
    text = re.sub(r'{{[^{}]*}}', '', text)

    # Remove file/image links [[File:...]] or [[Image:...]]
    text = re.sub(r'\[\[(File|Image):[^\]]*\]\]', '', text)

    # Simplify links [[target|text]] -> text
    text = re.sub(r'\[\[[^|\]]*\|([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)

    # Fix whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def parse_wiki_dump(xml_path, output_dir, max_articles=None):
    """Parse Wikipedia XML dump and extract articles"""
    os.makedirs(output_dir, exist_ok=True)

    # Output files
    articles_file = os.path.join(output_dir, "wiki_articles.txt")

    # Track progress
    article_count = 0

    print(f"Parsing XML file: {xml_path}")
    print(f"Saving articles to: {articles_file}")

    # Open output file
    with open(articles_file, 'w', encoding='utf-8') as out_file:
        # Set up for iterative parsing
        context = ET.iterparse(xml_path, events=('end',))

        # Track current article
        current_title = None
        is_redirect = False

        # Process elements
        for event, elem in tqdm(context, desc="Parsing XML"):
            tag = elem.tag.split('}')[-1]  # Handle namespace

            # Extract title
            if tag == 'title':
                current_title = elem.text

            # Check if it's a redirect
            elif tag == 'redirect':
                is_redirect = True

            # Process text content
            elif tag == 'text' and current_title and not is_redirect:
                if elem.text:
                    # Clean and save the text
                    clean_text = clean_wiki_text(elem.text)

                    # Skip stub articles and very short text
                    if len(clean_text) > 200:
                        out_file.write(f"# {current_title}\n\n")
                        out_file.write(clean_text)
                        out_file.write("\n\n<|endoftext|>\n\n")

                        article_count += 1

                        # Check if we've reached the maximum
                        if max_articles and article_count >= max_articles:
                            print(f"Reached maximum of {max_articles} articles")
                            break

            # Clear current article at the end of a page
            elif tag == 'page':
                current_title = None
                is_redirect = False

            # Clear element to save memory
            elem.clear()

    print(f"Extracted {article_count} articles")

if __name__ == "__main__":
    xml_path = "data/simplewiki-latest.xml"  # Path to your uncompressed XML file
    output_dir = "processed_data"

    # Start with a small number for testing
    parse_wiki_dump(xml_path, output_dir, max_articles=10000)
