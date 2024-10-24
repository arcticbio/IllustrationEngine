import json
import requests
from langchain_community.llms import Ollama

class BookImageGenerator:
    def __init__(self, json_file_path, output_file_path, paragraphs_per_page=3):
        """
        Initialize the BookImageGenerator with configuration parameters.
       
        Args:
            json_file_path (str): Path to input JSON file
            output_file_path (str): Path for output JSON file
            paragraphs_per_page (int): Number of paragraphs to include per page/image
        """
        self.json_file_path = json_file_path
        self.output_file_path = output_file_path
        self.paragraphs_per_page = paragraphs_per_page
        self.image_generation_url = "" #Set up image generation service (ex. Flux endpoint)
        self.image_generation_headers = {"Content-Type": "application/json"}
        self.ollama = Ollama(base_url='http://localhost:11434', model="mistral-nemo")

    def create_pages(self, paragraphs):
        """
        Group paragraphs into pages.
       
        Args:
            paragraphs (list): List of paragraph dictionaries
           
        Returns:
            list: List of page dictionaries containing grouped paragraphs
        """
        pages = []
        for i in range(0, len(paragraphs), self.paragraphs_per_page):
            page_paragraphs = paragraphs[i:i + self.paragraphs_per_page]
            first_para = page_paragraphs[0]
            pages.append({
                "chapter_number": first_para["chapter_number"],
                "starting_paragraph_number": first_para["paragraph_number"],
                "paragraphs": page_paragraphs
            })
        return pages

    def extract_paragraph_chunks(self, json_data, chunk_size):
        """
        Extract and organize paragraphs into chunks from the JSON data.
       
        Args:
            json_data (dict): Input JSON data
            chunk_size (int): Size of each chunk
           
        Returns:
            list: List of chunk dictionaries
        """
        chunks = []
        for chapter_index, chapter in enumerate(json_data.get("chapters", []), start=1):
            chapter_paragraphs = []
            for paragraph_index, block in enumerate(chapter.get("paragraphs", []), start=1):
                sentences = block.get("sentences", [])
                paragraph_text = " ".join(sentences)
                chapter_paragraphs.append({
                    "chapter_number": chapter_index,
                    "paragraph_number": paragraph_index,
                    "paragraph_text": paragraph_text
                })
           
            # Create pages for this chapter's paragraphs
            pages = self.create_pages(chapter_paragraphs)
            chunks.extend([{
                "chapter_number": chapter_index,
                "pages": pages[i:i + chunk_size]
            } for i in range(0, len(pages), chunk_size)])
        return chunks

    def summarize_chapter(self, chapter_text):
        """Generate chapter summary using Ollama."""
        response = self.ollama.invoke(f"Please summarize the following chapter: {chapter_text}")
        return response.strip()

    def extract_visual_elements(self, text):
        """Extract visual elements from text using Ollama."""
        response = self.ollama.invoke(f"Please extract the primary visual elements of this text: {text}")
        return response.strip()

    def describe_scene(self, page_text, visual_elements, chapter_summary):
        """Generate scene description for a page using Ollama."""
        prompt = (f"Using the following general visual elements extracted from the text: '{visual_elements}', "
                 f"the overall context from this chapter summary: '{chapter_summary}', and the specific story text in this page: '{page_text}', "
                 f"provide a concise visual description of the scene.")
        response = self.ollama.invoke(prompt)
        return response.strip()

    def create_image_prompt(self, scene_description):
        """Create image generation prompt using Ollama."""
        prompt = f"Create a concise image generation prompt based on this scene description; focus on the key elements of the narrative: '{scene_description}'"
        response = self.ollama.invoke(prompt)
        return response.strip()

    def generate_image(self, chapter_number, paragraph_number, image_prompt):
        """Generate image using external API."""
        prompt_prefix = "Create an image inspired by the epic, cinematic style of Peter Jackson's Lord of the Rings trilogy. Use dramatic, sweeping landscapes, with a focus on natural elements such as verdant rolling hills, towering mountains, and mystical forests. Emphasize the contrast between light and shadow to create a sense of depth and atmosphere. Characters should have intricate, detailed costumes with natural, earthy tones and weathered textures. The lighting should evoke the mood of a high-fantasy world, using soft, golden light for tranquil scenes, and darker, more foreboding tones for moments of tension. The overall style should be grounded in realism, but with an otherworldly, mythical quality.  Here is the specific scene to illustrate:"
        conditioned_prompt = f"{prompt_prefix} {image_prompt}"
       
        data = {
            "prompt": conditioned_prompt,
            "chapter": chapter_number,
            "para_idx": paragraph_number,
            "sent_idx": 0
        }

        response = requests.post(
            self.image_generation_url,
            headers=self.image_generation_headers,
            data=json.dumps(data)
        )
       
        return {
            "requested_prompt": conditioned_prompt,
            "status_code": response.status_code,
            "response_json": response.json()
        }

    def process_book(self, chunk_size=50, num_chunks_to_display=1):
        """
        Process the book and generate images for each page.
       
        Args:
            chunk_size (int): Number of pages per chunk
            num_chunks_to_display (int): Number of chunks to process
        """
        # Load JSON data
        with open(self.json_file_path, 'r') as file:
            book_data = json.load(file)

        # Extract and process chunks
        chunks = self.extract_paragraph_chunks(book_data, chunk_size)
        output_data = []

        # Process each chunk
        for chunk_idx, chapter_chunk in enumerate(chunks[:num_chunks_to_display], start=1):
            chapter_number = chapter_chunk["chapter_number"]
            pages = chapter_chunk["pages"]

            # Process each page in the chunk
            for page in pages:
                # Combine all paragraph texts in the page
                page_text = " ".join(p["paragraph_text"] for p in page["paragraphs"])
               
                # Generate chapter summary and visual elements
                chapter_summary = self.summarize_chapter(page_text)
                visual_elements = self.extract_visual_elements(page_text)
               
                # Generate scene description and image prompt
                scene_description = self.describe_scene(page_text, visual_elements, chapter_summary)
                image_prompt = self.create_image_prompt(scene_description)
               
                # Generate image
                image_response = self.generate_image(
                    chapter_number - 1,
                    page["starting_paragraph_number"],
                    image_prompt
                )

                # Store results
                output_data.append({
                    "chapter_number": chapter_number,
                    "starting_paragraph_number": page["starting_paragraph_number"],
                    "paragraphs": [p["paragraph_text"] for p in page["paragraphs"]],
                    "scene_description": scene_description,
                    "image_prompt": image_prompt,
                    "chapter_summary": chapter_summary,
                    "image_generation_response": image_response
                })

        # Save results
        with open(self.output_file_path, 'w') as output_file:
            json.dump(output_data, output_file, indent=4)

        print(f"Processing complete. Output saved to: {self.output_file_path}")

# Update these paths to match your environment
generator = BookImageGenerator(
    json_file_path="path/to/your/hobbit.json",
    output_file_path="path/to/your/output/hobbit_page_chunks_output.json",
    paragraphs_per_page=3
)

# Add progress tracking
from tqdm.notebook import tqdm
from IPython.display import display, JSON

def process_book_with_progress(self, chunk_size=50, num_chunks_to_display=1):
    """
    Process the book and generate images for each page with progress tracking.
    """
    # Load JSON data
    with open(self.json_file_path, 'r') as file:
        book_data = json.load(file)

    # Extract and process chunks
    chunks = self.extract_paragraph_chunks(book_data, chunk_size)
    output_data = []

    # Process each chunk with progress bar
    for chunk_idx, chapter_chunk in enumerate(tqdm(chunks[:num_chunks_to_display], desc="Processing chunks")):
        chapter_number = chapter_chunk["chapter_number"]
        pages = chapter_chunk["pages"]

        # Process each page in the chunk with progress bar
        for page in tqdm(pages, desc=f"Processing pages in chunk {chunk_idx + 1}", leave=False):
            # Combine all paragraph texts in the page
            page_text = " ".join(p["paragraph_text"] for p in page["paragraphs"])
           
            # Generate chapter summary and visual elements
            chapter_summary = self.summarize_chapter(page_text)
            visual_elements = self.extract_visual_elements(page_text)
           
            # Generate scene description and image prompt
            scene_description = self.describe_scene(page_text, visual_elements, chapter_summary)
            image_prompt = self.create_image_prompt(scene_description)
           
            # Generate image
            #image_response = self.generate_image(
                #chapter_number - 1,
                #page["starting_paragraph_number"],
                #image_prompt
            #)

            # Store and display current results
            current_result = {
                "chapter_number": chapter_number,
                "starting_paragraph_number": page["starting_paragraph_number"],
                "paragraphs": [p["paragraph_text"] for p in page["paragraphs"]],
                "scene_description": scene_description,
                "image_prompt": image_prompt,
                "chapter_summary": chapter_summary,
                #"image_generation_response": image_response
            }
           
            output_data.append(current_result)
           
            # Display current result in notebook
            display(JSON(current_result))

    # Save results
    with open(self.output_file_path, 'w') as output_file:
        json.dump(output_data, output_file, indent=4)

    print(f"Processing complete. Output saved to: {self.output_file_path}")

# Add the new method to the class
BookImageGenerator.process_book_with_progress = process_book_with_progress

# Example: Try different page sizes
generator = BookImageGenerator(
    json_file_path="/ReadToCoop-master/server/books/hobbit.json",
    output_file_path="/ReadToCoop-master/server/books/hobbit_paragraph_chunks_output.json",
    paragraphs_per_page=15
)

generator.process_book_with_progress(
    chunk_size=60,
    num_chunks_to_display=1
)
