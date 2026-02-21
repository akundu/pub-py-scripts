import argparse
import sys
import re
from pathlib import Path
import google.genai as genai
from google.genai.errors import APIError

# Try to import PDF reading libraries
try:
    import PyPDF2
    PDF_LIBRARY = "PyPDF2"
except ImportError:
    try:
        import pdfplumber
        PDF_LIBRARY = "pdfplumber"
    except ImportError:
        PDF_LIBRARY = None

# Try to import DOCX reading library
try:
    from docx import Document
    DOCX_LIBRARY = "python-docx"
except ImportError:
    DOCX_LIBRARY = None

# --- 1. Model Configuration ---
MODEL_ALIASES = {
    "flash": "gemini-2.5-flash",
    "pro": "gemini-2.5-pro",
    "flash-lite": "gemini-2.5-flash-lite",
    "gemini-3": "gemini-3-pro-preview"
}

# --- 2. PDF Reading Functions ---
def read_pdf_pypdf2(file_path: str) -> str:
    """Read PDF using PyPDF2 library."""
    text_content = []
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}\n")
        return "\n".join(text_content)
    except Exception as e:
        raise Exception(f"Error reading PDF with PyPDF2: {e}")

def read_pdf_pdfplumber(file_path: str) -> str:
    """Read PDF using pdfplumber library."""
    text_content = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}\n")
        return "\n".join(text_content)
    except Exception as e:
        raise Exception(f"Error reading PDF with pdfplumber: {e}")

def read_pdf(file_path: str) -> str:
    """Read text content from a PDF file."""
    if PDF_LIBRARY is None:
        raise Exception(
            "No PDF library available. Please install one:\n"
            "  pip install PyPDF2\n"
            "  OR\n"
            "  pip install pdfplumber"
        )
    
    if PDF_LIBRARY == "PyPDF2":
        return read_pdf_pypdf2(file_path)
    elif PDF_LIBRARY == "pdfplumber":
        return read_pdf_pdfplumber(file_path)
    else:
        raise Exception(f"Unknown PDF library: {PDF_LIBRARY}")

def read_docx(file_path: str) -> str:
    """Read text content from a DOCX file."""
    if DOCX_LIBRARY is None:
        raise Exception(
            "No DOCX library available. Please install:\n"
            "  pip install python-docx"
        )
    
    try:
        doc = Document(file_path)
        text_content = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        return "\n".join(text_content)
    except Exception as e:
        raise Exception(f"Error reading DOCX with python-docx: {e}")

# --- 2.5. Directory File Finding Functions ---
def find_resume_file(directory: str) -> str:
    """
    Find a resume file in the directory by searching for files with 'resume' in the name.
    
    Args:
        directory: Path to the directory to search
        
    Returns:
        str: Path to the resume file
        
    Raises:
        FileNotFoundError: If no resume file is found
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Search for files with 'resume' in the name (case-insensitive)
    resume_patterns = [
        "*resume*.pdf",
        "*resume*.PDF",
        "*Resume*.pdf",
        "*RESUME*.pdf",
        "*cv*.pdf",
        "*CV*.pdf"
    ]
    
    for pattern in resume_patterns:
        matches = list(directory_path.glob(pattern))
        if matches:
            # Prefer exact "resume" matches over "cv" matches
            exact_matches = [m for m in matches if 'resume' in m.name.lower()]
            if exact_matches:
                return str(exact_matches[0])
            return str(matches[0])
    
    raise FileNotFoundError(f"No resume file found in directory: {directory}")

def find_output_txt(directory: str) -> str:
    """
    Find output.txt file in the directory.
    
    Args:
        directory: Path to the directory to search
        
    Returns:
        str: Path to output.txt, or None if not found
    """
    directory_path = Path(directory)
    output_file = directory_path / "output.txt"
    if output_file.exists():
        return str(output_file)
    return None

def find_piia_pdf(directory: str) -> str:
    """
    Find a PDF file with 'PIIA' in the name.
    
    Args:
        directory: Path to the directory to search
        
    Returns:
        str: Path to the PIIA PDF file, or None if not found
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        return None
    
    # Search for PDF files with 'PIIA' in the name (case-insensitive)
    piia_patterns = [
        "*PIIA*.pdf",
        "*piia*.pdf",
        "*PIIA*.PDF",
        "*Piia*.pdf"
    ]
    
    for pattern in piia_patterns:
        matches = list(directory_path.glob(pattern))
        if matches:
            return str(matches[0])
    
    return None

def find_docx_file(directory: str) -> str:
    """
    Find a .docx file in the directory.
    
    Args:
        directory: Path to the directory to search
        
    Returns:
        str: Path to the .docx file, or None if not found
    """
    directory_path = Path(directory)
    if not directory_path.is_dir():
        return None
    
    # Search for .docx files (case-insensitive)
    docx_patterns = [
        "*.docx",
        "*.DOCX",
        "*.Docx"
    ]
    
    for pattern in docx_patterns:
        matches = list(directory_path.glob(pattern))
        if matches:
            return str(matches[0])
    
    return None

# --- 3. Core Gemini Query Function ---
def query_gemini(prompt: str, model_alias: str):
    """
    Queries the Gemini API using a specified model and the final combined prompt.

    Args:
        prompt (str): The final, combined text to send to the model.
        model_alias (str): The friendly alias of the model to use (e.g., 'pro', 'flash').

    Returns:
        str: The generated text response, or an error message.
    """
    
    # Check if the alias is valid
    if model_alias not in MODEL_ALIASES:
        return f"Error: Invalid model alias '{model_alias}'. Choose from: {list(MODEL_ALIASES.keys())}"
    
    model_id = MODEL_ALIASES[model_alias]
    print(f"--- Querying Model: {model_id} ---", file=sys.stderr)
    
    # Initialization: The genai.Client() automatically looks for the 
    # GEMINI_API_KEY environment variable.
    try:
        client = genai.Client()
    except Exception as e:
        return f"Client initialization failed. Make sure your API key is set in the GEMINI_API_KEY environment variable.\nDetails: {e}"

    try:
        # Call the API using the selected model
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt]
        )
        
        # Check for generated content
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            return response.text
        else:
            finish_reason = response.candidates[0].finish_reason if response.candidates else 'No candidates'
            print(f"DEBUG: Response candidate status: {finish_reason}", file=sys.stderr)
            return f"Model returned an empty response or was blocked. Finish reason: {finish_reason}"

    except APIError as e:
        return f"Gemini API Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- 3.5. Output Parsing Functions ---
def extract_sections(result: str) -> tuple:
    """
    Extract the SKILLSETS and JOB DESCRIPTION sections from the Gemini response.
    
    Args:
        result: The full response text from Gemini
        
    Returns:
        tuple: (skillsets_text, job_description_text)
    """
    skillsets_text = ""
    job_description_text = ""
    
    # Try to extract UNIQUE SKILLSETS section
    skillsets_patterns = [
        r"###\s*UNIQUE\s*SKILLSETS\s*\n(.*?)(?=###|$)",
        r"##\s*UNIQUE\s*SKILLSETS\s*\n(.*?)(?=##|$)",
        r"UNIQUE\s*SKILLSETS\s*\n(.*?)(?=JOB|STRATEGIC|ADDITIONAL|$)",
    ]
    
    for pattern in skillsets_patterns:
        match = re.search(pattern, result, re.IGNORECASE | re.DOTALL)
        if match:
            skillsets_text = match.group(1).strip()
            break
    
    # Try to extract JOB DESCRIPTION section
    job_desc_patterns = [
        r"###\s*JOB\s*DESCRIPTION\s*\n(.*?)(?=###|$)",
        r"##\s*JOB\s*DESCRIPTION\s*\n(.*?)(?=##|$)",
        r"JOB\s*DESCRIPTION\s*\n(.*?)(?=STRATEGIC|ADDITIONAL|$)",
    ]
    
    for pattern in job_desc_patterns:
        match = re.search(pattern, result, re.IGNORECASE | re.DOTALL)
        if match:
            job_description_text = match.group(1).strip()
            break
    
    return skillsets_text, job_description_text

def save_structured_output(directory: str, skillsets: str, job_description: str):
    """
    Save the extracted sections to separate files in the directory.
    
    Args:
        directory: Directory path where files should be saved
        skillsets: The SKILLSETS section text
        job_description: The JOB DESCRIPTION section text
    """
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    
    # Save SKILLSETS
    if skillsets:
        skillsets_file = directory_path / "SKILLSETS.txt"
        with open(skillsets_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("UNIQUE SKILLSETS\n")
            f.write("="*70 + "\n\n")
            f.write(skillsets)
            f.write("\n" + "="*70 + "\n")
        print(f"SKILLSETS saved to: {skillsets_file}", file=sys.stderr)
    else:
        print("Warning: Could not extract SKILLSETS section from response", file=sys.stderr)
    
    # Save JOB DESCRIPTION
    if job_description:
        job_desc_file = directory_path / "JOB_DESCRIPTION.txt"
        with open(job_desc_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("JOB DESCRIPTION\n")
            f.write("="*70 + "\n\n")
            f.write(job_description)
            f.write("\n" + "="*70 + "\n")
        print(f"JOB DESCRIPTION saved to: {job_desc_file}", file=sys.stderr)
    else:
        print("Warning: Could not extract JOB DESCRIPTION section from response", file=sys.stderr)

# --- 4. PERM Application Prompt Builder ---
def build_perm_prompt(linkedin_url: str, additional_text: str, resume_text: str, piia_text: str = None, starting_job_description: str = None) -> str:
    """
    Builds a comprehensive prompt for PERM application analysis.
    
    Args:
        linkedin_url: LinkedIn profile URL
        additional_text: Additional text from file
        resume_text: Resume text extracted from PDF
        piia_text: Optional PIIA document text
        starting_job_description: Optional starting job description from .docx file
    
    Returns:
        str: Complete prompt for Gemini
    """
    
    piia_section = ""
    if piia_text:
        piia_section = f"""
### PIIA Document Content:
{piia_text}
"""
    
    starting_job_desc_section = ""
    if starting_job_description:
        starting_job_desc_section = f"""
### Starting Job Description (Base Template):
{starting_job_description}

**IMPORTANT**: Use the above job description as your starting point and foundation. You should:
- Preserve the structure, format, and core content of the starting job description
- Enhance it a lot by incorporating the candidate's unique skillsets, qualifications, and background. So that it can make the job description more specific, detailed and tailored to the candidate.
- Add specific details that make this candidate uniquely qualified for the position
- Ensure the final job description reflects both the base requirements AND the candidate's distinctive attributes
- Make the job description more specific and detailed while maintaining the original framework
"""
    
    prompt = """You are an expert immigration attorney specializing in PERM (Program Electronic Review Management) labor certification applications for Green Card sponsorship.

Your task is to analyze the candidate's professional profile and create a comprehensive PERM application strategy.
Make the unique skillsets, Job Description/Job Requirements as specific and detailed as possible.

## INPUT INFORMATION:

### LinkedIn Profile:
{linkedin_url}

### Additional Information:
{additional_text}

### Resume Content:
{resume_text}
{piia_section}{starting_job_desc_section}
## YOUR ANALYSIS SHOULD INCLUDE:

1. **UNIQUE SKILLSETS IDENTIFICATION**
   - Identify the candidate's unique technical skills, certifications, and specialized knowledge
   - Highlight combinations of skills that are rare or in high demand
   - Note any domain-specific expertise (e.g., specific industries, technologies, methodologies)
   - Identify any unique educational background or training

2. **JOB REQUIREMENTS FRAMING**
   - Based on the candidate's unique profile, craft specific job requirements that:
     * Accurately reflect the candidate's qualifications
     * Are specific enough to demonstrate the need for this particular candidate
     * Follow PERM requirements (must be legitimate job requirements, not tailored to exclude U.S. workers)
     * Highlight the unique combination of skills that make this candidate suitable
   
3. **JOB DESCRIPTION STRUCTURE**
   - Create a detailed job description that:
     * Incorporates the unique skillsets identified
     * Uses industry-standard terminology
     * Clearly articulates why this specific combination of skills is necessary
     * Includes minimum requirements (education, experience, specific skills)
     * Describes job duties that align with the candidate's background
{job_desc_instruction}

4. **STRATEGIC RECOMMENDATIONS**
   - Provide recommendations on how to frame the application to maximize approval chances
   - Suggest any additional documentation or evidence that might strengthen the case
   - Identify potential challenges and how to address them
   - Note any areas where the job requirements might need adjustment to be more defensible

## OUTPUT FORMAT:

Please provide your analysis in the following structured format:

### UNIQUE SKILLSETS
[List the candidate's unique skillsets with explanations]

### JOB REQUIREMENTS FRAMING
[Detailed job requirements that incorporate the unique skillsets]

### JOB DESCRIPTION
[Complete job description for the PERM application]

### STRATEGIC RECOMMENDATIONS
[Recommendations for strengthening the application]

### ADDITIONAL NOTES
[Any other relevant observations or suggestions]

---

Please be thorough, specific, and ensure all recommendations comply with PERM regulations and Department of Labor requirements.
"""
    
    job_desc_instruction = ""
    if starting_job_description:
        job_desc_instruction = """
     * **STARTING POINT**: Use the provided starting job description as the base template
     * **ENHANCEMENT**: Add unique candidate-specific details, skills, and qualifications to the base template
     * **PRESERVATION**: Maintain the original structure and core requirements while enhancing with candidate-specific attributes
"""
    else:
        job_desc_instruction = ""
    
    return prompt.format(
        linkedin_url=linkedin_url,
        additional_text=additional_text if additional_text else "No additional information provided.",
        resume_text=resume_text,
        piia_section=piia_section if piia_text else "",
        starting_job_desc_section=starting_job_desc_section if starting_job_description else "",
        job_desc_instruction=job_desc_instruction
    )

# --- 5. Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Analyze candidate profile for PERM Green Card application. "
                    "Can work in directory mode (--directory) or individual file mode."
    )
    
    # Directory mode (new)
    parser.add_argument(
        "--directory",
        type=str,
        default=None,
        help="Path to directory containing resume PDF, output.txt, and PIIA PDF. "
             "When provided, automatically finds these files in the directory."
    )
    
    # Individual file mode (original, for backward compatibility)
    parser.add_argument(
        "--linkedin",
        type=str,
        default=None,
        help="LinkedIn profile URL of the candidate (required if not using --directory)"
    )
    
    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Path to a text file containing additional information about the candidate"
    )
    
    parser.add_argument(
        "--resume-pdf",
        type=str,
        default=None,
        help="Path to the candidate's resume PDF file (required if not using --directory)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3",
        choices=list(MODEL_ALIASES.keys()),
        help="Model alias to use (default: gemini-3)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the full output to a file (in addition to stdout). "
             "In directory mode, structured output is automatically saved to the directory."
    )
    
    args = parser.parse_args()
    
    # Validate PDF library availability
    if PDF_LIBRARY is None:
        print("ERROR: No PDF reading library found.", file=sys.stderr)
        print("Please install one of the following:", file=sys.stderr)
        print("  pip install PyPDF2", file=sys.stderr)
        print("  OR", file=sys.stderr)
        print("  pip install pdfplumber", file=sys.stderr)
        return 1
    
    # Note: DOCX library is optional, will warn if needed
    
    # Determine mode: directory mode or individual file mode
    directory_mode = args.directory is not None
    
    if directory_mode:
        # Directory mode: find files automatically
        directory = args.directory
        print(f"Working in directory mode: {directory}", file=sys.stderr)
        
        # Initialize optional file paths
        docx_path = None
        
        # Find resume file
        resume_pdf_path = None
        try:
            resume_pdf_path = find_resume_file(directory)
            print(f"Found resume: {resume_pdf_path}", file=sys.stderr)
        except FileNotFoundError as e:
            print(f"Warning: {e}", file=sys.stderr)
            print("Continuing without resume file...", file=sys.stderr)
        
        # Find output.txt
        output_txt_path = find_output_txt(directory)
        if output_txt_path:
            print(f"Found additional info file: {output_txt_path}", file=sys.stderr)
        else:
            print("No output.txt found in directory (optional)", file=sys.stderr)
        
        # Find PIIA PDF
        piia_pdf_path = find_piia_pdf(directory)
        if piia_pdf_path:
            print(f"Found PIIA PDF: {piia_pdf_path}", file=sys.stderr)
        else:
            print("No PIIA PDF found in directory (optional)", file=sys.stderr)
        
        # Find DOCX file (starting job description)
        docx_path = find_docx_file(directory)
        if docx_path:
            print(f"Found starting job description DOCX: {docx_path}", file=sys.stderr)
        else:
            print("No .docx file found in directory (optional)", file=sys.stderr)
        
        # Read files
        additional_text = ""
        if output_txt_path:
            try:
                with open(output_txt_path, 'r', encoding='utf-8') as f:
                    additional_text = f.read()
                print(f"Successfully loaded additional information from: {output_txt_path}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error reading '{output_txt_path}': {e}", file=sys.stderr)
        
        # Read resume PDF if found
        resume_text = ""
        if resume_pdf_path:
            print(f"Reading resume PDF: {resume_pdf_path}", file=sys.stderr)
            try:
                resume_text = read_pdf(resume_pdf_path)
                if not resume_text.strip():
                    print("Warning: The PDF appears to be empty or could not be read.", file=sys.stderr)
                    resume_text = "No resume content available."
                else:
                    print(f"Successfully extracted text from PDF ({len(resume_text)} characters)", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error reading PDF '{resume_pdf_path}': {e}", file=sys.stderr)
                resume_text = "No resume content available."
        else:
            resume_text = "No resume file found in directory."
        
        # Read PIIA PDF if found
        piia_text = None
        if piia_pdf_path:
            print(f"Reading PIIA PDF: {piia_pdf_path}", file=sys.stderr)
            try:
                piia_text = read_pdf(piia_pdf_path)
                if not piia_text.strip():
                    print("Warning: The PIIA PDF appears to be empty or could not be read.", file=sys.stderr)
                else:
                    print(f"Successfully extracted text from PIIA PDF ({len(piia_text)} characters)", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error reading PIIA PDF '{piia_pdf_path}': {e}", file=sys.stderr)
        
        # Read DOCX file if found (starting job description)
        starting_job_description = None
        if docx_path:
            print(f"Reading starting job description DOCX: {docx_path}", file=sys.stderr)
            try:
                if DOCX_LIBRARY is None:
                    print("Warning: python-docx library not available. Install with: pip install python-docx", file=sys.stderr)
                else:
                    starting_job_description = read_docx(docx_path)
                    if not starting_job_description.strip():
                        print("Warning: The DOCX file appears to be empty or could not be read.", file=sys.stderr)
                    else:
                        print(f"Successfully extracted text from DOCX ({len(starting_job_description)} characters)", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Error reading DOCX file '{docx_path}': {e}", file=sys.stderr)
        
        # For directory mode, LinkedIn URL is optional (can be in output.txt or not needed)
        linkedin_url = args.linkedin if args.linkedin else "Not provided (check output.txt for LinkedIn information)"
        
    else:
        # Individual file mode (original behavior)
        if not args.linkedin:
            print("Error: --linkedin is required when not using --directory mode", file=sys.stderr)
            return 1
        if not args.resume_pdf:
            print("Error: --resume-pdf is required when not using --directory mode", file=sys.stderr)
            return 1
        
        linkedin_url = args.linkedin
        
        # Read additional text file if provided
        additional_text = ""
        if args.text_file:
            try:
                with open(args.text_file, 'r', encoding='utf-8') as f:
                    additional_text = f.read()
                print(f"Successfully loaded additional information from: {args.text_file}", file=sys.stderr)
            except FileNotFoundError:
                print(f"Error: The file '{args.text_file}' was not found.", file=sys.stderr)
                return 1
            except Exception as e:
                print(f"Error reading file '{args.text_file}': {e}", file=sys.stderr)
                return 1
        
        # Read PDF resume
        print(f"Reading resume PDF: {args.resume_pdf}", file=sys.stderr)
        try:
            resume_text = read_pdf(args.resume_pdf)
            if not resume_text.strip():
                print("Warning: The PDF appears to be empty or could not be read.", file=sys.stderr)
            else:
                print(f"Successfully extracted text from PDF ({len(resume_text)} characters)", file=sys.stderr)
        except Exception as e:
            print(f"Error reading PDF '{args.resume_pdf}': {e}", file=sys.stderr)
            return 1
        
        piia_text = None
        starting_job_description = None
    
    # Build the prompt
    print("Building PERM application analysis prompt...", file=sys.stderr)
    prompt = build_perm_prompt(linkedin_url, additional_text, resume_text, piia_text, starting_job_description)
    
    # Query Gemini
    print("Querying Gemini API...", file=sys.stderr)
    result = query_gemini(prompt, args.model)
    
    # Check for errors
    if "Client initialization failed" in result:
        print("ACTION REQUIRED: Please set your GEMINI_API_KEY environment variable.", file=sys.stderr)
        return 1
    elif "Error:" in result and ("Invalid model alias" in result or "Gemini API Error" in result):
        print(result, file=sys.stderr)
        return 1
    elif "Model returned an empty response" in result:
        print(result, file=sys.stderr)
        return 1
    elif "An unexpected error occurred" in result:
        print(result, file=sys.stderr)
        return 1
    
    # Print the result
    print("\n" + "="*70, file=sys.stdout)
    print("PERM APPLICATION ANALYSIS", file=sys.stdout)
    print("="*70, file=sys.stdout)
    print(result, file=sys.stdout)
    print("="*70 + "\n", file=sys.stdout)
    
    # Extract and save structured output if in directory mode
    if directory_mode:
        print("Extracting structured sections from response...", file=sys.stderr)
        skillsets, job_description = extract_sections(result)
        save_structured_output(directory, skillsets, job_description)
    
    # Save full output to file if requested
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("PERM APPLICATION ANALYSIS\n")
                f.write("="*70 + "\n")
                f.write(f"LinkedIn URL: {linkedin_url}\n")
                if directory_mode:
                    f.write(f"Directory: {directory}\n")
                    if resume_pdf_path:
                        f.write(f"Resume PDF: {resume_pdf_path}\n")
                    else:
                        f.write(f"Resume PDF: Not found\n")
                    if output_txt_path:
                        f.write(f"Additional Info File: {output_txt_path}\n")
                    if piia_pdf_path:
                        f.write(f"PIIA PDF: {piia_pdf_path}\n")
                    if docx_path:
                        f.write(f"Starting Job Description DOCX: {docx_path}\n")
                else:
                    f.write(f"Resume PDF: {args.resume_pdf}\n")
                    if args.text_file:
                        f.write(f"Additional Info File: {args.text_file}\n")
                f.write("="*70 + "\n\n")
                f.write(result)
                f.write("\n" + "="*70 + "\n")
            print(f"Full output also saved to: {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not save output to file '{args.output}': {e}", file=sys.stderr)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

