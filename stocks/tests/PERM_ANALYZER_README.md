# PERM Application Analyzer

This program analyzes a candidate's profile (LinkedIn, resume, and additional information) to generate a comprehensive PERM (Program Electronic Review Management) Green Card application strategy.

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install google-genai PyPDF2
   ```
   OR
   ```bash
   pip install google-genai pdfplumber
   ```

2. **Set your Gemini API key:**
   ```bash
   export GEMINI_API_KEY='your-api-key-here'
   ```

## Usage

### Basic Usage

```bash
python tests/perm_application_analyzer.py \
    --linkedin "https://www.linkedin.com/in/candidate-profile" \
    --resume-pdf /path/to/resume.pdf
```

### With Additional Information File

```bash
python tests/perm_application_analyzer.py \
    --linkedin "https://www.linkedin.com/in/candidate-profile" \
    --resume-pdf /path/to/resume.pdf \
    --text-file /path/to/additional_info.txt
```

### Save Output to File

```bash
python tests/perm_application_analyzer.py \
    --linkedin "https://www.linkedin.com/in/candidate-profile" \
    --resume-pdf /path/to/resume.pdf \
    --text-file /path/to/additional_info.txt \
    --output perm_analysis_output.txt
```

### Use Different Model

```bash
python tests/perm_application_analyzer.py \
    --linkedin "https://www.linkedin.com/in/candidate-profile" \
    --resume-pdf /path/to/resume.pdf \
    --model pro
```

Available models: `flash`, `pro`, `flash-lite`, `gemini-3` (default)

## Arguments

- `--linkedin` (required): LinkedIn profile URL of the candidate
- `--resume-pdf` (required): Path to the candidate's resume PDF file
- `--text-file` (optional): Path to a text file with additional candidate information
- `--model` (optional): Gemini model to use (default: `gemini-3`)
- `--output` (optional): Path to save the analysis output to a file

## Output

The program generates a comprehensive analysis including:

1. **Unique Skillsets**: Identifies the candidate's unique technical skills, certifications, and specialized knowledge
2. **Job Requirements Framing**: Crafts specific job requirements that accurately reflect the candidate's qualifications
3. **Job Description**: Creates a detailed job description incorporating the unique skillsets
4. **Strategic Recommendations**: Provides recommendations for maximizing approval chances

## Example Additional Information File

See `example_additional_info.txt` for a template of what to include in the additional information file.

## Notes

- The program automatically detects and uses either PyPDF2 or pdfplumber for PDF reading
- Make sure your PDF is readable (not scanned images without OCR)
- The LinkedIn URL is included in the prompt but the program does not automatically fetch the profile content - you may want to manually add LinkedIn profile details to the additional text file for best results

