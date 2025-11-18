import os
import argparse
import sys # Import sys for file operations
from google import genai
from google.genai.errors import APIError

# --- 1. Model Configuration ---
# Define a dictionary mapping friendly aliases to their official model IDs.
MODEL_ALIASES = {
    "flash": "gemini-2.5-flash",
    "pro": "gemini-2.5-pro",
    "flash-lite": "gemini-2.5-flash-lite",
    # Add other models here as needed, like:
    # "embedding": "text-embedding-004",
}

# --- 2. Core Embeddable Function ---
def query_gemini(prompt: str, model_alias: str):
    """
    Queries the Gemini API using a specified model and the final combined prompt.

    Args:
        prompt (str): The final, combined text to send to the model (instruction + file content).
        model_alias (str): The friendly alias of the model to use (e.g., 'pro', 'flash').

    Returns:
        str: The generated text response, or an error message.
    """
    
    # Check if the alias is valid
    if model_alias not in MODEL_ALIASES:
        return f"Error: Invalid model alias '{model_alias}'. Choose from: {list(MODEL_ALIASES.keys())}"
    
    model_id = MODEL_ALIASES[model_alias]
    print(f"--- Querying Model: {model_id} ---")
    
    # Initialization: The genai.Client() automatically looks for the 
    # GEMINI_API_KEY environment variable. This is the recommended secure method.
    try:
        client = genai.Client()
    except Exception as e:
        # This occurs if the API key is missing or invalid
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
            # Add logging for blocked or empty response for debugging
            print(f"DEBUG: Response candidate status: {response.candidates[0].finish_reason if response.candidates else 'No candidates'}", file=sys.stderr)
            return f"Model returned an empty response or was blocked."

    except APIError as e:
        return f"Gemini API Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"



def main():
    # You must install the library first: pip install google-genai
    
    parser = argparse.ArgumentParser(
        description="Run a prompt and optionally a file's content against the Gemini API with model selection."
    )
    
    parser.add_argument(
        "--instruction", 
        type=str, 
        required=True, # Instruction is now mandatory
        help="The specific instruction or question to ask the model about the context (e.g., 'Summarize the Q4 sales data in the file')."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional path to a file (like a CSV) containing the context data. The file content is appended to the instruction."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="flash", # Switched default to 'pro' since file analysis benefits from it
        choices=list(MODEL_ALIASES.keys()),
        help="Model alias to use ('flash', 'pro', or 'flash-lite')."
    )
    
    args = parser.parse_args()
    
    # --- Context Loading and Combination Logic ---
    file_content = ""
    
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            print(f"Successfully loaded context from file: {args.file}", file=sys.stderr)
            final_prompt = f"{args.instruction}\n\n--- FILE CONTENT START ---\n{file_content}\n--- FILE CONTENT END ---"
        except FileNotFoundError:
            print(f"Error: The file '{args.file}' was not found.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file '{args.file}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        final_prompt = args.instruction
    
    # Run the core function
    result = query_gemini(final_prompt, args.model)
    
    # Print the result
    print("\n" + "="*50)
    print("GENERATED RESPONSE")
    print("="*50)
    print(result)
    print("="*50 + "\n")

    if "Client initialization failed" in result:
        print("ACTION REQUIRED: Please set your GEMINI_API_KEY environment variable.")
        return 1
    elif "Model returned an empty response or was blocked." in result:
        print("Model returned an empty response or was blocked.", file=sys.stderr)
        return 1
    elif "Gemini API Error" in result:
        print("Gemini API Error: {result}", file=sys.stderr)
        return 1
    elif "An unexpected error occurred" in result:
        print("An unexpected error occurred: {result}", file=sys.stderr)
        return 1

    return 0

# --- 3. Script Execution (Example of how to run the function) ---
if __name__ == "__main__":
    sys.exit(main())