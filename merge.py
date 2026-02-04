
import os

def combine_python_files_non_recursive(input_folder, output_file_name="combined_python_code.py", ext=".py"):
    """
    Reads all .py files directly in a specified folder (non-recursively)
    and combines their content into a single output file.
    Args:
        input_folder (str): The path to the folder containing the .py files.
        output_file_name (str): The name of the output file where combined code will be saved.
    """
    combined_content = []
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    # List all entries in the given folder
    for entry_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, entry_name)
        
        # Check if it's a file and ends with .py
        if os.path.isfile(file_path) and entry_name.endswith(ext):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    combined_content.append(f"# --- Content from {file_path} ---\n")
                    combined_content.append(f.read())
                    combined_content.append("\n\n") # Add some separation between files
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    if not combined_content:
        print(f"No .py files found directly in '{input_folder}'.")
        return
    try:
        with open(output_file_name, 'w', encoding='utf-8') as outfile:
            outfile.writelines(combined_content)
        print(f"All .py files combined successfully into '{output_file_name}'.")
    except Exception as e:
        print(f"Error writing to output file '{output_file_name}': {e}")

def combine_python_files(input_folder, output_file_name="combined_python_code.py"):
    """
    Reads all .py files in a specified folder and combines their content
    into a single output file.

    Args:
        input_folder (str): The path to the folder containing the .py files.
        output_file_name (str): The name of the output file where combined code will be saved.
    """
    combined_content = []
    
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        combined_content.append(f"# --- Content from {file_path} ---\n")
                        combined_content.append(f.read())
                        combined_content.append("\n\n") # Add some separation between files
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if not combined_content:
        print(f"No .py files found in '{input_folder}'.")
        return

    try:
        with open(output_file_name, 'w', encoding='utf-8') as outfile:
            outfile.writelines(combined_content)
        print(f"All .py files combined successfully into '{output_file_name}'.")
    except Exception as e:
        print(f"Error writing to output file '{output_file_name}': {e}")

# Example usage:
# Create a dummy folder and some files for demonstration
# os.makedirs("my_python_folder", exist_ok=True)
# with open("my_python_folder/file1.py", "w") as f:
#     f.write("print('Hello from file1')\ndef func1(): pass")
# with open("my_python_folder/file2.py", "w") as f:
#     f.write("import sys\n# Some comment\nprint('Hello from file2')")
# with open("my_python_folder/test.txt", "w") as f:
#     f.write("This is not a python file.")

# Uncomment the line below to run the example
# combine_python_files("my_python_folder")

# To use it, replace "my_python_folder" with the path to your desired folder.
# You can also specify a different output file name:
# combine_python_files("path/to/your/folder", "my_combined_code.py")

if __name__ == "__main__":
    combine_python_files_non_recursive("C:/research/SemanticNormalForms","allCode.txt", ".py")
    combine_python_files_non_recursive("C:/research/SemanticNormalForms/data","allFilesInData.txt", ".json")