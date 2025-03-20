import os

def read_files_into_memory(root_dir):
    file_contents = {}
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    file_contents[file_path] = content
            except Exception as e:
                file_contents[file_path] = f"Could not read file: {e}"
    return file_contents

# Function to retrieve content based on filename
def get_file_content(file_contents, filename_query):
    matched_files = {k: v for k, v in file_contents.items() if filename_query in k}
    return matched_files

if __name__ == "__main__":
    current_directory = os.getcwd()
    file_contents = read_files_into_memory(current_directory)
    
    # Example usage:
    query = input("Enter the file name you want to query: ")
    results = get_file_content(file_contents, query)
    
    if results:
        for path, content in results.items():
            print(f"\n--- Content from {path} ---\n")
            print(content)
    else:
        print("No files found matching the query.")

