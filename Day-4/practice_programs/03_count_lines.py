
#03 - Count the number of lines in a file.


def count_lines(filename):
    
    with open(filename, "r") as f:
        
        lines = f.readlines()
        
    return len(lines)


if __name__ == "__main__":
    
    filename = "notes.txt"
    
    try:
        
        total = count_lines(filename)
        
        print(f"Total lines in '{filename}': {total}")
        
    except FileNotFoundError:
        
        print(f"Error: '{filename}' does not exist. Run 01_file_write_read.py first.")