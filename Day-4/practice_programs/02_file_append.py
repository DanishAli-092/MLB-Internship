
#02 - Append new data to an existing file.


def append_to_file(filename, new_line):
    
    # 'a' mode adds to the end of file without erasing existing content
    with open(filename, "a") as f:
        
        
        f.write(new_line + "\n")
        
    print(f"Appended: '{new_line}'")
    


def read_and_display(filename):
    
    with open(filename, "r") as f:
        
        print("\n--- current file Contnt--- ---")
        
        print(f.read())


if __name__ == "__main__":
    
    filename = "notes.txt"

    # Make sure file exists first (run 01_file_write_read.py before this)
    
    append_to_file(filename, "Appending is done using mode 'a'")
    
    append_to_file(filename, "Multiple appends just keep adding lines")

    read_and_display(filename)