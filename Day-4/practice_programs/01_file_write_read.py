
#01 - Create a text file and write data into it then read and display it.


def write_to_file(filename, lines):
    
    # 'w' mode creates the file if it doesn't exist
    with open(filename, "w") as f:
        
        for line in lines:
            
            f.write(line + "\n")
            
    print(f"Data written to {filename}")


def read_and_display(filename):
    
    with open(filename, "r") as f:
        
        content = f.read()
    print("--- file contnts -------------")
    
    print(content)


if __name__ == "__main__":
    
    filename = "notes.txt"
    
    data = [
        "Day 4 of MLB Internship",
        "Learning file handling in Python",
        "with statement auto-closes files"
    ]
    

    write_to_file(filename, data)
    
    read_and_display(filename)