import json

import os


# ============ Book Class ============
class Book:
    def __init__(self, book_id, title, author, is_borrowed=False):
        
        self.book_id = book_id
        
        self.title = title
        
        self.author = author
        
        self.is_borrowed = is_borrowed

    def to_dict(self):
        
        # converting book object into dictionary so we can save it in json
        return {
            
            "book_id": self.book_id,
            
            "title": self.title,
            
            "author": self.author,
            
            "is_borrowed": self.is_borrowed
        }

    @staticmethod
    def from_dict(data):
        
        # creating a book object back from dictionary (used when loading json)
        
        return Book(data["book_id"], data["title"], data["author"], data
                    ["is_borrowed"])

    def __str__(self):
        
        status = "Borrowed" if self.is_borrowed else "Available"
        
        return f"[{self.book_id}] {self.title} by {self.author} - {status}"
    


# ============ Library Class ============
class Library:
    def __init__(self, data_file="library_data.json"):
        
        self.data_file = data_file
        
        self.books = []
        
        self.load_books()

    def load_books(self):
        
        
        # load books from json file if it exists, otherwise start with empty list
        
        if os.path.exists(self.data_file):
            try:
                
                with open(self.data_file, "r") as f:
                    
                    data = json.load(f)
                    
                    self.books = [Book.from_dict(b) for b in data]
                    
            except (json.JSONDecodeError, FileNotFoundError):
                
                self.books = []
        else:
            self.books = []

    def save_books(self):
        # save all books back to json file
        
        
        with open(self.data_file, "w") as f:
            
            json.dump([book.to_dict() for book in self.books], f, indent=4)
            

    def add_book(self, book_id, title, author):
        
        if self.find_book(book_id):
            
            print("This book id already exists, try a different one.")
            
            return
        new_book = Book(book_id, title, author)
        
        self.books.append(new_book)
        
        self.save_books()
        
        print(f"Great! '{title}' has been added to the library.")

    def view_books(self):
        
        if not self.books:
            print("Library is empty right now.")
            
            return
        
        for book in self.books:
            
            print(book)

    def find_book(self, book_id):
        
        for book in self.books:
            
            if book.book_id == book_id:
                
                return book
            
        return None

    def search_book(self, keyword):
        
        results = [b for b in self.books
                   
                   if keyword.lower() in b.title.lower() or keyword.lower() in b.author.lower()]
        if not results:
            
            print("Couldn't find any book matching that.")
            
        else:
            for book in results:
                print(book)

    def borrow_book(self, book_id):
        
        book = self.find_book(book_id)
        if not book:
            print("Book not found in library.")
            return
        
        
        if book.is_borrowed:
            print("Sorry, this book is already borrowed.")
            return
        
        
        book.is_borrowed = True
        
        self.save_books()
        print(f"You have borrowed '{book.title}'. Enjoy reading!")

    def return_book(self, book_id):
        
        book = self.find_book(book_id)
        if not book:
            print("Book not found in library.")
            
            return
        if not book.is_borrowed:
            print("This book wasn't borrowed in the first place.")
            
            return
        book.is_borrowed = False
        
        
        self.save_books()
        print(f"Thanks for returning '{book.title}'.")


# ============ Menu / Main Program ============
def show_menu():
    print("\n----- LIBRARY MANAGEMENT SYSTEM -----")
    
    
    print("1. Add a new book")
    print("2. View all books")
    
    print("3. Search for a book")
    
    print("4. Borrow a book")
    
    print("5. Return a book")
    print("6. Exit")


def main():
    
    
    library = Library()

    while True:
        show_menu()
        
        choice = input("Enter your choice (1-6): ").strip()
        

        try:
            
            if choice == "1":
                
                book_id = input("Enter book ID: ").strip()
                title = input("Enter book title: ").strip()
                author = input("Enter author name: ").strip()
                if not book_id or not title or not author:
                    print("Please fill all the fields, nothing can be empty.")
                    continue
                library.add_book(book_id, title, author)

            elif choice == "2":
                library.view_books()

            elif choice == "3":
                
                keyword = input("Enter title or author to search: ").strip()
                
                library.search_book(keyword)

            elif choice == "4":
                book_id = input("Enter book ID to borrow: ").strip()
                
                library.borrow_book(book_id)

            elif choice == "5":
                book_id = input("Enter book ID to return: ").strip()
                
                library.return_book(book_id)

            elif choice == "6":
                print("Thanks for using the library system, goodbye!")
                
                break

            else:
                print("Invalid input, please choose a number from 1 to 6.")

        except Exception as e:
            print(f"Something went wrong: {e}")


if __name__ == "__main__":
    main()