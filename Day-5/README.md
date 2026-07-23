# Day 5 — Object-Oriented Programming (OOP)

This folder covers Day 5 of the internship: learning OOP fundamentals through small practice scripts, then applying them in a working mini project — a console-based Library Management System that saves its data to a JSON file.

## Folder layout

```
Day-5/
├── 01_practice/
│   ├── 01_oop_basics.py       -> Student, Employee, Car classes
│   └── 02_inheritance.py      -> Person, Student, Teacher (inheritance + overriding)
├── 02_library_management_system/
│   ├── 03_library_system.py   -> Book and Library classes, main project
│   └── library_data.json      -> generated automatically, stores book records
└── README.md
```

## Running it

Practice scripts:
```
cd 01_practice
python 01_oop_basics.py
python 02_inheritance.py
```

The library system:
```
cd 02_library_management_system
python 03_library_system.py
```

Menu options once it's running: add a book, view all books, search, borrow, return, or exit.

---

## OOP, explained the way I'd explain it to myself before I understood it

Before OOP, most of the code I wrote was just variables and functions sitting next to each other, connected only because they happened to be in the same file. OOP changes that — it lets you group a piece of data with the functions that operate on that data, and call the whole bundle an object.

A concrete example from this project: a book isn't just a `title` string floating around, and `borrow` isn't a random standalone function either. They belong together, because "borrowing" only makes sense in the context of a specific book's data (its ID, whether it's already taken, etc). OOP is what lets me write `book.borrow()` instead of `borrow(book_id, all_books_list)`.

This structure is also why frameworks like PyTorch or OpenCV are built the way they are — a model, a layer, an image — all of these are objects with their own data and their own behavior bundled together, not loose functions passing data back and forth.

## Class vs Object — the distinction that trips people up first

A **class** doesn't hold any real data by itself — it's a definition of what a thing should look like once it exists. An **object** is what you get after that definition is actually used to create something.

```python
class Book:
    def __init__(self, book_id, title, author, is_borrowed=False):
        self.book_id = book_id
        self.title = title
        self.author = author
        self.is_borrowed = is_borrowed
```

Nothing exists yet at this point — this is just a definition sitting in memory. The moment I write:

```python
b1 = Book("B001", "Atomic Habits", "James Clear")
```

that's when an actual object gets created, with its own real values. I can create as many of these as I want from the same class definition — that's the whole point of writing it once.

## Attributes and Methods

Attributes hold the object's state — for `Book`, that's `book_id`, `title`, `author`, `is_borrowed`. Methods are the behavior tied to that object — in this project, that's things like `to_dict()`, `from_dict()`, and the `__str__()` that controls how a book prints:

```python
def __str__(self):
    status = "Borrowed" if self.is_borrowed else "Available"
    return f"[{self.book_id}] {self.title} by {self.author} - {status}"
```

`print(b1)` triggers this automatically and gives something like `[B001] Atomic Habits by James Clear - Available`.

## The constructor — `__init__`

`__init__` is not something I call myself — Python runs it automatically the instant an object is created, and its whole job is to set that object's starting attributes. Without it, I'd have to create an empty object and then manually assign every attribute one line at a time, which is exactly what a constructor exists to avoid.

```python
def __init__(self, book_id, title, author, is_borrowed=False):
    self.book_id = book_id
    self.title = title
    self.author = author
    self.is_borrowed = is_borrowed
```

Note `is_borrowed=False` — a default value, so every new book starts as available unless told otherwise.

## Making more than one object

Objects made from the same class are completely independent of each other — changing one has zero effect on the others, because each one gets its own separate copy of the attributes:

```python
b1 = Book("B001", "The Hobbit", "Tolkien")
b2 = Book("B002", "Dune", "Herbert")
```

Marking `b1` as borrowed doesn't touch `b2` at all. The `Library` class keeps every object it manages inside one list:

```python
self.books = []
```

The same idea is what `01_oop_basics.py` walks through directly — building several `Student`, `Employee`, and `Car` objects from a single class each, and looping over them.

## `self` — what it actually refers to

`self` refers to whichever object a method is currently being run on. It's not a special keyword with hidden magic — Python passes the object in automatically as the first argument every time you call a method on it.

```python
def borrow_book(self, book_id):
    book = self.find_book(book_id)
    if book:
        book.is_borrowed = True
```

`library.borrow_book("B001")` — Python quietly turns this into "run `borrow_book`, with `self` set to `library`." That's why `self` shows up as the first parameter in every method: it's how the method knows which object's data to work on, especially once there are multiple objects of the same class floating around.

---

## Inheritance

Inheritance exists so a new class can reuse everything an existing class already has, and only add or change the parts that are actually different — instead of rewriting the same attributes and methods twice.

This project practices it with `Person`, `Student`, and `Teacher` in `02_inheritance.py`. Both students and teachers are still people first — they both have a name and an age — so it doesn't make sense to define `name` and `age` separately in each class.

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"Hi, I am {self.name}.")


class Student(Person):
    def __init__(self, name, age, roll_no):
        super().__init__(name, age)
        self.roll_no = roll_no

    def introduce(self):
        print(f"Hi, I am {self.name}, a student with roll number {self.roll_no}.")
```

`Student(Person)` is what establishes the inheritance — `Student` now has access to everything `Person` defines. `super().__init__(name, age)` runs the parent's constructor first, so I don't have to re-write the logic for storing `name` and `age` — I just add the one new thing `Student` needs (`roll_no`).

`Teacher` follows the same pattern, adding `subject` instead of `roll_no`.

### Overriding

Both `Student` and `Teacher` redefine `introduce()` even though `Person` already has one. This is overriding — the child class keeps the same method name but supplies its own version, so calling `introduce()` on a `Student` object behaves differently than calling it on a plain `Person` or a `Teacher`.

### Why the library system itself has no inheritance

I considered whether `Book` needed a subclass (something like a separate class for e-books or reference-only books), but decided against forcing it in. Every book in this system is treated identically — there's no behavioral difference between them that would justify a parent/child split. Adding one just to demonstrate inheritance again would have been inheritance for its own sake rather than for a real reason, so I kept it isolated to the `Person`/`Student`/`Teacher` practice file where it's actually solving a duplication problem.

## Encapsulation

Encapsulation is about not letting code outside a class casually reach in and change that class's data directly. Instead, changes go through methods that can validate the change first.

In the library system, a book is never modified like this from outside the class:
```python
book.is_borrowed = True   # skips any validation
```

It always goes through:
```python
library.borrow_book(book_id)
```

which checks first whether the book exists and whether it's already borrowed, and only then updates the state. If I ever add something like due dates or borrow limits later, that logic only needs to change in one place — inside the method  rather than everywhere the attribute happens to be touched directly.

The `Book` class also keeps its own JSON-conversion logic self-contained through `to_dict()` and `from_dict()`, so `Library` never needs to know the internal structure of a book to save or load it — it just calls those two methods.

---

## Problems I actually ran into

**Objects turned into plain dictionaries after loading from JSON, and my code broke.**
JSON has no concept of a `Book` object — it only understands plain text, numbers, lists, and dictionaries. So when `library_data.json` gets loaded back with `json.load()`, what comes back is a list of dictionaries, not a list of `Book` objects. My first version of `load_books()` just did `self.books = json.load(f)` directly, and every method that called something like `book.title` or `book.is_borrowed` immediately crashed with `AttributeError: 'dict' object has no attribute 'title'`, because a dictionary uses `book["title"]`, not `book.title`.
Fixed it by writing `from_dict()` as a `@staticmethod` on the `Book` class, so loading always rebuilds real `Book` objects from the raw dictionaries, and `to_dict()` does the reverse when saving. The rest of the program only ever deals with actual `Book` objects, never raw dictionaries.

**Duplicate book IDs would silently overwrite each other.**
Early on, `add_book()` just appended a new `Book` straight into the list without checking anything first. If I entered a book ID that already existed, the library ended up with two separate `Book` objects sharing the same ID — which broke `borrow_book()` and `return_book()`, since `find_book()` always grabs the *first* match and had no way of knowing which of the two was meant.
Fixed it by making `add_book()` call `find_book()` first and reject the operation with an error message if that ID is already taken, so every book ID in the system stays unique.

**Wrong shell for the command I was using.** I tried creating empty files with `type nul > filename.py`, which is CMD syntax — but my terminal was PowerShell, so it kept failing with `Cannot find path '...\nul'`. Switched to PowerShell's `New-Item filename.py`, which worked immediately.

**Folder order in VS Code wasn't what I expected.** I wanted `01_practice` listed before `02_library_management_system` in the sidebar, but VS Code sorts folders alphabetically regardless of the order they were created in — so `library_management_system` kept showing up first. Renaming both folders with number prefixes fixed it, since digits sort ahead of letters.

**Lost track of my working directory mid-navigation.** I ran `cd ..` one extra time by mistake, which took the terminal all the way up to the Desktop instead of staying inside `Day-5`. That caused a `Cannot find path` error, and then `python` couldn't locate the file I was trying to run. Running `pwd` to check exactly where I was, then navigating back down with the correct relative path, fixed it.

**`self` didn't make sense for a while.** It looked like unnecessary repetition in every method definition until I understood it wasn't optional boilerplate — it's literally how a method knows which object it's currently operating on, which matters the moment there's more than one object of the same class.

---

## Checklist against the task requirements

- Book and Library represented as separate classes
- Inheritance applied where there was an actual duplication problem to solve (Person → Student, Teacher), not forced into the library system
- Multiple independent objects created and tested in every practice file
- Method overriding shown through `introduce()`
- Book data persisted across runs via JSON
- Invalid menu input and missing/corrupt JSON files handled without crashing