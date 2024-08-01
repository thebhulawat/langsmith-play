from typing import TypedDict, List, Optional

# TypedDict example
class Person(TypedDict):
    name: str
    age: int
    hobbies: List[str]

# Regular class example
class Car:
    def __init__(self, make: str, model: str, year: int):
        self.make = make
        self.model = model
        self.year = year

    def description(self) -> str:
        return f"{self.year} {self.make} {self.model}"

# Function using TypedDict
def greet_person(person: Person) -> str:
    return f"Hello, {person['name']}! You are {person['age']} years old."

# Function using regular class
def car_info(car: Car) -> str:
    return f"Car: {car.description()}"

# Optional type hint example
def find_hobby(person: Person, index: int) -> Optional[str]:
    if 0 <= index < len(person['hobbies']):
        return person['hobbies'][index]
    return None

# Usage examples
if __name__ == "__main__":
    # Using TypedDict
    alice: Person = {
        "name": "Alice",
        "age": 30,
        "hobbies": ["reading", "hiking", "photography"]
    }
    
    print(greet_person(alice))
    print(f"Alice's first hobby: {find_hobby(alice, 0)}")

    # Using regular class
    my_car = Car("Toyota", "Corolla", 2022)
    print(car_info(my_car))

    # Demonstrating type checking
    # Uncomment the following lines to see type checking errors
    # bob: Person = {
    #     "name": "Bob",
    #     "age": "thirty",  # Type error: should be int
    #     "hobbies": ["gaming"]
    # }
    
    # invalid_car = Car("Honda", "Civic", "2023")  # Type error: year should be int