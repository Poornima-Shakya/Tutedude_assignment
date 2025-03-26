

# Employee Management System (EMS)
# define the main menu 

def menu():
    
    while True:
        print("Employee Management System")
        print("1. Add Employee")
        print("2. View All Employees")
        print("3. Search for Employee")
        print("4. Exit")
        choice = input("Enter your choice: ")
        
        if choice == '1':
            add_employee()
        elif choice == '2':
            view_employees()
        elif choice == '3':
            search_employee()
        elif choice == '4':
            print("Thank you for using the Employee Management System. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

#  Initialize The  Employee Dictionary

employees = {
    101: {'name': 'Sayam', 'age': 27, 'department': 'HR', 'salary': 50000},
    102: {'name': 'Rajiv', 'age': 30, 'department': 'IT', 'salary': 70000},
    103: {'name': 'Abhi', 'age': 34, 'department': 'HR', 'salary': 80000},
    104: {'name': 'Raj', 'age': 29, 'department': 'IT', 'salary': 60000}
}
# add the add_employee function to add the employee details

def add_employee():
    while True:
        try:
            emp_id = int(input("Enter the Employee ID: "))
            if emp_id in employees:
                print("Employee ID already exists. Please enter a unique ID.")
                continue
            name = input("Enter Employee Name: ")
            age = int(input("Enter Employee Age: "))
            department = input("Enter Employee Department: ")
            salary = float(input("Enter Employee Salary: "))
            
            employees[emp_id] = {'name': name, 'age': age, 'department': department, 'salary': salary}
            print("Employee added successfully!")
            break
        except ValueError:
            print("Invalid input. Please enter the  correct details.")
            
# view the employees details

def view_employees():
    if not employees:
        print("No employees available.")
        return
    
    print("\n Employee Details:")
    print("ID   |   Name    | Age | Department | Salary")
    print("------------------------------------------------")
    for emp_id, details in employees.items():
        
        print(f"{emp_id:<4} | {details['name']:<9} | {details['age']:<3} | {details['department']:<10} | {details['salary']:<7}")
        
# search the employee

def search_employee():
    
    try:
        emp_id = int(input("Enter the  Employee ID to search: "))
        if emp_id in employees:
            details = employees[emp_id]
            print("\nEmployee Found:")
            print(f"ID: {emp_id}\n Name: {details['name']}\n Age: {details['age']}\n Department: {details['department']}\n Salary: {details['salary']}")
        else:
            print("Employee not found.")
    except ValueError:
        print("Invalid input. Please enter a valid Employee ID.")

# Run the program

if __name__ == "__main__":
    menu()
