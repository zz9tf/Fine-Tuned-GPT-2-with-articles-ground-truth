from database import Database

help_menu = """
/help                  Show this help message
/update                Update indexes as config or create them if they don't exist
/show                  Show all index IDs
/use [index_id]        Use the specified index
/stop [index_id]       Stop using the specified index
/delete [index_id]     Delete the specified index
/clear                 Delete the entire database
/bye                   Exit the application
"""

class SystemInterface():
    def __init__(self, config_path='./code/llamaIndex/config.yaml'):
        self.database = Database(config_path)
        self.indexes = [self.database.load_index()]
    
    def check_input(self, user_input):
        if len(user_input.split()) != 2:
            print("Invaild command. Too many parameters.")
            print(help_menu)

    def create_or_update_index_as_config(self):
        self.database.create_or_update_indexes()

    def show_index(self):
        pass

    def use_index(self, index_id):
        pass

    def stop_index(self, index_id):
        pass

    def delete_index(self, index_id):
        pass

    def clear_database(self):
        pass
        
    def query(self, user_input):
        pass

    def run(self):
        print("Welcome to the Glyco chat!")
        print(help_menu)

        while True:
            user_input = input("------------------------------------------\nType a command or '/help' to see the menu:\n>>> ")
            if user_input == "/help":
                print(help_menu)

            elif user_input == "/update":
                self.create_index(user_input.split()[1])

            elif user_input == "/show":
                self.show_index()

            elif user_input.split()[0] == "/use":
                self.check_input(user_input)
                self.use_index(user_input.split()[1])

            elif user_input.split()[0] == "/stop":
                self.check_input(user_input)
                self.stop_index(user_input.split()[1])

            elif user_input.split()[0] == "/delete":
                self.check_input(user_input)
                self.delete_index(user_input.split()[1])

            elif user_input == "/clear":
                self.clear_database()

            elif user_input == "/bye":
                print("Goodbye!")
                break

            else:
                self.query(user_input)

if __name__ == "__main__":
    SystemInterface(config_path='./code/llamaIndex/config.yaml').run()
