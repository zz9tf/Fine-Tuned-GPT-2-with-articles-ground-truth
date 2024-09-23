import os, shutil, yaml, time
from basic.database import Database

help_menu = """
/help                              Show this help message
/update                            Update indexes as config or create them if they don't exist
/show                              Show all index IDs
/use [index_id or llm_name]        Use the specified index or llm
/stop [index_id]                   Stop using the specified index
/delete [index_id]                 Delete the specified index
/clear                             Delete the entire database
/bye                               Exit the application
Enter                              Enter without input to reset config
"""

input_query = """\

------------------------------------------
[Interface] Type a command or '/help' to see the menu:
You are using indexes: {}   llm: {}   ReRank: {}
>>> """

class SystemInterface():
    def __init__(self, config_dir_path):
        self.root_path = '../..'
        self.database = Database(root_path=self.root_path, config_dir_path=config_dir_path)
        self.config_dir_path = config_dir_path
        self._load_configs()
        self.current_index_id = self.config['rag']['default_index']
        if self.config['rag']['default_index'] != None:
            self.engine = self.database.load_index(self.current_index_id, llm=self.config['rag']['llm'], is_rerank=self.config['rag']['isReRank'])
            print("Using default index: {}".format(self.current_index_id))
    
    def _load_configs(self):
        config_path = os.path.abspath(os.path.join(self.root_path, self.config_dir_path, 'config.yaml'))
        with open(config_path, 'r') as config:
            self.config = yaml.safe_load(config)
        prefix_config_path = os.path.abspath(os.path.join(self.root_path, self.config_dir_path, 'prefix_config.yaml'))
        with open(prefix_config_path, 'r') as prefix_config:
            self.prefix_config = yaml.safe_load(prefix_config)
    def check_input(self, user_input, parameter_num):
        if len(user_input.split()) != parameter_num+1:
            print("Invaild command. Require {} parameters".format(parameter_num))
            print(help_menu)
            return False
        return True

    def create_or_update_index_as_config(self):
        self.database.create_or_update_indexes()

    def show_index(self):
        print(f"{'NAME':<20} {'SIZE':<15} {'MODIFIED':<20}")
        for index in self.database.get_all_index_ids():
            print(f"{index['id']:<20} {index['size']:.2f}MB{'':<8} {index['modified_date']:<20}")

    def use_engine(self, index_id_or_llm_name):
        # TODO Multiple indexes loading
        if index_id_or_llm_name in self.prefix_config['llm']:
            self.database._get_llm(index_id_or_llm_name)
            print("[Interface] LLM {} using".format(index_id_or_llm_name))
        else:
            self.current_index_id = index_id_or_llm_name
            self.engine = self.database.load_index(index_id=self.current_index_id, llm_name=self.config['rag']['llm'], is_rerank=self.config['rag']['isReRank'])
            print("[Interface] Index {} loaded".format(index_id_or_llm_name))

    def stop_engine(self, index_id):
        # TODO stop_index
        pass

    def delete_engine(self, index_id):
        indexes_dir_path = os.path.abspath(os.path.join(self.root_path, self.config['indexes_dir_path'], index_id))
        if os.path.exists(indexes_dir_path):
            shutil.rmtree(indexes_dir_path)
        print('[Interface] Deleted {}'.format(index_id))

    def clear_database(self):
        indexes_dir_path = os.path.abspath(os.path.join(self.root_path, self.config['indexes_dir_path']))
        if os.path.exists(indexes_dir_path):
            # Iterate through all the files and directories within the specified directory
            for filename in os.listdir(indexes_dir_path):
                file_path = os.path.join(indexes_dir_path, filename)
                try:
                    # Check if it is a file and remove it
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    # Check if it is a directory and remove it
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        print('[Interface] Cleared')
        
    def query(self, user_input):
        self.engine = self.database.load_index(index_id=self.current_index_id, llm=self.config['rag']['llm'], is_rerank=self.config['rag']['isReRank'])
        print(self.engine.query)
        exit()
        response = self.engine.query(user_input)
        print("[Interface response] ", end="")
        response.print_response_stream()
        
        print("\n")

        print("[Source]")
        for i, node in enumerate(response.source_nodes):
            metadata = "file_name: {} | file_path: {} | page_label: {}".format(node.metadata['file_name'], node.metadata['file_path'], node.metadata['page_label'])
            for char in "{}. {}\n[content (length {})] {}...\n".format(i+1, metadata, len(node.text), node.text[:500]):
                print(char, end='', flush=True)
                time.sleep(0.005)

    def get_source(self):
        # TODO get_source
        pass

    def run(self):
        print("Welcome to the Glyco chat!")
        print(help_menu)

        while True:
            user_input = input(input_query.format(self.current_index_id, self.config['rag']['llm'], self.config['rag']['isReRank']))
            if user_input == "":
                self._load_configs()
                self.database._load_configs()
                continue
            elif user_input == "/help":
                if self.check_input(user_input, 0):
                    print(help_menu)

            elif user_input == "/update":
                if self.check_input(user_input, 0):
                    self.create_or_update_index_as_config()

            elif user_input == "/show":
                if self.check_input(user_input, 0):
                    self.show_index()
            
            elif user_input == "/current":
                if self.check_input(user_input, 0):
                    print("Currently using index: {}".format(self.current_index_id))

            elif user_input.split()[0] == "/use":
                if self.check_input(user_input, 1):
                    self.use_engine(user_input.split()[1])

            elif user_input.split()[0] == "/stop":
                if self.check_input(user_input, 1):
                    self.stop_engine(user_input.split()[1])

            elif user_input.split()[0] == "/delete":
                if self.check_input(user_input, 1):
                    self.delete_engine(user_input.split()[1])

            elif user_input == "/clear":
                if self.check_input(user_input, 0):
                    self.clear_database()
            elif user_input == "/gpt4-output":
                pass

            elif user_input == "/bye":
                if self.check_input(user_input, 0):
                    print("Goodbye!")
                    exit()
            else:
                self.query(user_input)

if __name__ == "__main__":
    SystemInterface(config_dir_path='./code/llamaIndex/configs').run()
