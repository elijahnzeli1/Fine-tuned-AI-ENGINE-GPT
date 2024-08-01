from ast import Dict
import os
from utils.code_parser import CodeParser
from utils.file_templates import get_file_template

class ProjectAssembler:
    def __init__(self, config):
        self.config = config
        self.code_parser = CodeParser()
        self.supported_languages = ['python', 'javascript', 'java', 'kotlin', 'swift', 'dart']

    def assemble_project(self, architecture, generated_code, project_type, language):
        if language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}")

        project_structure = self._create_project_structure(architecture, project_type, language)
        file_contents = self._distribute_code(generated_code, project_structure, language)
        return self._write_project_files(file_contents)

    def _create_project_structure(self, architecture, project_type, language):
        project_structure = {
            'src': {},
            'tests': {},
            'docs': {},
            'config': {},
        }

        if project_type == 'web_development':
            project_structure['src']['frontend'] = {'components': {}, 'pages': {}, 'styles': {}}
            project_structure['src']['backend'] = {'api': {}, 'models': {}, 'services': {}}
        elif project_type == 'mobile_app':
            project_structure['src']['ui'] = {'screens': {}, 'components': {}}
            project_structure['src']['logic'] = {'models': {}, 'services': {}}
        elif project_type == 'data_science':
            project_structure['src']['data'] = {'raw': {}, 'processed': {}}
            project_structure['src']['models'] = {}
            project_structure['notebooks'] = {}
        
        # Add language-specific structures
        if language == 'python':
            project_structure['requirements.txt'] = ''
        elif language in ['javascript', 'typescript']:
            project_structure['package.json'] = ''
            project_structure['node_modules'] = {}
        elif language in ['java', 'kotlin']:
            project_structure['build.gradle'] = ''
            project_structure['src']['main'] = {'java': {}, 'resources': {}}
            project_structure['src']['test'] = {'java': {}}
        
        # Add architecture-specific structures
        for component in architecture['components']:
            component_path = component['path'].split('/')
            current_level = project_structure
            for path_part in component_path:
                if path_part not in current_level:
                    current_level[path_part] = {}
                current_level = current_level[path_part]

        return project_structure

    def _distribute_code(self, generated_code, project_structure, language):
        file_contents = {}
        parsed_code = self.code_parser.parse_code(generated_code, language)

        for item in parsed_code:
            file_path = self._determine_file_path(item, project_structure, language)
            if file_path:
                if file_path not in file_contents:
                    file_contents[file_path] = get_file_template(language, file_path)
                file_contents[file_path] += item['code'] + '\n\n'

        # Add necessary config files
        file_contents.update(self._create_config_files(project_structure, language))

        return file_contents

    def _determine_file_path(self, code_item, project_structure, language):
        if code_item['type'] == 'class':
            return self._find_path_for_class(code_item['name'], project_structure, language)
        elif code_item['type'] == 'function':
            return self._find_path_for_function(code_item['name'], project_structure, language)
        elif code_item['type'] == 'variable':
            return self._find_path_for_variable(code_item['name'], project_structure, language)
        return None

    def _find_path_for_class(self, class_name, project_structure, language):
        # Logic to determine the appropriate file path for a class
        # This would depend on the project structure and naming conventions
        pass

    def _find_path_for_function(self, function_name, project_structure, language):
        # Logic to determine the appropriate file path for a function
        pass

    def _find_path_for_variable(self, variable_name, project_structure, language):
        # Logic to determine the appropriate file path for a variable
        pass

    def _create_config_files(self, project_structure, language):
        config_files = {}
        if language == 'python':
            config_files['requirements.txt'] = self._generate_requirements_txt()
        elif language in ['javascript', 'typescript']:
            config_files['package.json'] = self._generate_package_json()
        elif language in ['java', 'kotlin']:
            config_files['build.gradle'] = self._generate_build_gradle()
        
        # Add other common config files
        config_files['README.md'] = self._generate_readme()
        config_files['.gitignore'] = self._generate_gitignore(language)

        return config_files

    def _generate_requirements_txt(self) -> str:
        """
        Generates the content for the requirements.txt file.

        Returns:
            str: The content for the requirements.txt file.
        """
        packages = set()
        for dependency in self.config['dependencies']:
            packages.add(dependency['name'])
            if 'version' in dependency:
                packages.add(f"{dependency['name']}=={dependency['version']}")
        return '\n'.join(packages) + '\n'



    def _generate_package_json(self) -> Dict[str, str]:
        """
        Generates the content for the package.json file.

        Returns:
            Dict[str, str]: The content for the package.json file.
        """
        # Logic to generate package.json content
        return {}


    def generate_build_gradle(self) -> str:
        """
        Generates the content for the build.gradle file.

        Returns:
            str: The content for the build.gradle file.
        """
        return self._generate_build_gradle_content().lstrip()

    def _generate_build_gradle_content(self):
        """
        Generates the content for the build.gradle file.

        Returns:
            str: The content for the build.gradle file.
        """
        content = []
        if self.config.get('dependencies', []):
            content.append("dependencies {")
            for dependency in self.config['dependencies']:
                if 'version' in dependency:
                    content.append(f"    {dependency['name']}: {dependency['version']}")
                else:
                    content.append(f"    {dependency['name']}")
            content.append("}")
        content.append("}")
        return '\n'.join(content)

    def _generate_readme(self) -> str:
        """
        Generates the content for the README.md file.

        Returns:
            str: The content for the README.md file.
        """
        description = self.config.get('description', '')
        dependencies = self.config.get('dependencies', [])
        readme = f"# {description}\n\n"
        if dependencies:
            readme += "## Dependencies\n\n"
            for dep in dependencies:
                readme += f"- {dep['name']} {dep.get('version', '')}\n"
        return readme


    def _generate_gitignore(self, language):
        """
        Generates the content for the .gitignore file based on the language.

        Args:
            language (str): The language of the project.

        Returns:
            str: The content for the .gitignore file.
        """
        # Determine the .gitignore file content based on the language
        language_to_gitignore = {
            'python': '.venv\n__pycache__\n*.py[co]\n*.egg\n*.egg-info\n*.spec\n*.mo\n*.pot\n*.log\n*.db\n*.sqlite\n*.db-journal\n*.flaskenv\n*.env\n.pytest_cache\n.coverage\n.cache\nnosetests.xml\ncoverage.xml\n*.cache\n*.pyc\n*.pyo\n*.egg-info\n*.mo\n*.pot\n*.log\n*.db\n*.sqlite\n*.db-journal\n*.flaskenv\n*.env\n.pytest_cache\n.coverage\n.cache\nnosetests.xml\ncoverage.xml',
            'javascript': 'node_modules\n.env\n.env.local\n.env.development.local\n.env.production.local\n.env.test.local\nnpm-debug.log\nyarn-error.log\nyarn-debug.log\npids\nlogs\n*.pid\n*.seed\n*.log\n*.crash\n*.pid\n*.log\n*.crash\n*.txt\n*.md\n*.tgz\n*.gz\npackage-lock.json\nyarn.lock',
            'java': 'target\n*.class\n*.jar\n*.war\n*.ear\n*.zip\n*.tar.gz\n*.rar\n*.7z\n*.log\n*.log.?\nlocal.properties\n.settings\n.project\n.target\nbin\nobj\n*.iml\n*.ipr\n*.iws\n*.idea\n*.sublime-project\n*.sublime-workspace\n*.suo\n*.ntvs*\n*.njsproj\n*.sln\n*.csproj\n*.vbproj',
            'kotlin': 'target\n*.class\n*.jar\n*.war\n*.ear\n*.zip\n*.tar.gz\n*.rar\n*.7z\n*.log\n*.log.?\nlocal.properties\n.settings\n.project\n.target\nbin\nobj\n*.iml\n*.ipr\n*.iws\n*.idea\n*.sublime-project\n*.sublime-workspace\n*.suo\n*.ntvs*\n*.njsproj\n*.sln\n*.csproj\n*.vbproj',
            'swift': 'DerivedData\n*.dSYM\n*.xcarchive\n*.xcresult',
            'dart': '.dart_tool\n*.log\n*.pid\n*.pub\nbuild\n.dart_tool\n*.log\n*.pid\n*.pub\nbuild',
        }

        if language in language_to_gitignore:
            return language_to_gitignore[language]
        else:
            return ''
    def render_template(self, template_name, context):
        """
        Renders a template with the given context.

        :param template_name: The name of the template file.
        :param context: A dictionary of variables to pass to the template.
        :return: The rendered template as a string.
        """
        # Assuming templates are stored in a 'templates' directory
        template_path = f"templates/{template_name}.txt"
        try:
            with open(template_path, 'r') as file:
                template = file.read()
                # Simple rendering logic: replace {{ var_name }} with the value from context
                for key, value in context.items():
                    template = template.replace(f"{{{{ {key} }}}}", str(value))
                return template
        except FileNotFoundError:
            return "Template not found."



    def _write_project_files(self, file_contents):
        for file_path, content in file_contents.items():
            full_path = os.path.join(self.config['output_directory'], file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
        return f"Project files written successfully to {self.config['output_directory']}"
