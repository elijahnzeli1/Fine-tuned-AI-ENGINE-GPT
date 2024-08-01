import ast
import astroid

class CodeParser:
    @staticmethod
    def parse_python(code):
        try:
            tree = ast.parse(code)
            return CodeParser._analyze_ast(tree)
        except SyntaxError:
            return None

    @staticmethod
    def parse_python_advanced(code):
        try:
            tree = astroid.parse(code)
            return CodeParser._analyze_astroid(tree)
        except astroid.exceptions.AstroidSyntaxError:
            return None

    @staticmethod
    def _analyze_ast(tree):
        analysis = {
            'imports': [],
            'functions': [],
            'classes': [],
            'global_variables': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                analysis['imports'].extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                analysis['imports'].append(f"{node.module}.{node.names[0].name}")
            elif isinstance(node, ast.FunctionDef):
                analysis['functions'].append(node.name)
            elif isinstance(node, ast.ClassDef):
                analysis['classes'].append(node.name)
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                analysis['global_variables'].append(node.targets[0].id)
        
        return analysis

    @staticmethod
    def _analyze_astroid(tree):
        analysis = {
            'imports': [],
            'functions': set(),
            'classes': set(),
            'global_variables': set()
        }
        
        for node in tree.walk():
            if isinstance(node, astroid.node_classes.ImportFrom):
                analysis['imports'].append(f"{node.modname}.{node.names[0][0]}")
            elif isinstance(node, astroid.node_classes.Import):
                for name in node.names:
                    analysis['imports'].append(name[0])
            elif isinstance(node, astroid.node_classes.FunctionDef):
                analysis['functions'].add(node.name)
            elif isinstance(node, astroid.node_classes.AssignName):
                analysis['global_variables'].add(node.name)
            elif isinstance(node, astroid.node_classes.ClassDef):
                analysis['classes'].add(node.name)
        
        return analysis
    
    # @staticmethod
    # def _analyze_astroid(tree):
    #     # Implement more advanced analysis using astroid
