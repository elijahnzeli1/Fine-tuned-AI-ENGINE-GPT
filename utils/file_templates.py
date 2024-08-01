import os
import json
import random
import string
from typing import Dict, List, Any

TEMPLATES = {
    # Existing templates (Python, JavaScript, Java, Kotlin, Swift, Dart)
    # ... (keep these as they were) ...

    # New and expanded templates
    'typescript': {
        'app.ts': '''
import express from 'express';

const app = express();
const port = 3000;

app.get('/', (req, res) => {{
  res.send('Hello from {project_name}!');
}});

app.listen(port, () => {{
  console.log(`{project_name} app listening at http://localhost:${{port}}`);
}});
'''
    },
    'rust': {
        'main.rs': '''
fn main() {{
    println!("Welcome to {project_name}!");
}}
'''
    },
    'go': {
        'main.go': '''
package main

import "fmt"

func main() {{
    fmt.Println("Welcome to {project_name}!")
}}
'''
    },
    'csharp': {
        'Program.cs': '''
using System;

namespace {project_name}
{{
    class Program
    {{
        static void Main(string[] args)
        {{
            Console.WriteLine("Welcome to {project_name}!");
        }}
    }}
}}
'''
    },
    'ruby': {
        'app.rb': '''
require 'sinatra'

get '/' do
  'Welcome to {project_name}!'
end
'''
    },
    'react': {
        'App.jsx': '''
import React from 'react';

function App() {{
  return (
    <div>
      <h1>Welcome to {project_name}!</h1>
    </div>
  );
}}

export default App;
'''
    },
    'vue': {
        'App.vue': '''
<template>
  <div id="app">
    <h1>{{ message }}</h1>
  </div>
</template>

<script>
export default {{
  name: 'App',
  data() {{
    return {{
      message: 'Welcome to {project_name}!'
    }}
  }}
}}
</script>
'''
    },
    'angular': {
        'app.component.ts': '''
import {{ Component }} from '@angular/core';

@Component({{
  selector: 'app-root',
  template: '<h1>{{{{ title }}}}</h1>'
}})
export class AppComponent {{
  title = 'Welcome to {project_name}!';
}}
'''
    },
    'nextjs': {
        'pages/index.js': '''
export default function Home() {{
  return <h1>Welcome to {project_name}!</h1>
}}
'''
    },
    'express': {
        'server.js': '''
const express = require('express');
const app = express();
const port = 3000;

app.get('/', (req, res) => {{
  res.send('Welcome to {project_name}!');
}});

app.listen(port, () => {{
  console.log(`{project_name} app listening at http://localhost:${{port}}`);
}});
'''
    },
    'godot': {
        'Main.gd': '''
extends Node

func _ready():
    print("Welcome to {project_name}!")
'''
    },
    'unreal': {
        'MyGameMode.h': '''
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "MyGameMode.generated.h"

UCLASS()
class {PROJECT_NAME_UPPER}_API AMyGameMode : public AGameModeBase
{{
    GENERATED_BODY()

public:
    AMyGameMode();

    virtual void StartPlay() override;
}};
''',
        'MyGameMode.cpp': '''
#include "MyGameMode.h"
#include "Engine/Engine.h"

AMyGameMode::AMyGameMode()
{{
}}

void AMyGameMode::StartPlay()
{{
    Super::StartPlay();

    if (GEngine)
    {{
        GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Yellow, TEXT("Welcome to {project_name}!"));
    }}
}}
'''
    },
    'pytorch': {
        'model.py': '''
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Create an instance of the model
model = SimpleNet()
print(model)
'''
    },
    'tensorflow': {
        'model.py': '''
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
print(model.summary())
'''
    }
}

def suggest_stack(description: str) -> Dict[str, List[str]]:
    description = description.lower()
    stack = {
        "frontend": [],
        "backend": [],
        "database": [],
        "mobile": [],
        "game": [],
        "ml": []
    }

    # Frontend suggestions
    if "web" in description or "frontend" in description:
        stack["frontend"] = ["JavaScript", "TypeScript", "React", "Vue", "Angular"]
        if "static" in description or "fast" in description:
            stack["frontend"].append("Next.js")

    # Backend suggestions
    if "backend" in description or "server" in description:
        stack["backend"] = ["Python", "JavaScript", "TypeScript", "Java", "C#", "Go", "Ruby"]
        if "microservices" in description or "scalable" in description:
            stack["backend"].extend(["Node.js with Express", "Go"])
        if "enterprise" in description:
            stack["backend"].extend(["Java", "C#"])

    # Database suggestions
    if "database" in description or "data" in description:
        stack["database"] = ["PostgreSQL", "MySQL", "MongoDB"]
        if "big data" in description or "analytics" in description:
            stack["database"].extend(["Cassandra", "Hadoop"])

    # Mobile suggestions
    if "mobile" in description or "app" in description:
        stack["mobile"] = ["React Native", "Flutter", "Swift (iOS)", "Kotlin (Android)"]

    # Game development suggestions
    if "game" in description:
        stack["game"] = ["Unity (C#)", "Unreal Engine (C++)", "Godot"]

    # Machine Learning suggestions
    if any(term in description for term in ["machine learning", "ml", "ai", "deep learning", "neural network"]):
        stack["ml"] = ["Python", "PyTorch", "TensorFlow", "scikit-learn"]
        if "deep learning" in description or "neural network" in description:
            stack["ml"].extend(["PyTorch", "TensorFlow"])

    return stack

import os
from typing import Dict, Any, List

class ProjectGenerator:
    def __init__(self, project_name: str, project_type: str, description: str):
        self.project_name = project_name
        self.project_type = project_type
        self.description = description
        self.templates = TEMPLATES
        self.suggested_stack = suggest_stack(description)
        self.project_structure = self._generate_project_structure()

    def _generate_project_structure(self) -> Dict[str, Any]:
        if self.project_type == "web_application":
            return {
                "src/": {
                    "frontend/": {
                        "index.html": "<h1>Welcome to {project_name}!</h1>",
                        "styles/": {"main.css": "/* Add your styles here */"},
                        "scripts/": {"main.js": "console.log('Welcome to {project_name}!');"}
                    },
                    "backend/": {
                        "app.py": self.templates['python']['flask_app.py'],
                        "requirements.txt": self.templates['python']['requirements.txt']
                    },
                },
                "tests/": {"test_main.py": "# Add your tests here"},
                "README.md": self.templates['common']['README.md'],
                ".gitignore": self.templates['common']['.gitignore'],
            }
        elif self.project_type == "game":
            return {
                "src/": {
                    "scripts/": {
                        "main.cs": self.templates['csharp']['Program.cs'],
                        "player.cs": self.templates['game']['unity_script.cs']
                    },
                    "assets/": {
                        "sprites/": {},
                        "sounds/": {},
                    }
                },
                "docs/": {"game_design.md": "# Game Design Document"},
                "README.md": self.templates['common']['README.md'],
                ".gitignore": self.templates['common']['.gitignore'],
            }
        elif self.project_type == "mobile":
            return {
                "src/": {
                    "app/": {
                        "main.dart": self.templates['dart']['main.dart'],
                        "views/": {"home.dart": "// Home view goes here"},
                        "models/": {"user.dart": "// User model goes here"},
                    },
                    "assets/": {
                        "images/": {},
                        "fonts/": {},
                    }
                },
                "pubspec.yaml": self.templates['dart']['pubspec.yaml'],
                "README.md": self.templates['common']['README.md'],
                ".gitignore": self.templates['common']['.gitignore'],
            }
        elif self.project_type == "ml":
            return {
                "src/": {
                    "data/": {"dataset.csv": "# Your dataset goes here"},
                    "models/": {"model.py": self.templates['pytorch']['model.py']},
                    "utils/": {"data_loader.py": "# Data loading utilities"},
                    "train.py": "# Training script",
                    "predict.py": "# Prediction script",
                },
                "notebooks/": {"exploratory_analysis.ipynb": "# Jupyter notebook for data analysis"},
                "requirements.txt": "numpy\npandas\nscikit-learn\ntorch\nmatplotlib",
                "README.md": self.templates['common']['README.md'],
                ".gitignore": self.templates['common']['.gitignore'],
            }
        else:
            raise ValueError(f"Unsupported project type: {self.project_type}")

    def generate_files(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self._create_files(self.project_structure, output_dir)

    def _create_files(self, structure: Dict[str, Any], current_dir: str):
        for name, content in structure.items():
            path = os.path.join(current_dir, name)
            if isinstance(content, dict):
                os.makedirs(path, exist_ok=True)
                self._create_files(content, path)
            else:
                with open(path, 'w') as f:
                    rendered_content = self._render_template(content)
                    f.write(rendered_content)

    def _render_template(self, template: str) -> str:
        return template.format(
            project_name=self.project_name,
            class_name=self.project_name.capitalize(),
            project_description=self.description,
            PROJECT_NAME_UPPER=self.project_name.upper()
        )

def suggest_stack(description: str) -> Dict[str, List[str]]:
    description = description.lower()
    stack = {
        "frontend": [],
        "backend": [],
        "database": [],
        "mobile": [],
        "game": [],
        "ml": []
    }

    # Frontend suggestions
    if "web" in description or "frontend" in description:
        stack["frontend"] = ["JavaScript", "TypeScript", "React", "Vue", "Angular"]
        if "static" in description or "fast" in description:
            stack["frontend"].append("Next.js")

    # Backend suggestions
    if "backend" in description or "server" in description:
        stack["backend"] = ["Python", "JavaScript", "TypeScript", "Java", "C#", "Go", "Ruby"]
        if "microservices" in description or "scalable" in description:
            stack["backend"].extend(["Node.js with Express", "Go"])
        if "enterprise" in description:
            stack["backend"].extend(["Java", "C#"])

    # Database suggestions
    if "database" in description or "data" in description:
        stack["database"] = ["PostgreSQL", "MySQL", "MongoDB"]
        if "big data" in description or "analytics" in description:
            stack["database"].extend(["Cassandra", "Hadoop"])

    # Mobile suggestions
    if "mobile" in description or "app" in description:
        stack["mobile"] = ["React Native", "Flutter", "Swift (iOS)", "Kotlin (Android)"]

    # Game development suggestions
    if "game" in description:
        stack["game"] = ["Unity (C#)", "Unreal Engine (C++)", "Godot"]

    # Machine Learning suggestions
    if any(term in description for term in ["machine learning", "ml", "ai", "deep learning", "neural network"]):
        stack["ml"] = ["Python", "PyTorch", "TensorFlow", "scikit-learn"]
        if "deep learning" in description or "neural network" in description:
            stack["ml"].extend(["PyTorch", "TensorFlow"])

    return stack

def generate_project(project_name: str, project_type: str, description: str, output_dir: str):
    generator = ProjectGenerator(project_name, project_type, description)
    generator.generate_files(output_dir)
    print(f"Project '{project_name}' has been generated in '{output_dir}'")
    print("\nSuggested technology stack based on your description:")
    for category, technologies in generator.suggested_stack.items():
        if technologies:
            print(f"{category.capitalize()}: {', '.join(technologies)}")

if __name__ == "__main__":
    project_name = input("Enter project name: ")
    project_type = input("Enter project type (web_application/game/mobile/ml): ")
    description = input("Enter project description: ")
    output_dir = input("Enter output directory: ")
    
    generate_project(project_name, project_type, description, output_dir)