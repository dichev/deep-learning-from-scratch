import re

readme_path = 'README.md'
marker = '<!-- auto-generated-bellow -->'
whitelist = {
    'Layers': [
        'src/lib/layers.py',
        'src/lib/autoencoders.py',
    ],
    'Optimizers': [
        'src/lib/optimizers.py',
        'src/lib/regularizers.py',
    ],
    'Models / Networks': [
        'src/models/shallow_models.py',
        'src/models/energy_based_models.py',
        'src/models/recurrent_networks.py',
        'src/models/convolutional_networks.py',
        'src/models/residual_networks.py',
        'src/models/blocks/convolutional_blocks.py',
    ],
    "Example usages": [
        'examples/',
        'examples/convolutional',
        'examples/energy_based',
        'examples/recurrent',
        'examples/shallow',
    ]
}


# Find the marker from where the writing begins
with open(readme_path, 'r') as file:
    content = file.read()
    position = content.find(marker)
    text = content[:content.find(marker) + len(marker)]
    assert position != -1, f'The marker: "{marker}" is not found'


# Collect and format all the whitelisted classes and functions
for group, paths in whitelist.items():
    text += f'\n\n### {group}\n'
    for path in paths:
        if '.py' in path:
            module = path.replace('src/', '').replace('/', '.').replace('.py', '')
            text += f'\n`{module}` [➜]({path})\n'
            with open(path, 'r') as file:
                pattern = r'\n(class|def) (\w+).*\n\s+(?:"""\s+Paper: (.*?)\s+(https?:\S+))?'
                info = re.findall(pattern, file.read())
                for _, cls, paper, link in info:
                    text += f'- {cls} ([*{paper}*]({link}))\n' if paper else f'- {cls}\n'
        else:
            text += f'- {path} [➜]({path})\n'


# Finally write the content
with open(readme_path, 'w', encoding='utf-8') as readme:
    readme.write(text)
    print(f'Document files generated:\n {readme_path}')

