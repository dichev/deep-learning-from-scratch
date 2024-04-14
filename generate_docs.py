import re

readme_path = 'README.md'
marker_start = '<!-- auto-generated-start -->'
marker_end = '<!-- auto-generated-end -->'
whitelist = {
    'Layers': [
        'src/lib/layers.py',
        'src/lib/autoencoders.py',
    ],
    'Optimizers': [
        'src/lib/optimizers.py',
    ],
    'Models / Networks': [
        'src/models/shallow_models.py',
        'src/models/energy_based_models.py',
        'src/models/recurrent_networks.py',
        'src/models/convolutional_networks.py',
        'src/models/residual_networks.py',
        'src/models/graph_networks.py',
        'src/models/attention_networks.py',
        'src/models/transformer_networks.py',
        'src/models/blocks/convolutional_blocks.py',
    ],
    "Example usages": [
        'examples/',
        'examples/convolutional',
        'examples/energy_based',
        'examples/graph',
        'examples/recurrent',
        'examples/attention',
        'examples/transformer',
        'examples/shallow',
    ]
}


# Find the marker from where the writing begins
with open(readme_path, 'r') as file:
    content = file.read()
    start, end = content.find(marker_start), content.find(marker_end)
    assert start != -1 and end != -1, f'The markers are not found: {marker_start=}, {marker_end=}. Aborting!'
    header, text, footer = content[:start + len(marker_start)], '', content[end:]


# Collect and format all the whitelisted classes and functions
for group, paths in whitelist.items():
    text += f'\n\n### {group}\n'
    for path in paths:
        if '.py' in path:
            module = path.replace('src/', '').replace('/', '.').replace('.py', '')
            text += f'\n`{module}` [➜]({path})\n'
            with open(path, 'r') as file:
                pattern = r'\nclass (\w+).*\n\s+(?:"""\s+Paper: (.*?)\s+(https?:\S+))?'
                info = re.findall(pattern, file.read())
                for cls, paper, link in info:
                    text += f'- {cls}'
                    if paper:
                        text += f' <sup>[*[{paper}]*]({link})</sup>'
                    text += '\n'
        else:
            text += f'- {path} [➜]({path})\n'


# Finally write the content
with open(readme_path, 'w', encoding='utf-8') as readme:
    readme.write(header + '\n' + text + '\n' + footer)
    print(f'Document files generated:\n {readme_path}')

