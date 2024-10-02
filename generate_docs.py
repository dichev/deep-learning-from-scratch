import re
from pybars import Compiler as Handlebars
from utils.other import chunk_equal

readme_path = 'README.md'
template_path = 'docs/README.template.hbs'
whitelist = {
    'blocks': [
        'src/lib/layers.py',
        'src/lib/autoencoders.py',
        'src/lib/optimizers.py',
    ],
    'models': [
        'src/models/shallow_models.py',
        'src/models/energy_based_models.py',
        'src/models/recurrent_networks.py',
        'src/models/convolutional_networks.py',
        'src/models/residual_networks.py',
        'src/models/graph_networks.py',
        'src/models/attention_networks.py',
        'src/models/transformer_networks.py',
        'src/models/visual_transformers.py',
        'src/models/diffusion_models.py',
        'src/models/blocks/convolutional_blocks.py',
    ]
}
examples = [
    'examples/',
    'examples/convolutional',
    'examples/energy_based',
    'examples/graph',
    'examples/recurrent',
    'examples/attention',
    'examples/transformer',
    'examples/diffusion',
    'examples/shallow',
]


# Collect and format all the whitelisted classes and functions
sections = {}
citations = {}
for group, paths in whitelist.items():
    modules = []
    for path in paths:
        if '.py' in path:
            module = path.replace('src/', '').replace('/', '.').replace('.py', '')
            items = []
            with open(path, 'r') as file:
                pattern = r'\nclass (\w+).*\n\s+(?:\"\"\"\s+Paper: (.*?)\s+(https?:\S+))?(\"\"\"ignore docs\"\"\")?'
                info = re.findall(pattern, file.read())
                for cls, paper, link, ignore in info:
                    if not ignore:
                        if paper:
                            if paper not in citations:
                                citations[paper] = {'title': paper, 'link': link, 'id': len(citations)+1}
                            items.append({'cls': cls, 'ref': citations[paper]['id'], 'paper': paper})
                        else:
                            items.append({'cls': cls })
            modules.append({'name': module, 'path': path, 'items': items})
    sections[group] = modules



# Render markdown from a handlebars template
with open(template_path, 'r', encoding='utf-8') as f:
    template_file = f.read()
    template = Handlebars().compile(template_file)

for block in sections['blocks']:
    block['items'] = chunk_equal(block['items'], 3)

text = template({
    'blocks': sections['blocks'],
    'models': sections['models'],
    'examples': examples,
    'citations': citations,
})


# Finally write the content
with open(readme_path, 'w', encoding='utf-8') as readme:
    readme.write(text)
    print(f'Document files generated:\n {readme_path}')


