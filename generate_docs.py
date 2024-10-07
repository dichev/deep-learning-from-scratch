import re
import os
from pybars import Compiler as Handlebars
from utils.other import chunk_equal

readme_path = 'README.md'
template_path = 'docs/README.template.hbs'
images_dir = 'docs/images/'
whitelist = {
    'blocks': [
        'src/lib/layers.py',
        'src/lib/autoencoders.py',
        'src/lib/optimizers.py',
        'src/lib/regularizers.py',
    ],
    'models': [
        'src/models/shallow_models.py',
        'src/models/energy_based_models.py',
        'src/models/recurrent_networks.py',
        'src/models/convolutional_networks.py',
        'src/models/residual_networks.py',
        'src/models/blocks/convolutional_blocks.py',
        'src/models/graph_networks.py',
        'src/models/attention_networks.py',
        'src/models/transformer_networks.py',
        'src/models/visual_transformers.py',
        'src/models/diffusion_models.py',
    ],
    'examples': [
        'examples/shallow/README.md',
        'examples/energy_based/README.md',
        'examples/recurrent/README.md',
        'examples/convolutional/README.md',
        'examples/graph/README.md',
        'examples/attention/README.md',
        'examples/transformers/README.md',
        'examples/diffusion/README.md',
    ]
}



# Collect and format all the whitelisted classes and functions
sections = {}
citations = {}
all_images = [f for f in os.listdir(images_dir) if f.endswith('.png')]
for group, paths in whitelist.items():
    modules = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as file:
            items = []

            if '.py' in path:
                pattern = r'\n(?:class|def) (\w+).*\n\s+(?:\"\"\"\s+Paper: (.*?)\s+(https?:\S+))?(\"\"\"ignore docs\"\"\")?'
                info = re.findall(pattern, file.read())
                for cls, paper, link, ignore in info:
                    if not ignore:
                        if paper:
                            if paper not in citations:
                                citations[paper] = {'title': paper, 'link': link, 'id': len(citations)+1}
                            items.append({'cls': cls, 'ref': citations[paper]['id'], 'paper': paper})
                        else:
                            items.append({'cls': cls })
                module = path.replace('src/', '').replace('/', '.').replace('.py', '')
                modules.append({'name': module, 'path': path, 'items': items})

            elif '.md' in path:
                pattern = r'<h3>(.*?)</h3>'
                matches = re.findall(pattern, file.read())
                items.extend(matches)
                link = path.replace('/README.md', '')
                category = path.split('/')[1]
                images =  [img for img in all_images if img.startswith(category)]
                modules.append({'name': link, 'path': link, 'items': items, 'images': images })

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
    'examples': sections['examples'],
    'citations': citations,
})


# Finally write the content
with open(readme_path, 'w', encoding='utf-8') as readme:
    readme.write(text)
    print(f'Document files generated:\n {readme_path}')


