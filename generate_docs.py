import re
from pybars import Compiler as Handlebars
from utils.other import chunk_equal

readme_path = 'README.md'
marker_start = '<!-- auto-generated-start -->'
marker_end = '<!-- auto-generated-end -->'
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



# Find the marker from where the writing begins
with open(readme_path, 'r') as file:
    content = file.read()
    start, end = content.find(marker_start), content.find(marker_end)
    assert start != -1 and end != -1, f'The markers are not found: {marker_start=}, {marker_end=}. Aborting!'
    header, text, footer = content[:start + len(marker_start)], '', content[end:]


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
template = """

### Building blocks

<table>
    {{#each blocks}}
    <tr>
        <td width="300">
            <code><a href="{{path}}">{{name}}</a></code>
        </td>
        {{#each items}}
        <td>
            {{#each this}}
            {{cls}}{{#if paper}}&nbsp;<sup><a href="#ref{{ref}}" title="{{paper}}">{{ref}}</a></sup>{{/if}}<br>
            {{/each}}
        </td>
        {{/each}}
    </tr>
    {{/each}}
</table>



### Models / Networks

<table>
    <tr>
        <th width="300">Family</th>
        <th>Models</th>
    </tr>
    {{#each models}}
    <tr>
        <td>
            <code><a href="{{path}}">{{name}}</a></code>
        </td>
        <td>
            {{#each items}}
            {{cls}}{{#if paper}}&nbsp;<sup><a href="#ref{{ref}}" title="{{paper}}">{{ref}}</a></sup>{{/if}}{{#unless @last}}, {{/unless}}
            {{/each}}
        </td>
    </tr>
    {{/each}}
</table>




### Example usages
{{#each examples}}
- {{this}} [➜]({{this}})
{{/each}}


<hr/>


### References
{{#each citations}}
{{id}}. <a name="ref{{id}}" href="{{link}}">{{title}}</a>
{{/each}}

"""

template = Handlebars().compile(template)

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
    readme.write(header + '\n' + text + '\n' + footer)
    print(f'Document files generated:\n {readme_path}')


