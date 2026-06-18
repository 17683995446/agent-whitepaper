import ast
import os

base = '/workspace/agent-whitepaper/src/agent_whitepaper/stages'
files = [f'stage_{i:03d}.py' for i in range(1, 11)]

for fname in files:
    fpath = os.path.join(base, fname)
    with open(fpath, 'r') as f:
        content = f.read()
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f'{fname}: SYNTAX ERROR - {e}')
        continue
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    methods = []
    for cls in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
        for item in cls.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(f'{cls.name}.{item.name}')
    # Check for markdown residue (triple backticks)
    has_md = '```' in content
    print(f'{fname}: {len(classes)} classes, {len(functions)} module funcs, {len(methods)} methods, markdown_residue={has_md}')
    print(f'  Classes: {classes}')
    if methods:
        print(f'  Methods (first 5): {methods[:5]}{"..." if len(methods)>5 else ""}')
