import ast
import json
import re
import sys

def check_code(text,tests):
    text=re.sub(r'^```(?:python)?\s*|\s*```$','',text.strip())
    tree=ast.parse(text)
    # 本地回归题只需要纯列表/整数操作；拒绝 I/O、导入和属性访问。
    forbidden=(ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal,ast.While)
    safe_methods = {'append','extend','sort','reverse','count','index','pop','copy'}
    if any(isinstance(n,forbidden) or isinstance(n,ast.Name) and n.id.startswith('_')
           or isinstance(n,ast.Attribute) and n.attr not in safe_methods for n in ast.walk(tree)):
        return False
    allowed={'abs':abs,'sum':sum,'sorted':sorted,'set':set,'len':len,'range':range,'min':min,'max':max,
             'list':list,'enumerate':enumerate,'zip':zip,'int':int,'reversed':reversed}
    # 限制源码结构，且仅执行本项目固定生成题，避免运行任意模块初始化代码。
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!='solve':return False
    if tree.body[0].decorator_list or tree.body[0].args.defaults:return False
    namespace={'__builtins__':allowed}
    exec(compile(tree,'<qwen38-fixed-code-regression>','exec'),namespace)
    for t in tests:
        actual = namespace['solve'](t['input'].copy())
        if type(actual) is not type(t['expected']) or actual != t['expected']: return False
    return True


if __name__ == '__main__':
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256*1024**2, 256*1024**2))
    value=json.load(sys.stdin)
    try:
        okay=check_code(value['text'], value['tests'])
    except Exception:
        okay=False
    print(json.dumps({'passed':okay}))
