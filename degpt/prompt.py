"""
This script is designed for editing the bare query

bare query - questions or information that inspire
ChatGPT to understand the program deeply instead of
caring about the response of ChatGPT

{
    'type': 'bare_query',
    'name': 'XXX',
    'prompt': {
        'role': 'user',
        'content': 'You should provide programming advice.',
    }
}
"""

import os
import json


BPATH = os.path.dirname(os.path.abspath(__file__))
BPATH = os.path.join(BPATH, 'prompt.json')
print(BPATH)
assert(os.path.exists(BPATH))

with open(BPATH, 'r') as r:
    lst = json.load(r)

while True:
    choice = input('1. list 2. add new 3. keyword-based search 4. delete 5. exit: ')
    if choice == '1':
        print('\n-----------------')
        for _id, _query in enumerate(lst):
            _type = _query['type']
            _name = _type
            if 'name' in _query.keys():
                _name = _query['name']
            print(_id, f': ({_type}) ({_name})', repr(_query['prompt']['content']))
    elif choice == '2':
        _type = input('Input your query type:')
        _name = input('Input your query name (use type as default):')
        if not _name:
            _name = _type
        _query = input('Input your query content:')
        new_entry = {
            'type': _type,
            'name': _name,
            'prompt': {
                'role': 'user',
                'content': _query,
            }
        }
        lst.append(new_entry)
    elif choice == '3':
        k = input('Input your keyword:')
        for _id, _query in enumerate(lst):
            _c = _query['prompt']['content']
            if k in _c:
                print(_id, ':', _c)
    elif choice == '4':
        i = input('Input the id of deleted enetry:')
        lst.pop(int(i))
    elif choice == '5':
        with open(BPATH, 'w') as w:
            json.dump(lst, w, indent=4)
        exit(0)
    else:
        pass
