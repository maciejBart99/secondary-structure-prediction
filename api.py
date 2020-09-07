from flask import Flask, request, send_file
from capt import explain_model
from test import main

app = Flask('explain-model')


@app.route('/sequence/result')
def result():
    loss, acc, res = main('model_q8.pth', False)

    return {
        'loss': loss,
        'acc': acc,
        'resultTable': res
    }


@app.route('/sequence/explain', methods=['POST'])
def explain():
    content = request.get_json()
    if 'seq' not in content or 'target' not in content or 'class' not in content or 'pos' not in content or 'path' not in content:
        return 'BAD REQUEST', 400
    else:
        return {
            'path': explain_model(content['seq'], content['pos'], content['class'], content['target'], content['path'])
        }


@app.route('/sequence/serve-file')
def serve():
    if not request.args.get('path'):
        return 'NOT FOUND', 404
    else:
        return send_file(request.args.get('path'), mimetype='image/gif')
