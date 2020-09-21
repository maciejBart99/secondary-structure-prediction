from flask import Flask, request, send_file
from capt import explain_model, get_details
from test import main

app = Flask('explain-model')


classification_mode = 'q8'


@app.route('/sequence/switch')
def switch():
    global classification_mode
    content = request.get_json()
    classification_mode = content['classification']
    return None


@app.route('/sequence/result')
def result():
    loss, acc, res = main('model_q8.pth', False, classification_mode)

    return {
        'loss': loss,
        'acc': acc,
        'resultTable': res,
        'mode': classification_mode
    }


@app.route('/sequence/explain', methods=['POST'])
def explain():
    content = request.get_json()
    if 'seq' not in content or 'target' not in content or 'class' not in content or 'pos' not in content or 'path' not in content:
        return 'BAD REQUEST', 400
    else:
        data = explain_model(content['seq'], content['class'], content['target'], content['path'], content['boundaries'])
        target = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                target.append({
                    'structuralClass': i,
                    'relativePosition': j,
                    'value': data[i, j]
                })
        return {'data': target} , 200


@app.route('/sequence/serve-file')
def serve():
    if not request.args.get('path'):
        return 'NOT FOUND', 404
    else:
        return send_file(request.args.get('path'), mimetype='image/gif')


@app.route('/sequence/details/<int:details>')
def det(details: int):
    data, central = get_details(details, int(request.args.get('pos')), int(request.args.get('boundaries')))
    result = []
    for i in range(data.shape[0]):
        semi_res = []
        for j in range(data.shape[1]):
            semi_res.append(float(data[i, j]))
        result.append(semi_res)
    return {
        'featureProfile': result,
        'centralItem': central
    }
