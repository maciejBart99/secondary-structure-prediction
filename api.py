from flask import Flask, request, send_file
from capt import explain_model, get_details
from feature import feature_apply
from test import main
import numpy as np
from capt import data as d

app = Flask('explain-model')


classification_mode = 'q8'


@app.route('/sequence/switch', methods=['POST'])
def switch():
    global classification_mode
    content = request.get_json()
    classification_mode = content['classification']
    return {
        'status': 'ok'
    }


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
    if 'seq' not in content or 'target' not in content or 'class' not in content or 'pos' not in content or 'path' not in content or 'encodeFeatures' not in content:
        return 'BAD REQUEST', 400
    else:
        data, r_l, l_l = explain_model(content['seq'], content['class'], content['target'], content['path'], content['boundaries'], classification_mode)
        if content['encodeFeatures']:
            sh = data.shape
            data = data.reshape((1, sh[0], sh[1]))
            data = feature_apply(data, np.array([sh[1]]), d[:, :, l_l:r_l])
            data = data.reshape((16, sh[1]))
        target = []
        data = data * 1000_000
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
    if request.args.get('encodeFeatures') == 'true':
        sh = data.shape
        data = data.transpose((1,0)).reshape((1, sh[1], sh[0]))
        data = feature_apply(data, np.array([sh[0]]), data)
        data = data.reshape((16, sh[0])).transpose((1,0))
    for i in range(data.shape[0]):
        semi_res = []
        for j in range(data.shape[1]):
            semi_res.append(float(data[i, j]))
        result.append(semi_res)
    return {
        'featureProfile': result,
        'centralItem': central
    }
