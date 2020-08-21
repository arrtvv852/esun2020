from src.BertNERExtractor import *
from src.Utils import *
from flask import Flask, request
from flask import render_template

app = Flask(__name__)

#Type 1
@app.route("/", methods=['GET'])

def index():
    content = request.args.get('content')
    result = extract(content)
    return '{}\n'.format(result)

port = 5001
app.debug = True

print("Bert NER Server listening... port:{}".format(port))
app.run(host="0.0.0.0", port=port)
