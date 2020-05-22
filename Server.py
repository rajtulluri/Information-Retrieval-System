from flask import Flask
from Information_retrieval_system import information_system

app = Flask(__name__)

@app.route('/<query>')
def querying(query):
	indices = information_system(query)
	return indices

if __name__ == '__main__':
	app.run(host='0.0.0.0')