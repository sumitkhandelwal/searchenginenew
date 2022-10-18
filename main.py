import os
import urllib.request
from app import app
from flask import Flask, request, redirect, jsonify
from werkzeug.utils import secure_filename
import wordnetcode as tf
from os import path

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/file-upload', methods=['POST'])
def upload_file():
	# check if the post request has the file part
	if 'file' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message' : 'No file selected for uploading'})
		resp.status_code = 400
		return resp
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		resp = jsonify({'message' : 'File successfully uploaded'})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify({'message' : 'Allowed file type is .csv'})
		resp.status_code = 400
		return resp

@app.route('/search', methods=['GET'])
def search():
	try :
		args = request.args
		data = args.get('texttosearch')
		reply, score  = tf.searchnlp(str(data))
		return jsonify({'StatusCode ': 200, 'TextToSearch': data, 'RequestId' : reply, 'Score':score})
	except Exception as e:
		print(e)
		return jsonify({'StatusCode ': 400,'RequestId' : 'NULL', 'Score':-1})

@app.route('/', methods = ['GET', 'POST'])
def root():
	stringmsg = 'Application Running'
	return stringmsg

if __name__ == "__main__":
	app.run(debug=True)
