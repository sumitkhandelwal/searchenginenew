# from flask import  request, Flask, jsonify
# import tfidfnlpcode as tf
# # creating a Flask app
# app = Flask(__name__)
#
# @app.route('/search', methods=['GET'])
# def search():
#     try :
#         args = request.args
#         data = args.get('texttosearch')
#         reply = tf.searchnlp(str(data))
#         return jsonify({'Status Code ': 200, 'Text to Search': data, 'Request Id' : reply})
#     except Exception as e:
#         return jsonify({'Status Code ': 400,'Error': e})
#
# if __name__ == '__main__':
#     app.run(debug=True)
#

from flask import Flask
UPLOAD_FOLDER = 'upload'
app = Flask(__name__)
#app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024