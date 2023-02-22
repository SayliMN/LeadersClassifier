from flask import Flask, request, jsonify
import util

app = Flask(__name__)


@app.route("/classify_image", methods=['GET', 'POST'])
def classify_image():

    # creating a request image object
    image_data = request.form['image_data']
    response = jsonify(util.classify_image(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


# create a main method for a python flask server that runs on port 5000
if __name__ == "__main__":
    print("Starting python flask server for classification of leaders images")
    util.load_saved_artifacts()
    app.run(port=5000)
