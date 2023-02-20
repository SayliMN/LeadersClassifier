from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/classify_image", methods=['GET', 'POST'])
def classify_image():
    return "Hello"


# create a main method for a python flask server that runs on port 5000
if __name__ == "__main__":
    app.run(port=5000)
