from flask import Flask
import util
app = Flask(__name__)

@app.route("/classify_image", methods = ['GET','POST'])
def classify_image():
    return "Please find all names of the leaders"

# create a main method for a python flask server that runs on port 5000
if __name__ == "__main__":
    app.run(port=5000)
