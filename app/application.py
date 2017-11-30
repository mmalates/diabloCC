from flask import Flask
from utils import Predictor
from flask import render_template

application = Flask(__name__)


@application.route('/')
def landing():
    return render_template("base.html")


@application.route('/model/', methods=['GET'])
def model():
    predictions = Predictor()
    data = predictions.get_forecast()
    data = predictions.processing(data)
    return predictions.get_predictions(data)


if __name__ == "__main__":
    application.run()
