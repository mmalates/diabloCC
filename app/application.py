from flask import Flask
from flask import render_template, flash, redirect, url_for, session, jsonify
from flask_oauth import OAuth
import requests
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# print a nice greeting.


# def say_hello(username="World"):
#     return '<p>Hello %s!</p>\n' % username
#
#
# # some bits of text for the page.
# header_text = '''
#     <html>\n<head> <title>EB Flask Test</title> </head>\n<body>'''
# instructions = '''
#     <p><em>Hint</em>: This is a RESTful web service! Append a username
#     to the URL (for example: <code>/Thelonious</code>) to say hello to
#     someone specific.</p>\n'''
# home_link = '<p><a href="/">Back</a></p>\n'
# footer_text = '</body>\n</html>'

# EB looks for an 'application' callable by default.
application = Flask(__name__)

user_info = "test"
user_photo_url = 'https://demo.keypasco.com/res-1.2.2/img/User_ring.png'


@application.route('/')
@application.route('/landing')
def landing():
    global user_info, user_photo_url
    if user_info is not "":
        return render_template("landing.html",
                               title='Landing',
                               user=user_info,
                               user_photo=user_photo_url,
                               forecast_data={},
                               tenday_fc=[],
                               prediction=[])
    else:
        return render_template("landing.html",
                               title='Landing',
                               user_photo='https://demo.keypasco.com/res-1.2.2/img/User_ring.png',
                               forecast_data={})

# add a rule for the index page.
# application.add_url_rule('/', 'index', (lambda: header_text +
#                                         say_hello() + instructions + footer_text))


# add a rule when the page is accessed with a name appended to the site
# URL.
# application.add_url_rule('/<username>', 'hello', (lambda username:
#                                                   header_text + say_hello(username) + home_link + footer_text))
#

@application.route('/model/', methods=['POST'])
def model():
    url240 = 'https://api.weather.com/v1/geocode/37.82616/-121.980217/forecast/hourly/240hour.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e'

    response240 = requests.get(url240)
    jsoned240 = response240.json()
    forecast240 = jsoned240['forecasts']
    keys = []
    keys = forecast240[0].keys()
    matrix = []
    for item in forecast240:
        list = []
        for key, value in item.iteritems():
            list.append(value)
        matrix.append(list)
    df240 = pd.DataFrame(matrix, columns=keys)
    df240['fcst_valid_local'] = pd.to_datetime(df240['fcst_valid_local'])
    df240['date'] = df240.fcst_valid_local.dt.date
    averaged = df240.groupby(by='date').mean()
    averaged.head()
    averaged.columns
    df240 = averaged[['vis', 'dewpt', 'mslp']]
    df240['date'] = df240.index
    df240['date'] = pd.to_datetime(df240['date'])
    df240['date'] = df240.date.dt.strftime('%m-%d %a')
    print df240
    url = 'https://api.weather.com/v1/geocode/37.821373/-121.968079/forecast/daily/10day.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&units=e'

    response = requests.get(url)
    jsoned = response.json()
    tenday = []
    for item in jsoned['forecasts']:
        if 'day' in item.keys():
            for key, value in item.iteritems():
                if key == 'qpf':
                    prec = value
                if key == 'min_temp':
                    temp_low = value
                if key == 'fcst_valid_local':
                    date = value.split('T')[0]
                if key == 'day':
                    for keyday, valday in value.iteritems():
                        if keyday == 'wspd':
                            wind = valday
                        if keyday == 'temp':
                            temp_high = valday
                        if keyday == 'rh':
                            humid = valday
            # dow = pd.to_datetime(date,'%Y-%m-%d')
            tenday.append([date, temp_high, temp_low, humid, wind, prec])

    data = pd.DataFrame(tenday, columns=[
                        'date', 'temp_high', 'temp_low', 'hum_avg', 'wind_avg', 'prec'])

    data['date'] = pd.to_datetime(data['date'])
    data['DOW'] = data.date.dt.dayofweek
    data['DOY'] = data.date.dt.dayofyear
    data['date'] = data.date.dt.strftime('%m-%d %a')
    data = data.merge(df240, how='left', on='date')
    data['vis_avg'] = data['vis']
    data['dew_avg'] = data['dewpt']
    data['sea_avg'] = data['mslp']

    def dateparse(x): return pd.datetime.strptime(x, '%Y-%m-%d')
    df = pd.read_csv('data/train_clean.csv',
                     parse_dates=['date'], date_parser=dateparse)

    features = ['DOW', 'DOY', 'temp_high', 'temp_low',
                'hum_avg', 'wind_avg', 'prec', 'vis_avg', 'sea_avg', 'dew_avg']

    X = df[features]
    y = df['rounds']
    data['prediction'] = RF_model(X, y, data).astype(int)
    return render_template('model.html', title='Ten Day Forecast', tenday_fc=data.values, user=user_info, forecast_data={}, user_photo=user_photo_url, prediction=data.prediction.values)


def RF_model(X, y, forecast):
    rf = RandomForestRegressor(n_estimators=50, bootstrap=False,
                               max_depth=8, max_features='log2', random_state=123, n_jobs=4)
    rf.fit(X, y)
    return rf.predict(forecast[['DOW', 'DOY', 'temp_high', 'temp_low', 'hum_avg', 'wind_avg', 'prec', 'sea_avg', 'dew_avg', 'vis_avg']])

#
# GOOGLE_CLIENT_ID = '619170425013-r2ge51f3mv49ml5o0grla2qqcp1fq6da.apps.googleusercontent.com'
#
# GOOGLE_CLIENT_SECRET = 'oGMtg3by2h_EqOWTPPbVe1jY'
# # one of the Redirect URIs from Google APIs console
# REDIRECT_URI = '/oauth2callback'
#
# SECRET_KEY = 'development key'
# DEBUG = True
#
# app.debug = DEBUG
# app.secret_key = SECRET_KEY
# oauth = OAuth()
#
# google = oauth.remote_app('google',
#                           base_url='https://www.google.com/accounts/',
#                           authorize_url='https://accounts.google.com/o/oauth2/auth',
#                           request_token_url=None,
#                           request_token_params={
#                               'scope': 'https://www.googleapis.com/auth/userinfo.email', 'response_type': 'code'},
#                           access_token_url='https://accounts.google.com/o/oauth2/token',
#                           access_token_method='POST',
#                           access_token_params={
#                               'grant_type': 'authorization_code'},
#                           consumer_key=GOOGLE_CLIENT_ID,
#                           consumer_secret=GOOGLE_CLIENT_SECRET)
#
#
# @application.route('/index')
# def index():
#     access_token = session.get('access_token')
#     if access_token is None:
#         return redirect(url_for('login'))
#
#     access_token = access_token[0]
#     from urllib2 import Request, urlopen, URLError
#
#     headers = {'Authorization': 'OAuth ' + access_token}
#     req = Request('https://www.googleapis.com/oauth2/v1/userinfo',
#                   None, headers)
#     try:
#         res = urlopen(req)
#     except URLError, e:
#         if e.code == 401:
#             # Unauthorized - bad token
#             session.pop('access_token', None)
#             return redirect(url_for('login'))
#         return res.read()
#
#     global user_info, user_photo_url
#     match = re.search(r'(http.*.jpg)', res.read())
#     user_photo_url = match.group(0)
#
#     try:
#         res = urlopen(req)
#     except URLError, e:
#         if e.code == 401:
#             # Unauthorized - bad token
#             session.pop('access_token', None)
#             return redirect(url_for('login'))
#         return res.read()
#
#     match = re.search(r'[\w\.-]+@[\w\.-]+', res.read())
#     user_info = match.group(0)
#     return redirect('/landing')
#
#
# @application.route('/login')
# def login():
#     callback = url_for('authorized', _external=True)
#     return google.authorize(callback=callback)
#
#
# @application.route('/logout')
# def logout():
#     global user_info
#     user_info = ""
#     session.pop('access_token')
#     return redirect('/landing')
#
#
# @application.route(REDIRECT_URI)
# @google.authorized_handler
# def authorized(resp):
#     access_token = resp['access_token']
#     session['access_token'] = access_token, ''
#     return redirect(url_for('index'))
#
#
# @google.tokengetter
# def get_access_token():
#     return session.get('access_token')


# run the app.
# DEBUG = True
# application.debug = DEBUG

if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.run()
