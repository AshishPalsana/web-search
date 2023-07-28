"""Python Flask WebApp Auth0 integration example
"""

import traceback
from os import environ as env

from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from flask import Flask, redirect, render_template, session, request, jsonify

import auth0
from auth0.authentication import GetToken, Database

from flask_sqlalchemy import SQLAlchemy
from jinja2 import constants
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, UserMixin, LoginManager, current_user

from OF_WebSearch.search_model.search_model import audio_similarity_search

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

app = Flask(__name__)
app.secret_key = env.get("APP_SECRET_KEY")

AUTH0_DOMAIN = env.get("AUTH0_DOMAIN")
CLIENT_ID = env.get("AUTH0_CLIENT_ID")
CLIENT_SECRET = env.get("AUTH0_CLIENT_SECRET")

login_manager = LoginManager()
login_manager.init_app(app)

login_manager.login_view = 'signInPage'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///D:/WebSearch/OF_WebSearch/user-data.db'
db = SQLAlchemy(app)


# User model
class User(UserMixin, db.Model):
    last_login = db.Column(db.DateTime, nullable=True)

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def is_active(self):
        # Assuming all users are active. If you have a way of deactivating users, you can reflect that here
        return True


oauth = OAuth(app)

oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid profile email",
    },
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration',
)

# Database instance for signup
database = Database(AUTH0_DOMAIN, CLIENT_ID)

# GetToken instance for login
token_helper = GetToken(AUTH0_DOMAIN, CLIENT_ID, client_secret=CLIENT_SECRET)

with app.app_context():
    db.create_all()


# Controllers API
@app.route("/")
def home():
    if not session:
        return redirect('/signInPage')
    else:
        return redirect('/searchPage')


@app.route("/searchPage")
def openSearchPage():
    return render_template(
        "searchPage.html",
        session=session
    )


@app.route("/resultsPage")
def openResultsPage():
    return render_template(
        "resultsPage.html",
    )


@app.route("/signInPage")
def openSignInPage():
    return render_template(
        "signInPage.html",
    )


@app.route("/signUpPage")
def openSignUpPage():
    return render_template(
        "signupPage.html",
    )


@login_manager.user_loader
def load_user(user_id):
    return session.get(User, int(user_id))


@app.route('/signup', methods=['POST'])
def signup():
    # Get data from form
    try:
        data = request.get_json()
        email = data['email']
        password = data['password']
        user = User.query.filter_by(email=email).first()  # check if user already exists
        if user != None:
            return {"message":'Email already exists'}, 400
        else:
            try:
                new_user = User(email=email)
                new_user.set_password(password)
                database.signup(email=email, password=password, connection='Username-Password-Authentication')
                db.session.add(new_user)
                db.session.commit()
                login_user(new_user)
                session['email'] = current_user.email
                # return redirect('/signInPage')
                return {'message':'Success'}, 200
            except auth0.exceptions.Auth0Error as e:
                return {"error":e.message+'dsfjhkbvsdajhfdsfjakhgh'}, 400
            except:
                return {'error':'Server Error'}, 400
    except:
        return {"error":'Error'}, 400

@app.route('/login', methods=['POST'])
def login():
    # return 'Login successful', 200
    try:
        # Get data from form
        data = request.get_json()
        email = data['email']
        password = data['password']
        # return {"status":200}
        remember = True if request.form.get('remember') else False
        with app.app_context():
            user = User.query.filter_by(email=email).first()
            if user != None:
                if check_password_hash(user.password, password):
                    try:
                        login_user(user, remember=remember)

                        realm = 'Username-Password-Authentication'  # default Auth0 DB connection
                        print("LOGIN3")
                        try:
                            response = token_helper.login(username=email, password=password, realm=realm)
                            print(response)
                            session[email] = {}
                            session[email]['token'] = response['access_token']
                            # Update session with user status
                            # session['user'] = True
                            # session['email'] = email

                            return {"message":"success"}, 200
                        except auth0.exceptions.RateLimitError as e:
                            return {"error":e.message}, 400
                        except auth0.exceptions.Auth0Error as e:
                            return {"error":e.message}, 400
                        except:
                            return {"error":'Internal Server Error'}, 400
                    except auth0.exceptions as e:
                        return {"error": e.message}, 400
                else:
                    return {"error":'Incorrect Password'}, 400
            else:
                return {"error":'Incorrect Email / SignUp'}, 400
    except:
        print(traceback.print_exc())
        return {"error":'Error'}, 400


@app.route("/logout")
def logout():
    # Clear entire session
    session.clear()
    return redirect('/signInPage')


@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    user_info = oauth.auth0.parse_id_token(token)
    session["user"] = user_info
    return redirect("/")


@app.route("/auth0callback", methods=["GET", "POST"])
def auth0_callback():
    token = oauth.auth0.authorize_access_token()
    user_info = oauth.auth0.userinfo_compliance(token)
    session["user"] = user_info
    return redirect("/")


@app.route("/callback", methods=["GET", "POST"])
def callback_handling():
    auth0.authorize_access_token()
    resp = auth0.get('userinfo')
    userinfo = resp.json()

    session[constants.JWT_PAYLOAD] = userinfo
    session[constants.PROFILE_KEY] = {
        'user_id': userinfo['sub'],
        'name': userinfo['name'],
        'picture': userinfo['picture']
    }
    return redirect('/dashboard')

@app.route("/upload", methods=["Post"])
def upload():
    try:
        print(request)
        file = request.files['file']
        audio_path = "tmp/temp_audio.wav"
        file.save(audio_path)
        audio_list = audio_similarity_search(audio_path)
        return  {"data":audio_list}, 200
    except:
        return {"error":traceback.print_exc()},400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=env.get("PORT", 3003), debug=True)
