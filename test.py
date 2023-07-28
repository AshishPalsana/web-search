# app.py
from flask import Flask, request, redirect, url_for, session, jsonify
from authlib.integrations.flask_client import OAuth
import os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "my-secret-key")

# Auth0 configuration
AUTH0_CLIENT_ID = "YOUR_AUTH0_CLIENT_ID"
AUTH0_CLIENT_SECRET = "YOUR_AUTH0_CLIENT_SECRET"
AUTH0_DOMAIN = "YOUR_AUTH0_DOMAIN"
AUTH0_CALLBACK_URL = "YOUR_AUTH0_CALLBACK_URL"
AUTH0_BASE_URL = f"https://{AUTH0_DOMAIN}"
AUTH0_ACCESS_TOKEN_URL = f"{AUTH0_BASE_URL}/oauth/token"
AUTH0_AUTHORIZE_URL = f"{AUTH0_BASE_URL}/authorize"
AUTH0_CLIENT_KWARGS = {"scope": "openid profile email", "audience": f"{AUTH0_BASE_URL}/userinfo"}

# Authlib OAuth object
oauth = OAuth()
auth0 = oauth.register(
    "auth0",
    client_id=AUTH0_CLIENT_ID,
    client_secret=AUTH0_CLIENT_SECRET,
    authorize_url=AUTH0_AUTHORIZE_URL,
    access_token_url=AUTH0_ACCESS_TOKEN_URL,
    client_kwargs=AUTH0_CLIENT_KWARGS,
)


@app.route("/")
def index():
    user_info = session.get("user_info")
    return f"Welcome{' ' + user_info['name'] if user_info else ''}!"


@app.route("/login")
def login():
    redirect_uri = url_for("callback", _external=True)
    return auth0.authorize_redirect(redirect_uri)


@app.route("/callback")
def callback():
    auth0.authorize_access_token()
    user_info = auth0.parse_id_token(auth0.fetch_id_token())
    session["user_info"] = user_info
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.pop("user_info", None)
    return redirect(url_for("index"))


@app.route("/activity")
def activity():
    user_info = session.get("user_info")
    if user_info:
        # Perform some activity/fun for authenticated users
        return jsonify({"message": f"Hello, {user_info['name']}! You performed an activity."})
    else:
        return jsonify({"message": "You need to log in to perform the activity."})


if __name__ == "__main__":
    app.run(debug=True)
