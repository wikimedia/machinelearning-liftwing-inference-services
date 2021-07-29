import requests

HOST = "<redacted>"
USERNAME = "<redacted>"
PASSWORD = "<redacted>"

session = requests.Session()
response = session.get(HOST, verify=False)
headers = {
    "Content-Type": "application/x-www-form-urlencoded",
}

data = {"login": USERNAME, "password": PASSWORD}
session.post(response.url, headers=headers, data=data)
session_cookie = session.cookies.get_dict()["authservice_session"]

print(session_cookie)
