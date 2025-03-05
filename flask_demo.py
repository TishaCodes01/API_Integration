from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'  # Return a response instead of just printing

if __name__ == "__main__":
    app.run(port=8000, debug=True)
