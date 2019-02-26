from flask import Flask, render_template, request
from recommend import recommend

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('login.html')


@app.route('/search/')
def search():
    query = request.args.get('query')
    dic = recommend(query)

    return render_template('search.html', DATA=dic)


if __name__ == "__main__":
    app.run(debug=True)
