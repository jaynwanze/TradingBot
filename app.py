from flask import Flask, render_template
from tradingBot import runTradingBot  # Import the function from tradingBot.py

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run_script", methods=["POST"])
def run_script():
    try:
        # Retrieve the cryptocurrency name and symbol from the form
        # cryptocurrency_name = request.form["cryptocurrency_name"]
        # cryptocurrency_symbol_pair = request.form["cryptocurrency_symbol_pair"]

        # Run trading bot and save script output
        script_output = runTradingBot()

        # render output from  as html page
        return render_template("result.html", script_output=script_output)
    except Exception as e:
        return render_template({"error": f"Script execution failed: {str(e)}"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
