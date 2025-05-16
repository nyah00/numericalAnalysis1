from flask import Flask, request, jsonify, render_template
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
x = sp.symbols('x')

def suggest_g_functions(f_expr):
    """Generate possible g(x) candidates from f(x) = 0."""
    try:
        f = sp.sympify(f_expr)
        g_functions = []

        # Common rearrangements for fixed-point iteration
        candidates = [
            ("Basic rearrangement: x = x - f(x)", x - f),
            ("Newton-like: x = x - f(x)/f'(x)", x - f / sp.diff(f, x)),
            ("Additive form: x = x + f(x)", x + f),
            ("Fractional form: x = x / (1 + f(x))", x / (1 + f)),
            ("Exponential form: x = exp(log(f(x) + x))", sp.exp(sp.log(f + x))),
        ]

        for name, g in candidates:
            try:
                simplified = sp.simplify(g)
                if x in simplified.free_symbols:  # Ensure g(x) still depends on x
                    g_functions.append({
                        "name": name,
                        "expression": str(simplified)
                    })
            except:
                continue

        return g_functions
    except:
        return []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/suggest_g", methods=["POST"])
def suggest_g():
    """API endpoint to suggest g(x) functions."""
    data = request.json
    f_expr = data["equation"]
    return jsonify({"g_functions": suggest_g_functions(f_expr)})

@app.route("/solve", methods=["POST"])
def solve():
    data = request.json
    f_expr = data["equation"]
    g_expr = data["g_function"]
    x0 = float(data["x0"])
    tol = float(data["tolerance"])
    max_iter = int(data["max_iter"])

    try:
        f = sp.sympify(f_expr)
        g = sp.sympify(g_expr)
        g_func = sp.lambdify(x, g, 'numpy')

        iterations = []
        x_values = []
        g_values = []
        errors = []

        x_prev = x0
        for i in range(max_iter):
            try:
                x_next = float(g_func(x_prev))
            except:
                return jsonify({
                    "status": "error",
                    "message": f"g(x) evaluation failed at x={x_prev:.4f}. Try a different g(x)."
                })

            error = abs(x_next - x_prev)
            iterations.append(i + 1)
            x_values.append(x_prev)
            g_values.append(x_next)
            errors.append(error)

            if error < tol:
                break
            x_prev = x_next

        # Generate plot
        fig, ax = plt.subplots()
        ax.plot(iterations, x_values, 'b-', label='x')
        ax.plot(iterations, g_values, 'r--', label='g(x)')
        ax.plot(iterations, errors, 'g:', label='Error')
        ax.legend()

        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return jsonify({
            "status": "success",
            "iterations": iterations,
            "x_values": x_values,
            "g_values": g_values,
            "errors": errors,
            "plot": plot_data,
            "solution": x_prev
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)